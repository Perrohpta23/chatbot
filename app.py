import os
from flask import Flask, render_template, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from uuid import uuid4
from openai import OpenAI

BASEDIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASEDIR, "chat.db")
DB_URI = "sqlite:///" + DB_PATH

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DB_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

print(f"💾 Usando BD en: {DB_PATH}")

client = OpenAI()

class User(db.Model):
    id = db.Column(db.String, primary_key=True)  # UUID en cookie
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Conversation(db.Model):
    id = db.Column(db.String, primary_key=True)  # UUID
    user_id = db.Column(db.String, db.ForeignKey("user.id"), index=True, nullable=False)
    title = db.Column(db.String, default="Nuevo chat")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    conversation_id = db.Column(db.String, db.ForeignKey("conversation.id"), index=True, nullable=False)
    role = db.Column(db.String, nullable=False)  # 'user' | 'assistant' | 'system'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

with app.app_context():
    db.create_all()

# ---------- HELPERS ----------
SYSTEM_PROMPT = "Eres un asistente colombiano, hablas con tono relajado y natural, usando expresiones como 'mano', 'bro' o 'parcero', pero sin exagerar. Das respuestas cortas y útiles."

def get_or_set_user(resp=None):
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid4())
        db.session.add(User(id=user_id))
        db.session.commit()
        if resp is None:
            resp = make_response()
        resp.set_cookie("user_id", user_id, max_age=60*60*24*365*2)  # 2 años
    else:
        if not User.query.get(user_id):
            db.session.add(User(id=user_id))
            db.session.commit()
    return user_id, resp

# ---------- RUTAS ----------
@app.route("/")
def home():
    resp = make_response(render_template("index.html"))
    _, resp = get_or_set_user(resp)
    return resp

@app.route("/conversations", methods=["GET"])
def list_conversations():
    user_id, _ = get_or_set_user()
    convs = Conversation.query.filter_by(user_id=user_id)\
        .order_by(Conversation.updated_at.desc()).all()
    return jsonify([
        {"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()}
        for c in convs
    ])

@app.route("/conversations", methods=["POST"])
def create_conversation():
    user_id, _ = get_or_set_user()
    title = (request.json or {}).get("title") or "Nuevo chat"
    conv = Conversation(id=str(uuid4()), user_id=user_id, title=title)
    db.session.add(conv)
    db.session.commit()
    return jsonify({"id": conv.id, "title": conv.title})

@app.route("/conversations/<conv_id>", methods=["PATCH"])
def rename_conversation(conv_id):
    user_id, _ = get_or_set_user()
    conv = Conversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()
    new_title = (request.json or {}).get("title", "").strip() or conv.title
    conv.title = new_title
    conv.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify({"ok": True})

@app.route("/conversations/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    user_id, _ = get_or_set_user()
    conv = Conversation.query.filter_by(id=conv_id, user_id=user_id).first_or_404()
    Message.query.filter_by(conversation_id=conv.id).delete()
    db.session.delete(conv)
    db.session.commit()
    return jsonify({"ok": True})

@app.route("/history", methods=["GET"])
def history():
    user_id, _ = get_or_set_user()
    conversation_id = request.args.get("conversation_id")
    conv = Conversation.query.filter_by(id=conversation_id, user_id=user_id).first_or_404()
    msgs = Message.query.filter_by(conversation_id=conv.id).order_by(Message.created_at.asc()).all()
    return jsonify([
        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
        for m in msgs
    ])

@app.route("/chat", methods=["POST"])
def chat():
    user_id, _ = get_or_set_user()
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    conversation_id = data.get("conversation_id")
    if not user_message:
        return jsonify({"error": "Mensaje vacío"}), 400

    # Si no mandan conversation_id, cree una nueva
    if not conversation_id:
        conv = Conversation(id=str(uuid4()), user_id=user_id, title="Nuevo chat")
        db.session.add(conv)
        db.session.commit()
        conversation_id = conv.id
    else:
        conv = Conversation.query.filter_by(id=conversation_id, user_id=user_id).first_or_404()

    # Cargar últimos N mensajes
    history_msgs = Message.query.filter_by(conversation_id=conversation_id)\
                    .order_by(Message.created_at.asc()).all()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history_msgs[-20:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_message})

    # Guardar mensaje del usuario
    db.session.add(Message(conversation_id=conversation_id, role="user", content=user_message))
    db.session.commit()

    # LLM
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        reply = response.choices[0].message.content
    except Exception as e:
        reply = "La IA no está disponible en este momento. Intente de nuevo mas tarde."

    # Guardar respuesta
    db.session.add(Message(conversation_id=conversation_id, role="assistant", content=reply))
    conv.updated_at = datetime.now(timezone.utc)
    db.session.commit()

    # Primer título bonito (si sigue en "Nuevo chat")
    if conv.title == "Nuevo chat":
        preview = (user_message[:30] + "…") if len(user_message) > 30 else user_message
        conv.title = preview or "Chat"
        db.session.commit()

    return jsonify({"reply": reply, "conversation_id": conversation_id})
    
if __name__ == "__main__":
    app.run(debug=True)