from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from uuid import uuid4
from openai import OpenAI

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chat.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

client = OpenAI()

# ---------- MODELOS ----------
class Conversation(db.Model):
    id = db.Column(db.String, primary_key=True)  # UUID como string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    conversation_id = db.Column(db.String, db.ForeignKey("conversation.id"), index=True, nullable=False)
    role = db.Column(db.String, nullable=False)         # 'user' o 'assistant' o 'system'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Crear tablas la primera vez
with app.app_context():
    db.create_all()

# ---------- RUTAS ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    conversation_id = data.get("conversation_id")

    if not user_message:
        return jsonify({"error": "Mensaje vacío"}), 400

    # Si no llegó conversation_id, cree uno nuevo
    if not conversation_id:
        conversation_id = str(uuid4())
        db.session.add(Conversation(id=conversation_id))
        db.session.commit()

    # Construir el contexto con los últimos N mensajes (p. ej. 20)
    history = Message.query.filter_by(conversation_id=conversation_id)\
                           .order_by(Message.created_at.asc())\
                           .all()
    messages = [{"role": "system", "content": "Eres un asistente colombiano, hablas con tono relajado y natural, usando expresiones como 'mano', 'bro' o 'parcero', pero sin exagerar. Das respuestas cortas y útiles."}]
    for m in history[-20:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_message})

    # Guardar el mensaje del usuario
    db.session.add(Message(conversation_id=conversation_id, role="user", content=user_message))
    db.session.commit()

    # Llamar al modelo
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    reply = response.choices[0].message.content

    # Guardar la respuesta del bot
    db.session.add(Message(conversation_id=conversation_id, role="assistant", content=reply))
    db.session.commit()

    return jsonify({"reply": reply, "conversation_id": conversation_id})

@app.route("/history", methods=["GET"])
def history():
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "Falta conversation_id"}), 400
    msgs = Message.query.filter_by(conversation_id=conversation_id)\
                        .order_by(Message.created_at.asc())\
                        .all()
    return jsonify([
        {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
        for m in msgs
    ])

if __name__ == "__main__":
    app.run(debug=True)