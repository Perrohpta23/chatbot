import os
import logging
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from openai import OpenAI
from sqlalchemy import event
from sqlalchemy.engine import Engine
from werkzeug.exceptions import RequestEntityTooLarge

# ================== Config básica ==================
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)

BASEDIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASEDIR, "chat.db")
DB_URI = "sqlite:///" + DB_PATH

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.urandom(32))
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024  # 256 KB
app.config["SQLALCHEMY_DATABASE_URI"] = DB_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# CORS: en producción pon orígenes específicos, no "*"
CORS(app, resources={r"/chat": {"origins": os.getenv("CORS_ORIGINS", "*")}})

db = SQLAlchemy(app)

# En SQLite, activa FK = ON
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, _):
    try:
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()
    except Exception:
        pass

# OpenAI
client = OpenAI(api_key=(os.getenv("OPENAI_API_KEY") or "").strip())
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ================== Modelos ==================
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.String, primary_key=True)  # UUID guardado en cookie
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class Conversation(db.Model):
    __tablename__ = "conversation"
    id = db.Column(db.String, primary_key=True)  # UUID
    user_id = db.Column(db.String, db.ForeignKey("user.id", ondelete="CASCADE"), index=True, nullable=False)
    title = db.Column(db.String, default="Nuevo chat")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    messages = db.relationship(
        "Message",
        cascade="all, delete-orphan",
        backref="conversation",
        passive_deletes=False,
    )


class Message(db.Model):
    __tablename__ = "message"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    conversation_id = db.Column(
        db.String, db.ForeignKey("conversation.id", ondelete="CASCADE"),
        index=True, nullable=False
    )
    role = db.Column(db.String, nullable=False)  # 'user' | 'assistant' | 'system'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


with app.app_context():
    db.create_all()

# ================== Helpers ==================
SYSTEM_PROMPT = (
    "Eres un asistente carismático. Das respuestas útiles. "
    "Cuando el usuario escriba matemáticas, responde en Markdown usando LaTeX: "
    "inline \\( … \\) y en bloque $$ … $$."
    "Si usas emojies, vas variandolos con cada respuesta si te parece adecuado. "
)

MAX_MSG_LEN = 4000  # aproximación simple

def clamp_text(s: str, n: int) -> str:
    s = s or ""
    s = s.strip()
    return s[:n] if len(s) > n else s

def get_or_set_user(resp=None):
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid4())
        db.session.add(User(id=user_id))
        db.session.commit()
        if resp is None:
            resp = make_response()
        resp.set_cookie(
            "user_id", user_id,
            max_age=60 * 60 * 24 * 365 * 2,  # 2 años
            httponly=True,
            samesite="Lax",
            secure=(os.getenv("COOKIE_SECURE", "0") == "1")
        )
    else:
        if not User.query.get(user_id):
            db.session.add(User(id=user_id))
            db.session.commit()
    return user_id, resp

# ================== Rutas ==================
@app.route("/")
def home():
    resp = make_response(render_template("index.html"))
    _, resp = get_or_set_user(resp)
    return resp

@app.route("/conversations", methods=["GET"])
def list_conversations():
    user_id, _ = get_or_set_user()
    convs = Conversation.query.filter_by(user_id=user_id) \
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
    # Parche seguro por si el esquema fue creado sin CASCADE alguna vez
    Message.query.filter_by(conversation_id=conv.id).delete(synchronize_session=False)
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
    user_message = clamp_text(data.get("message"), MAX_MSG_LEN)
    conversation_id = data.get("conversation_id")
    if not user_message:
        return jsonify({"error": "Mensaje vacío"}), 400

    # Crear o cargar conversación
    if not conversation_id:
        conv = Conversation(id=str(uuid4()), user_id=user_id, title="Nuevo chat")
        db.session.add(conv)
        db.session.commit()
        conversation_id = conv.id
    else:
        conv = Conversation.query.filter_by(id=conversation_id, user_id=user_id).first_or_404()

    # Cargar historial (últimos 20)
    history_msgs = Message.query.filter_by(conversation_id=conversation_id) \
        .order_by(Message.created_at.asc()).all()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history_msgs[-20:]:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_message})

    # Guardar mensaje usuario
    db.session.add(Message(conversation_id=conversation_id, role="user", content=user_message))
    db.session.commit()

    # LLM
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.4,
        )
        reply = response.choices[0].message.content
    except RequestEntityTooLarge:
        app.logger.exception("Payload demasiado grande")
        return jsonify({"error": "Mensaje demasiado grande."}), 413
    except Exception:
        app.logger.exception("Fallo en LLM")
        return jsonify({"error": "La IA no está disponible por el momento."}), 503

    # Guardar respuesta y actualizar
    db.session.add(Message(conversation_id=conversation_id, role="assistant", content=reply))
    conv.updated_at = datetime.now(timezone.utc)
    if conv.title == "Nuevo chat":
        preview = (user_message[:30] + "…") if len(user_message) > 30 else user_message
        conv.title = preview or "Chat"
    db.session.commit()

    return jsonify({"reply": reply, "conversation_id": conversation_id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")