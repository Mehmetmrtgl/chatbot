# app/routes/chat_routes.py
from flask import Blueprint, request, jsonify
from app.chat.utils import answer_user_question

chat_bp = Blueprint("chat", __name__)
"""
@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_question = data.get("question", "").strip()
    session_id = data.get("session_id", "default_user")
    chat_history = data.get("chat_history", [])
    if not user_question:
        return jsonify({"error": "Soru boş olamaz."}), 400

    result = answer_user_question(user_question,session_id,chat_history)
    return jsonify(result)
"""
# app/routes/chat_routes.py
from flask import Blueprint, request, Response
from app.chat.utils import answer_user_question
import json
# app/routes/chat_routes.py
from flask import Blueprint, request, jsonify
from app.chat.utils import answer_user_question
import re

chat_bp = Blueprint("chat", __name__)

# Lone surrogate karakterleri temizlemek için
SURROGATE_RE = re.compile(u'[\ud800-\udfff]')

def strip_surrogates(s: str) -> str:
    if not isinstance(s, str):
        return s
    return SURROGATE_RE.sub('', s)

@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}

    # Hem "question" hem "message" key'lerini destekle
    user_question = data.get("question") or data.get("message") or ""
    user_question = user_question.strip()
    session_id = data.get("session_id", "default_user")
    chat_history = data.get("chat_history", [])

    if not user_question:
        return jsonify({"error": "Soru boş olamaz."}), 400

    try:
        result = answer_user_question(user_question, session_id, chat_history)

        # Sadece metin alanlarını temizle
        for key in ("answer", "message"):
            if key in result and isinstance(result[key], str):
                result[key] = strip_surrogates(result[key])

        return jsonify(result)

    except Exception as e:
        print("HATA:", str(e))  # Burada emoji kullanma
        return jsonify({"error": "Sunucuda bir hata oluştu."}), 500
