import os
import json
from flask import Blueprint, request, jsonify, send_from_directory, Response
import re
from app.chat.utils import answer_user_question
chat_bp = Blueprint("chat", __name__)


SURROGATE_RE = re.compile(u'[\ud800-\udfff]')
PDF_FOLDER_PATH = os.path.join(os.getcwd(), "data", "pdfs")

def strip_surrogates(s: str) -> str:
    if not isinstance(s, str):
        return s
    return SURROGATE_RE.sub('', s)

@chat_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}


    user_question = data.get("question") or data.get("message") or ""
    user_question = user_question.strip()
    session_id = data.get("session_id", "default_user")
    chat_history = data.get("chat_history", [])

    if not user_question:
        return jsonify({"error": "Soru boş olamaz."}), 400

    try:
        result = answer_user_question(user_question, session_id, chat_history)


        for key in ("answer", "message"):
            if key in result and isinstance(result[key], str):
                result[key] = strip_surrogates(result[key])

        return jsonify(result)

    except Exception as e:
        print("HATA:", str(e))  # Burada emoji kullanma
        return jsonify({"error": "Sunucuda bir hata oluştu."}), 500
        

@chat_bp.route('/download/<path:filename>')
def download_file(filename):

    full_path = os.path.join(PDF_FOLDER_PATH, filename)
    
    if not os.path.exists(full_path):
        print(f"UYARI: İstenen dosya bulunamadı: {full_path}")
        return jsonify({"error": "Dosya sunucuda bulunamadı."}), 404

    return send_from_directory(PDF_FOLDER_PATH, filename, as_attachment=True)
