# app/routes/feedback_routes.py

from flask import Blueprint, request, jsonify
from app.db.models import db, Feedback, Question  # Question modelini de kontrol için al
from datetime import datetime

feedback_bp = Blueprint("feedback", __name__)

@feedback_bp.route("/api/feedback", methods=["POST"])
def store_feedback():
    data = request.get_json()
    question_id = data.get("question_id")
    is_liked = data.get("is_liked")  # true / false bekleniyor

    if question_id is None or is_liked is None:
        return jsonify({"error": "Eksik veri: question_id veya is_liked boş olamaz."}), 400

    question = Question.query.get(question_id)
    if not question:
        return jsonify({"error": "Belirtilen question_id veritabanında bulunamadı."}), 404

    # ✅ feedback_type = "like" / "dislike"
    feedback_type = "like" if is_liked else "dislike"

    new_feedback = Feedback(
        question_id=question_id,
        feedback_type=feedback_type,
        session_id=data.get("session_id", "anonymous")
    )
    db.session.add(new_feedback)
    db.session.commit()

    return jsonify({"status": "ok"}), 200
