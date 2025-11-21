import os
import json

from flask import Blueprint, request, jsonify
from app.db.models import db, Question,Feedback
admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/api/unapproved_questions", methods=["GET"])
def get_unapproved_questions():
    questions = Question.query.filter_by(is_approved=False).all()
    result = [{"id": q.id, "question": q.question, "answer": q.answer or ""} for q in questions]
    return jsonify({"questions": result})


@admin_bp.route("/api/approve_answer", methods=["POST"])
def approve_answer():
    data = request.get_json()
    question_id = data.get("question_id")
    new_answer = data.get("answer", "").strip()

    if not question_id or not new_answer:
        return jsonify({"error": "Eksik bilgi"}), 400

    question = Question.query.get(question_id)
    if not question:
        return jsonify({"error": "Soru bulunamadı"}), 404

    question.answer = new_answer
    question.is_approved = True
    db.session.commit()

    return jsonify({"status": "ok"}), 200
# admin_routes.py
@admin_bp.route("/api/feedback_logs", methods=["GET"])
def get_feedback_logs():
    feedbacks = (
        db.session.query(
            Feedback.id,
            Feedback.feedback_type,
            Feedback.created_at.label("timestamp"),
            Feedback.session_id,
            Question.question.label("question_text"),
            Question.answer.label("answer_text")
        )
        .join(Question, Feedback.question_id == Question.id)
        .order_by(Feedback.created_at.desc())
        .all()
    )

    result = []
    for fb in feedbacks:
        result.append({
            "id": fb.id,
            "feedback_type": fb.feedback_type,
            "timestamp": fb.timestamp,
            "session_id": fb.session_id,
            "question_text": fb.question_text,
            "answer_text": fb.answer_text,
        })

    return jsonify({"feedback": result})



# app/routes/admin_routes.py

from flask import jsonify
from sqlalchemy.sql import func
from app.db.models import ChatLog, Feedback, Question
from datetime import datetime, timedelta

@admin_bp.route("/api/stats", methods=["GET"])
def get_statistics():
    now = datetime.utcnow()
    start_of_day = datetime(now.year, now.month, now.day)
    start_of_week = now - timedelta(days=7)

    # Günlük toplam mesaj sayısı
    daily_messages = ChatLog.query.filter(ChatLog.timestamp >= start_of_day).count()

    # Haftalık toplam mesaj sayısı
    weekly_messages = ChatLog.query.filter(ChatLog.timestamp >= start_of_week).count()

    # Toplam onay bekleyen soru
    unapproved_count = Question.query.filter_by(is_approved=False).count()

    # Toplam 👍 ve 👎 sayısı
    total_likes = Feedback.query.filter_by(feedback_type="like").count()
    total_dislikes = Feedback.query.filter_by(feedback_type="dislike").count()

    # En çok beğenilen 5 cevap
    top_answers = (
        db.session.query(Question.answer, func.count(Feedback.id).label("likes"))
        .join(Feedback, Feedback.question_id == Question.id)
        .filter(Feedback.feedback_type == "like")
        .group_by(Question.answer)
        .order_by(func.count(Feedback.id).desc())
        .limit(5)
        .all()
    )

    return jsonify({
        "daily_messages": daily_messages,
        "weekly_messages": weekly_messages,
        "unapproved_count": unapproved_count,
        "total_likes": total_likes,
        "total_dislikes": total_dislikes,
        "top_answers": [{"answer": a[0], "likes": a[1]} for a in top_answers]
    })
@admin_bp.route("/api/analytics", methods=["GET"])
def get_analytics_data():
    total_messages = db.session.query(ChatLog).count()
    unanswered = db.session.query(Question).filter_by(is_approved=False).count()

    feedback_likes = db.session.query(Feedback).filter_by(feedback_type="like").count()
    feedback_dislikes = db.session.query(Feedback).filter_by(feedback_type="dislike").count()

    from sqlalchemy import extract, func
    hourly_distribution = db.session.query(
        func.extract('hour', ChatLog.timestamp).label('hour'),
        func.count().label('count')
    ).group_by('hour').order_by('hour').all()

    hourly_data = [{"hour": int(h), "count": c} for h, c in hourly_distribution]

    return jsonify({
        "total_messages": total_messages,
        "unanswered": unanswered,
        "likes": feedback_likes,
        "dislikes": feedback_dislikes,
        "hourly_distribution": hourly_data
    })


# admin_routes.py
@admin_bp.route("/api/questions_with_answers", methods=["GET"])
def get_questions_with_answers():
    questions = Question.query.filter(Question.is_approved == True).union(
        Question.query.filter(Question.source == "llm")
    ).all()

    result = [
        {
            "id": q.id,
            "question": q.question,
            "answer": q.answer,
            "timestamp": q.created_at.strftime("%Y-%m-%d %H:%M:%S") if q.created_at else ""
        }
        for q in questions
    ]
    return jsonify({"questions": result})
# admin_routes.py
@admin_bp.route("/api/update_answer", methods=["POST"])
def update_answer():
    data = request.get_json()
    qid = data.get("question_id")
    new_answer = data.get("answer")

    question = Question.query.get(qid)
    if question:
        question.answer = new_answer
        db.session.commit()
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Soru bulunamadı"}), 404
@admin_bp.route("/api/set_quality_score", methods=["POST"])
def set_quality_score():
    data = request.get_json()
    question_id = data.get("question_id")
    score = data.get("score")

    if not (1 <= score <= 5):
        return jsonify({"error": "Skor 1-5 arasında olmalı."}), 400

    question = db.session.query(Question).filter_by(id=question_id).first()
    if question:
        question.model_quality_score = score
        db.session.commit()
        return jsonify({"status": "ok"})
    return jsonify({"error": "Soru bulunamadı."}), 404
# routes/admin_routes.py
@admin_bp.route("/api/export_finetune_data", methods=["POST"])
def export_finetune_data():
    approved_qas = Question.query.filter_by(is_approved=True).all()
    export_path = "fine_tuning_data.jsonl"

    if not approved_qas:
        return jsonify({"status": "no_data", "message": "Onaylı veri yok."})

    existing_lines = set()
    if os.path.exists(export_path):
        with open(export_path, "r", encoding="utf-8") as f:
            for line in f:
                existing_lines.add(line.strip())

    new_lines = []
    for q in approved_qas:
        prompt = q.question.strip()
        completion = q.answer.strip()
        json_line = json.dumps({"prompt": prompt, "completion": completion})
        if json_line not in existing_lines:
            new_lines.append(json_line)

    if new_lines:
        with open(export_path, "a", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")

    return jsonify({
        "status": "ok",
        "exported": len(new_lines),
        "total": len(approved_qas)
    })
# routes/admin_routes.py
from flask import send_file
import os

@admin_bp.route("/api/download_fine_tune_data", methods=["GET"])
def download_fine_tune_data():
    file_path = "data/fine_tuning_data.jsonl"  # Yolunu kendi dosya konumuna göre güncelle
    if not os.path.exists(file_path):
        return jsonify({"error": "Dosya bulunamadı"}), 404
    return send_file(file_path, as_attachment=True)
from scripts.model_inference import generate_answer

@admin_bp.route("/api/generate_alternative", methods=["POST"])
def generate_alternative_answer():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Soru eksik"}), 400

    prompt = f"Soru: {question}\nAlternatif yanıt:"
    new_answer = generate_answer(prompt)
    return jsonify({"alternative": new_answer})


# Basit örnek (güvenli olmayan, demo amaçlı)
@admin_bp.route("/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    # Gerçek uygulamada şifre hash'lenmiş şekilde veritabanından kontrol edilir
    if username == "admin" and password == "admin123":
        return jsonify({"status": "ok", "token": "fake-admin-token"})
    else:
        return jsonify({"status": "fail"}), 401
@admin_bp.route("/api/generate_alternative", methods=["POST"])
def generate_alternative():
    question = request.json["question"]
    alt = generate_answer(original_question=question)
    return jsonify({"alternative": alt})
@admin_bp.route("/api/reject_answer/<int:question_id>", methods=["POST"])
def reject_answer(question_id):
    question = Question.query.get(question_id)
    if not question:
        return jsonify({"error": "Soru bulunamadı"}), 404

    # Sil veya is_approved=False olarak güncelle
    db.session.delete(question)  # alternatif: question.is_approved = False
    db.session.commit()
    return jsonify({"success": True})
@admin_bp.route("/api/mark_for_edit", methods=["POST"])
def mark_for_edit():
    data = request.get_json()
    question_id = data.get("question_id")
    answer = data.get("answer", "")

    question = Question.query.get(question_id)
    if not question:
        return jsonify({"error": "Soru bulunamadı"}), 404

    question.answer = answer
    question.approved = False  # Cevap düzenleyici sayfasına düşmesi için
    db.session.commit()

    return jsonify({"success": True})