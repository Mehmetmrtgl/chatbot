# scripts/db_utils.py

from app.db.models import db, Question, ChatLog
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func


from app.db.models import Question
from sqlalchemy import func

from sqlalchemy import func

def get_answer_from_db(user_question):
    result = (
        Question.query
        .filter(Question.is_approved == True)
        .filter(func.similarity(func.lower(Question.question), func.lower(user_question)) > 0.6)
        .order_by(func.similarity(func.lower(Question.question), func.lower(user_question)).desc())
        .first()
    )
    if result:
        return {"id": result.id, "answer": result.answer}
    return None




def get_chat_history(session_id: str, limit: int = 5):
    logs = (
        ChatLog.query
        .filter_by(session_id=session_id)
        .order_by(ChatLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [{"role": log.role, "text": log.message} for log in reversed(logs)]


def save_llm_answer_to_db(question_text, answer_text):
    try:
        new_q = Question(
            question=question_text.strip(),
            answer=answer_text.strip(),
            is_approved=False,
            source="llm"
        )
        db.session.add(new_q)
        db.session.commit()
        print("✅ Kaydedildi:", new_q.id)
        return new_q.id
    except IntegrityError:
        db.session.rollback()
        print("⚠️ Zaten kayıtlı:", question_text)
        existing = Question.query.filter_by(question=question_text.strip()).first()
        return existing.id if existing else None
    except Exception as e:
        db.session.rollback()
        print("❌ DB kayıt hatası:", e)
        return None
def get_last_question_id_by_text(question_text):
    existing = Question.query.filter_by(question=question_text.strip()).first()
    return existing.id if existing else None

from app.db.models import Question

def get_all_questions():
    return [q.question for q in Question.query.all()]
