from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Question(db.Model):
    __tablename__ = "questions"
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False, unique=True)  # ✅ doğru isim
    answer = db.Column(db.Text)
    is_approved = db.Column(db.Boolean, default=False)
    source = db.Column(db.String(20), default="manual")  # default doğru
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    model_quality_score = db.Column(db.Integer)

from datetime import datetime

class Feedback(db.Model):
    __tablename__ = "feedback"
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey("questions.id", ondelete="CASCADE"))
    feedback_type = db.Column(db.String)  # ← bu alan eksikse hata alırsın
    session_id = db.Column(db.String)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatLog(db.Model):
    __tablename__ = "chat_logs"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    role = db.Column(db.String, nullable=False)  # "user" veya "bot"
    message = db.Column(db.Text, nullable=False)
    session_id = db.Column(db.String, nullable=False)  # Kullanıcıyı ayırt etmek için

    def __repr__(self):
        return f"<ChatLog {self.timestamp} {self.role}: {self.message[:30]}>"
