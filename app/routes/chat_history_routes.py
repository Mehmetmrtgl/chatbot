# app/routes/chat_history_routes.py

from flask import Blueprint, jsonify
from app.db.models import ChatLog

chat_history_bp = Blueprint("chat_history", __name__)

@chat_history_bp.route("/api/chat_sessions", methods=["GET"])
def get_chat_sessions():
    sessions = ChatLog.query.with_entities(ChatLog.session_id).distinct().all()
    session_ids = [s[0] for s in sessions]
    return jsonify({"sessions": session_ids})

@chat_history_bp.route("/api/chat_history/<session_id>", methods=["GET"])
def get_chat_history(session_id):
    logs = ChatLog.query.filter_by(session_id=session_id).order_by(ChatLog.timestamp.asc()).all()
    result = [
        {
            "role": log.role,
            "message": log.message,
            "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for log in logs
    ]
    return jsonify({"history": result})
