# app/routes/autocomplete_routes.py
from flask import Blueprint, request, jsonify
from scripts.faiss_utils import get_similar_questions

autocomplete_bp = Blueprint("autocomplete", __name__)

@autocomplete_bp.route("/api/autocomplete", methods=["POST"])
def autocomplete():
    data = request.get_json()
    prefix = data.get("prefix", "")
    suggestions = get_similar_questions(prefix)[:5]  # İlk 5 öneri
    return jsonify({"suggestions": suggestions})
