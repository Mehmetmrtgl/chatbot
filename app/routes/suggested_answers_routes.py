# app/routes/suggestion_routes.py

from flask import Blueprint, jsonify
import json
import random
import os

suggestion_bp = Blueprint("suggestion_bp", __name__)
from flask import request
from scripts.faiss_utils import get_similar_questions
from app.chat.utils import normalize_input

@suggestion_bp.route("/api/suggested_questions", methods=["POST"])
def dynamic_suggestions():
    try:
        # 1. İstekten gelen son soru alınır
        data = request.get_json()
        last_question = data.get("last_question", "")
        normalized = normalize_input(last_question)

        # 2. Benzer sorular FAISS ile çekilir
        faiss_suggestions = get_similar_questions(normalized)[:2] if normalized else []

        # 3. Eğitim verisinden rastgele 3 soru ekle
        file_path = os.path.join("data", "finetune", "so.jsonl")
        with open(file_path, "r", encoding="utf-8") as f:
            prompts = [json.loads(line)["prompt"] for line in f if "prompt" in json.loads(line)]

        random_questions = random.sample(prompts, min(3, len(prompts)))
        combined = list(dict.fromkeys(faiss_suggestions + random_questions))[:5]

        return jsonify({"questions": combined})

    except Exception as e:
        print("❌ Hata /api/suggested_questions:", e)
        return jsonify({"questions": []})
