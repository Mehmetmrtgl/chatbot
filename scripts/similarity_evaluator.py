# scripts/similarity_evaluator.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def calculate_similarity(user_answer: str, reference_embeddings: np.ndarray, reference_texts: list):
    embedding = model.encode(user_answer, convert_to_tensor=False, normalize_embeddings=True)
    scores = cosine_similarity([embedding], reference_embeddings)[0]
    best_score = float(np.max(scores))
    best_match = reference_texts[int(np.argmax(scores))]
    return best_score, best_match
