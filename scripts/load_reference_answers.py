# scripts/load_reference_answers.py
import json
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def load_reference_answers(jsonl_path="data/fine_tune/train.jsonl", top_n=1000):
    reference_texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "completion" in data:
                reference_texts.append(data["completion"].strip())

    embeddings = model.encode(reference_texts, convert_to_tensor=False, normalize_embeddings=True)
    return reference_texts, np.array(embeddings)
