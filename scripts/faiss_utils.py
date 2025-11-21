# scripts/faiss_utils.py

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "../data/embeddings/autocomplete_index/index.faiss")
TEXTS_PATH = os.path.join(BASE_DIR, "../data/embeddings/autocomplete_index/texts.json")

embedder = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_PATH)

with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    all_questions = json.load(f)

def get_similar_questions(query, top_k=5):
    print(f"🔍 FAISS autocomplete sorgusu: {query}")
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)
    return [all_questions[i] for i in I[0] if i < len(all_questions)]
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import numpy as np

sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_similar_questions_hybrid(user_question, all_questions, faiss_top_k=5, bm25_top_k=5):
    faiss_results = get_similar_questions(user_question, top_k=faiss_top_k)

    tokenized_corpus = [q.split() for q in all_questions]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(user_question.split())
    top_bm25 = np.argsort(bm25_scores)[::-1][:bm25_top_k]
    bm25_results = [all_questions[i] for i in top_bm25]

    # Birleştir ve SBERT ile sırala
    combined = list(set(faiss_results + bm25_results))
    query_emb = sbert_model.encode(user_question, convert_to_tensor=True)
    corpus_emb = sbert_model.encode(combined, convert_to_tensor=True)
    sim_scores = util.cos_sim(query_emb, corpus_emb)[0]

    results = sorted(zip(combined, sim_scores.tolist()), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:5]]
# ✅ Global index ve questions listesi
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(TEXTS_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
else:
    index = faiss.IndexFlatL2(384)  # 384 → MiniLM çıktısı
    all_questions = []

def add_to_faiss_index(new_questions):
    global all_questions, index

    # Filtrele: zaten olanları ekleme
    new_questions = [q for q in new_questions if q not in all_questions]
    if not new_questions:
        print("🔁 Eklenecek yeni soru yok.")
        return

    print(f"➕ FAISS'e {len(new_questions)} yeni soru ekleniyor.")
    embeddings = embedder.encode(new_questions)
    index.add(np.array(embeddings).astype("float32"))
    all_questions.extend(new_questions)

    # Kaydet
    faiss.write_index(index, INDEX_PATH)
    with open(TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print("✅ FAISS index ve metin listesi güncellendi.")