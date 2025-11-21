from sentence_transformers import SentenceTransformer, util

sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def compare_similarity_with_reference(generated: str, reference: str) -> float:
    emb_gen = sbert_model.encode(generated, convert_to_tensor=True)
    emb_ref = sbert_model.encode(reference, convert_to_tensor=True)
    score = util.cos_sim(emb_gen, emb_ref).item()
    return score
