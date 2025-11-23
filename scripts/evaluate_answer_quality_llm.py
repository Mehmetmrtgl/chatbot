# scripts/evaluate_answer_quality_llm.py (Ollama versiyon)
from scripts.model_inference import generate_answer

def evaluate_answer_quality_llm(question: str, answer: str) -> float:
    prompt = f"""
Aşağıda bir soru ve bu soruya verilen cevap yer almaktadır. Lütfen bu cevabın kalitesini değerlendir:

Soru: {question}

Cevap: {answer}

Bu cevabın kalitesini 0 ile 1 arasında puanla.
Sadece bir sayı döndür.
"""
    raw = generate_answer(prompt, model_name="llama")  # artık "llama" da Qwen’e gidiyor
    try:
        score = float(raw.strip().split()[-1].replace(",", "."))
        return max(0.0, min(1.0, score))
    except:
        return 0.0
