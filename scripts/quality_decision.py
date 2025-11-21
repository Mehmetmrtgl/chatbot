# scripts/quality_decision.py
def evaluate_answer_quality(llm_score: float, sim_score: float, weight_llm=0.5, weight_sim=0.5, threshold=0.75):
    """
    İki kalite skorunu ağırlıklı ortalama ile birleştirir.
    """
    final_score = (llm_score * weight_llm) + (sim_score * weight_sim)

    if final_score >= threshold:
        return True, final_score  # Kaliteli kabul edildi
    else:
        return False, final_score  # DeepSeek'e geç
