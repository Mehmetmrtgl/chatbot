# scripts/evaluate_answer_quality_llm.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# LLaMA'yı kullanarak değerlendirme
EVAL_MODEL_PATH = "models/llama3"  # Aynı LLaMA kullanılabilir

tokenizer = AutoTokenizer.from_pretrained(EVAL_MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(EVAL_MODEL_PATH, device_map="auto", torch_dtype=torch.float32, local_files_only=True).eval()

def evaluate_answer_quality_llm(question: str, answer: str) -> float:
    prompt = f"""
Aşağıda bir soru ve bu soruya verilen cevap yer almaktadır. Lütfen bu cevabın kalitesini değerlendir:

Soru: {question}

Cevap: {answer}

Bu cevabın kalitesini 0 ile 1 arasında puanla.
Sadece bir sayı döndür.
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        score = float(decoded.strip().split()[-1].replace(",", "."))
        return max(0.0, min(1.0, score))
    except:
        return 0.0
