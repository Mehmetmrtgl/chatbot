# scripts/model_inference_deepseek.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize import sent_tokenize
import nltk

# -----------------------------------------------------
# 🧠 MODEL YOLLARI
# -----------------------------------------------------
MERGED_MODEL_PATH = "../models/deepseek_merged"
OFFLOAD_PATH = os.path.join(MERGED_MODEL_PATH, "offload")
os.makedirs(OFFLOAD_PATH, exist_ok=True)  # 💾 klasör garantileniyor

# -----------------------------------------------------
# 🔤 TOKENIZER AYARLARI
# -----------------------------------------------------
nltk.download("punkt", quiet=True)

tokenizer_ds = AutoTokenizer.from_pretrained(
    MERGED_MODEL_PATH,
    local_files_only=True
)
if tokenizer_ds.pad_token is None:
    tokenizer_ds.pad_token = tokenizer_ds.eos_token

# -----------------------------------------------------
# ⚙️ MODEL YÜKLEME (MERGED)
# -----------------------------------------------------
print("🔹 DeepSeek (Merged) modeli yükleniyor...")

model_ds = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
    offload_folder=OFFLOAD_PATH,
    low_cpu_mem_usage=True
).eval()

print("✅ DeepSeek merged model başarıyla yüklendi.")

# -----------------------------------------------------
# ✂️ YANITI KISALTMA FONKSİYONU
# -----------------------------------------------------
def shorten_answer(answer: str, max_sentences: int = 3) -> str:
    """Yanıtı belirli sayıda cümleyle sınırla."""
    sentences = sent_tokenize(answer)
    return " ".join(sentences[:max_sentences]).strip()

# -----------------------------------------------------
# 💬 CEVAP ÜRETİMİ
# -----------------------------------------------------
def generate_answer_deepseek(prompt: str) -> str:
    """DeepSeek merged modeliyle yanıt üretir."""
    torch.cuda.empty_cache()

    if not prompt.strip():
        return "⚠️ Boş bir istek girdiniz."

    inputs = tokenizer_ds(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model_ds.device)

    with torch.no_grad():
        outputs = model_ds.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer_ds.eos_token_id
        )

    output_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer_ds.decode(output_ids, skip_special_tokens=True).strip()

    return shorten_answer(answer)
