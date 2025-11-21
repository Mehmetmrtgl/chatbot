from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from scripts.db_utils import save_llm_answer_to_db
import nltk
from nltk.tokenize import sent_tokenize
import os
from langchain_huggingface import HuggingFaceEmbeddings  # SBERT için gerekliyse bırak

# -----------------------------------------------------
# 📦 Ortak Ayarlar
# -----------------------------------------------------
nltk.download('punkt', quiet=True)

OFFLOAD_PATH = os.path.abspath("offload")
os.makedirs(OFFLOAD_PATH, exist_ok=True)

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

ROOT_DIR = Path(__file__).resolve().parent.parent

# -----------------------------------------------------
# 🦙 LLaMA3 (MERGED) MODELİ
# -----------------------------------------------------
LLAMA_MERGED_PATH = ROOT_DIR / "models" / "llama3_merged"

llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MERGED_PATH, local_files_only=True)
if llama_tokenizer.pad_token is None:
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

llama_model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MERGED_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    local_files_only=True,
    offload_folder=OFFLOAD_PATH,
    low_cpu_mem_usage=True
).eval()

print("✅ LLaMA3 (merged) başarıyla yüklendi.")

# -----------------------------------------------------
# 🧠 DEEPSEEK (MERGED) MODELİ
# -----------------------------------------------------
DEEPSEEK_PATH = ROOT_DIR / "models" / "deepseek_merged"
DEEPSEEK_OFFLOAD = os.path.join(DEEPSEEK_PATH, "offload")
os.makedirs(DEEPSEEK_OFFLOAD, exist_ok=True)

deepseek_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_PATH, local_files_only=True)
if deepseek_tokenizer.pad_token is None:
    deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token

deepseek_model = AutoModelForCausalLM.from_pretrained(
    DEEPSEEK_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    local_files_only=True,
    offload_folder=DEEPSEEK_OFFLOAD,
    low_cpu_mem_usage=True
).eval()

print("✅ DeepSeek (merged) başarıyla yüklendi.")

# -----------------------------------------------------
# ✂️ KISA CEVAP FONKSİYONU
# -----------------------------------------------------
def shorten_answer(answer: str, max_sentences: int = 3) -> str:
    sentences = sent_tokenize(answer)
    return " ".join(sentences[:max_sentences]).strip()

# -----------------------------------------------------
# 💬 CEVAP ÜRETİCİ (GENEL)
# -----------------------------------------------------
def generate_answer(prompt: str, original_question: str = None, model_name: str = "llama") -> str:
    if model_name == "llama":
        model = llama_model
        tokenizer = llama_tokenizer
    elif model_name == "deepseek":
        model = deepseek_model
        tokenizer = deepseek_tokenizer
    else:
        raise ValueError("❌ Geçersiz model adı! 'llama' veya 'deepseek' olmalı.")

    if not prompt.strip():
        return "⚠️ Boş bir istek girdiniz."

    # 🔹 Instruction yerine sade, örnek tabanlı prompt
    final_prompt = f"Soru: {prompt.strip()}\nCevap:"

    inputs = tokenizer(
        final_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 🧹 Gereksiz önekleri temizle
    for prefix in ["Yanıt:", "Cevap:", "Answer:", "Response:", "?", "？"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    answer_limited = shorten_answer(answer)

    if original_question and len(answer_limited) > 5:
        print(f"📥 [{model_name}] Kaydediliyor: {original_question} → {answer_limited}")
        save_llm_answer_to_db(original_question, answer_limited)

    return answer_limited



# -----------------------------------------------------
# 🚀 MODEL TESTİ (manuel çalıştırma için)
# -----------------------------------------------------
if __name__ == "__main__":
    print("🧠 Model yükleme testi başlatılıyor...")

    # LLaMA kontrolü
    try:
        print(f"📦 LLaMA cihaz: {next(llama_model.parameters()).device}")
    except Exception as e:
        print(f"❌ LLaMA yüklenemedi: {e}")

    # DeepSeek kontrolü
    try:
        print(f"📦 DeepSeek cihaz: {next(deepseek_model.parameters()).device}")
    except Exception as e:
        print(f"❌ DeepSeek yüklenemedi: {e}")

    print("🚀 Tüm modeller başarıyla yüklenmiş görünüyor!")
