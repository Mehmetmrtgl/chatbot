from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
from scripts.db_utils import save_llm_answer_to_db

# Ollama OpenAI-uyumlu client
from openai import OpenAI

# -----------------------------------------------------
# 📦 Ortak Ayarlar
# -----------------------------------------------------
# Download NLTK tokenizer data (punkt_tab is the newer version, punkt is legacy)
try:
    nltk.download('punkt_tab', quiet=True)
except:
    # Fallback to punkt if punkt_tab fails
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass  # Continue even if download fails

# Ollama server zaten çalışıyor (sende curl 200 dönüyor)
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# HF GGUF model tag’in (Ollama’da görünen isim)
DEFAULT_OLLAMA_MODEL = "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0"

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"  # herhangi bir string yeter
)

# Projede hâlâ "llama"/"deepseek" diye çağırıyorsun.
# İkisini de aynı hafif Qwen modeline yönlendiriyoruz.
MODEL_MAP = {
    "llama": DEFAULT_OLLAMA_MODEL,
    "deepseek": DEFAULT_OLLAMA_MODEL,
}


# -----------------------------------------------------
# ✂️ KISA CEVAP FONKSİYONU
# -----------------------------------------------------
def shorten_answer(answer: str, max_sentences: int = 3) -> str:
    sentences = sent_tokenize(answer)
    return " ".join(sentences[:max_sentences]).strip()

# -----------------------------------------------------
# 💬 CEVAP ÜRETİCİ (GENEL) - OLLAMA
# -----------------------------------------------------
def generate_answer(prompt: str, original_question: str = None, model_name: str = "llama") -> str:
    if not prompt.strip():
        return "⚠️ Boş bir istek girdiniz."

    ollama_model = MODEL_MAP.get(model_name, DEFAULT_OLLAMA_MODEL)

    # 🔹 Senin eski prompt formatını koruyorum
    final_prompt = f"Soru: {prompt.strip()}\nCevap:"

    resp = client.chat.completions.create(
        model=ollama_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": final_prompt},
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )

    answer = resp.choices[0].message.content.strip()

    # 🧹 Gereksiz önekleri temizle (eski kodla aynı)
    for prefix in ["Yanıt:", "Cevap:", "Answer:", "Response:", "?", "？"]:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()

    # answer_limited = shorten_answer(answer)
    answer_limited = answer

    if original_question and len(answer_limited) > 5:
        print(f"📥 [{model_name}] Kaydediliyor: {original_question} → {answer_limited}")
        save_llm_answer_to_db(original_question, answer_limited)

    return answer_limited


# -----------------------------------------------------
# 🚀 MODEL TESTİ (manuel çalıştırma için)
# -----------------------------------------------------
if __name__ == "__main__":
    print("🧠 Ollama bağlantı testi başlatılıyor...")

    try:
        test = generate_answer("Merhaba, kısaca kendini tanıt.", model_name="llama")
        print("✅ Test cevabı:", test)
        print("🚀 Ollama üzerinden model başarıyla çalışıyor!")
    except Exception as e:
        print("❌ Ollama test hatası:", e)
