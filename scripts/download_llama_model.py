from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Hugging Face model adı
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


# Kaydetmek istediğin klasör
TARGET_PATH = "models/llama3"  # Bu klasör localde oluşacak

def download_model():

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set!")

    print("📥 Tokenizer indiriliyor...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token
    )

    print("📥 Model indiriliyor...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token
    )

    print(f"💾 Model {TARGET_PATH} klasörüne kaydediliyor...")
    tokenizer.save_pretrained(TARGET_PATH)
    model.save_pretrained(TARGET_PATH)
    print("✅ Model başarıyla indirildi ve kaydedildi.")

if __name__ == "__main__":
    os.makedirs(TARGET_PATH, exist_ok=True)
    download_model()
