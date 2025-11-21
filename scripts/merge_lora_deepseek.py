from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os, gc

BASE_PATH = "../models/deepseek"
LORA_PATH = "../models/deepseek_finetuned/checkpoint-642"
OUTPUT_PATH = "../models/deepseek_merged"

print("🚀 Model yükleniyor...")

# ⚠️ Merge sırasında device_map=None olmalı (tam yükleme yapar)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    torch_dtype=torch.float16,
    device_map=None,            # ✅ accelerate offload olmadan yükle
    low_cpu_mem_usage=False,    # ✅ tamamını belleğe yükle
    local_files_only=True
)

# LoRA ağırlıklarını yükle
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False,
    local_files_only=True
)

print("🔁 Merge işlemi başlatıldı...")
merged_model = model.merge_and_unload()

# 🧹 Bellek temizliği
del model, base_model
gc.collect()
torch.cuda.empty_cache()

print("💾 Model kaydediliyor...")
os.makedirs(OUTPUT_PATH, exist_ok=True)
merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=False)

tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"✅ Merge tamamlandı ve kaydedildi: {OUTPUT_PATH}")
