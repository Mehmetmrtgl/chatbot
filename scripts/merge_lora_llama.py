from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os, gc

BASE_PATH = "../models/llama3"
LORA_PATH = "../models/llama3_finetuned/checkpoint-1284"
OUTPUT_PATH = "../models/llama3_merged"

print("🚀 Model yükleniyor...")

# ⚙️ GPU belleğini verimli kullanmak için bfloat16 tercih edilir (RTX 40 serisi)
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16

# ⚠️ Merge işlemi için tüm ağırlıkları tam olarak yükle
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    torch_dtype=DTYPE,
    device_map=None,          # 🚫 accelerate offload yok, full yükleme
    low_cpu_mem_usage=False,  # ✅ Tüm katmanlar belleğe alınır
    local_files_only=True
)

# 🧠 LoRA ağırlıklarını yükle
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
