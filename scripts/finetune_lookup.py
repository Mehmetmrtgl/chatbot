# scripts/finetune_lookup.py

import json
from scripts.utils import normalize_input  # normalize_input fonksiyonu burada tanımlı

# 🔹 Fine-tuning dosyaları (hem train hem valid)
FINETUNE_FILES = [
    "./data/finetune/train.jsonl",
    "./data/finetune/valid.jsonl"
]

# 🔹 Sözlük: normalize edilmiş soru → cevap
fine_tune_dict = {}

for path in FINETUNE_FILES:
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue  # geçersiz JSON satırlarını atla

                # 🔸 Format 1: Eski format (prompt/completion)
                if "prompt" in item and "completion" in item:
                    norm_q = normalize_input(item["prompt"])
                    fine_tune_dict[norm_q] = item["completion"].strip()
                    continue

                # 🔸 Format 2: Yeni format (messages)
                if "messages" in item and isinstance(item["messages"], list):
                    user_msg = next(
                        (m["content"] for m in item["messages"] if m.get("role") == "user"),
                        None
                    )
                    assistant_msg = next(
                        (m["content"] for m in item["messages"] if m.get("role") == "assistant"),
                        None
                    )
                    if user_msg and assistant_msg:
                        norm_q = normalize_input(user_msg)
                        fine_tune_dict[norm_q] = assistant_msg.strip()
    except FileNotFoundError:
        print(f"❌ Dosya bulunamadı: {path}")
    except Exception as e:
        print(f"⚠️ Hata ({path}): {e}")

# 🔹 Lookup fonksiyonları
def is_in_finetune_data(normalized_input: str) -> bool:
    """Verilen normalize edilmiş soru fine-tune datasında var mı?"""
    return normalized_input in fine_tune_dict

def get_finetune_reference_answer(normalized_input: str) -> str:
    """Fine-tune datasındaki referans cevabı döndürür."""
    return fine_tune_dict.get(normalized_input, "")
