import json
import os
import random

SOURCE_PATH = "data/questions.jsonl"
OUTPUT_DIR = "data/finetune"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.jsonl")
VALID_PATH = os.path.join(OUTPUT_DIR, "valid.jsonl")

def prepare_finetune_data():
    if not os.path.exists(SOURCE_PATH):
        print(f"❌ {SOURCE_PATH} bulunamadı.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(SOURCE_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_data = []
    for line in lines:
        try:
            obj = json.loads(line)
            prompt = obj["prompt"].strip()
            completion = obj["completion"].strip()

            if prompt and completion:
                cleaned_data.append({
                    "prompt": prompt,
                    "completion": completion
                })
        except json.JSONDecodeError:
            print(f"⚠️ Hatalı JSON satırı atlandı:\n{line[:100]}...")

    if not cleaned_data:
        print("❌ Uygun veri bulunamadı.")
        return

    # Veriyi karıştır ve ayır
    random.shuffle(cleaned_data)
    split_idx = int(len(cleaned_data) * 0.8)
    train_set = cleaned_data[:split_idx]
    valid_set = cleaned_data[split_idx:]

    # JSONL olarak yaz
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VALID_PATH, "w", encoding="utf-8") as f:
        for item in valid_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Eğitim verisi: {len(train_set)} kayıt → {TRAIN_PATH}")
    print(f"✅ Doğrulama verisi: {len(valid_set)} kayıt → {VALID_PATH}")

if __name__ == "__main__":
    prepare_finetune_data()
