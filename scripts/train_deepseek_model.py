# --- train_deepseek_model.py (düzenlenmiş) ---
import os
# Bellek parçalanmasını azalt
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import time
import gc
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers.trainer_callback import EarlyStoppingCallback

# SDPA/flash attention kapat (Windows + 4bit için daha stabil)
from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)

# Opsiyonel: matmul hassasiyeti
torch.set_float32_matmul_precision("high")

print("✅ CUDA kullanılabilir mi?:", torch.cuda.is_available())
print("🖥️ GPU adı:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "YOK")

MODEL_PATH = "../models/deepseek"
SAVE_PATH  = "../models/deepseek_finetuned"
TRAIN_FILE = "../data/finetune/train.jsonl"
VALID_FILE = "../data/finetune/valid.jsonl"

print("📁 Model klasörü içeriği:", os.listdir(MODEL_PATH)[:10])

MAX_LEN = 384  # VRAM tepe kullanımını düşürmek için 512 -> 384

def main():
    print("🧠 Tokenizer ve model yükleniyor...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4-bit quant
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    # use_cache kapalı (eğitimde gerekli)
    base_model.config.use_cache = False
    # Bazı sürümlerde mevcutsa: base_model.config.attn_implementation = "eager"

    # LoRA
    base_model = prepare_model_for_kbit_training(base_model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(base_model, lora_cfg)
    print("📦 Model cihazı:", next(model.parameters()).device)

    # Veri
    print("📚 Veri yükleniyor...")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VALID_FILE})

    def tokenize(batch):
        texts = []
        for raw in batch["messages"]:
            msgs = json.loads(raw) if isinstance(raw, str) else raw
            system_msg = user_msg = assistant_msg = ""
            for m in msgs:
                role = m.get("role", "")
                if role == "system":
                    system_msg = m.get("content", "").strip()
                elif role == "user":
                    user_msg = m.get("content", "").strip()
                elif role == "assistant":
                    assistant_msg = m.get("content", "").strip()
            text = f"{system_msg}\n### Soru:\n{user_msg}\n\n### Cevap:\n{assistant_msg}"
            texts.append(text)

        return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["messages"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Eğitim ayarları
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        per_device_train_batch_size=2,            # Hâlâ crash olursa 1 yapın
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{SAVE_PATH}/logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        fp16=False,
        bf16=True,                                # RTX 40xx için
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=False,             # Windows’ta stabilite için KAPALI
        eval_accumulation_steps=16,               # Eval sırasında VRAM tepesini düşür
        dataloader_pin_memory=False               # Windows’ta bazen daha stabil
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"].select(range(50)),  # GPU yükünü azalt
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    try:
        print("🚀 Fine-tuning başlıyor...")
        start = time.time()
        trainer.train()
        print("⏱️ Eğitim süresi:", round(time.time() - start, 2), "saniye")

        print(f"✅ Eğitim tamamlandı. Model kaydediliyor → {SAVE_PATH}")
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)

    except Exception as e:
        print(f"❌ Eğitim sırasında hata oluştu: {e}")

    finally:
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 GPU belleği temizlendi ve işlem güvenli şekilde sonlandırıldı.")

if __name__ == "__main__":
    main()
