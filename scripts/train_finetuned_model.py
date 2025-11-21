import os
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

print("✅ CUDA kullanılabilir mi?:", torch.cuda.is_available())
print("🖥️ GPU adı:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "YOK")

MODEL_PATH = "../models/llama3"
SAVE_PATH = "../models/llama3_finetuned"
TRAIN_FILE = "../data/finetune/train.jsonl"
VALID_FILE = "../data/finetune/valid.jsonl"

print("📁 Model klasörü içeriği:", os.listdir(MODEL_PATH)[:10])

def main():
    # ---------------- Tokenizer ---------------- #
    print("🧠 Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- Model (4-bit) ---------------- #
    print("🧩 4-bit model yükleniyor...")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # RTX 40 serisinde daha stabil
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_cfg,
        device_map="auto",  # ✅ yeni sürümde gerekli
        low_cpu_mem_usage=True,  # ✅ device_map ile zorunlu
        torch_dtype=torch.bfloat16,  # RTX 4080 için stabil
        local_files_only=True
    )

    base_model = prepare_model_for_kbit_training(base_model)
    base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    # ---------------- LoRA ---------------- #
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(base_model, lora_cfg)
    print("📦 Model cihazı:", next(model.parameters()).device)

    # ---------------- Dataset ---------------- #
    print("📚 Veri yükleniyor...")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VALID_FILE})

    def tokenize(batch):
        texts = []
        for msgs in batch["messages"]:
            system_msg = ""
            user_msg = ""
            assistant_msg = ""
            for m in msgs:
                if m["role"] == "system":
                    system_msg = m["content"]
                elif m["role"] == "user":
                    user_msg = m["content"]
                elif m["role"] == "assistant":
                    assistant_msg = m["content"]

            # Chat formatını metne dönüştür
            text = f"{system_msg}\nUser: {user_msg}\nAssistant: {assistant_msg}"
            texts.append(text)

        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["messages"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---------------- Training ---------------- #
    training_args = TrainingArguments(
        output_dir=SAVE_PATH,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",
        logging_dir=f"{SAVE_PATH}/logs",
    )

    print("🚀 Fine-tuning başlıyor...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    start = time.time()
    trainer.train()
    print("⏱️ Eğitim süresi:", round(time.time() - start, 2), "saniye")

    # ---------------- Save ---------------- #
    print(f"✅ Eğitim tamamlandı. Model kaydediliyor → {SAVE_PATH}")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

if __name__ == "__main__":
    main()
