from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_PATH = "models/llama3_finetuned"  # Eğitimden sonra oluşan klasör

def load_model():
    print("📦 Model yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

def main():
    generator = load_model()
    print("\n🧠 Fine-tuned LLaMA 3 Chatbot'a Hoş Geldiniz!")
    print("✏️ Soru sor. Çıkmak için 'çık', 'exit' veya 'q' yaz.\n")

    while True:
        user_input = input("👤 Soru: ")
        if user_input.strip().lower() in ["çık", "exit", "q"]:
            print("👋 Görüşmek üzere!")
            break

        prompt = f"{user_input}\nCevap:"
        result = generator(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)[0]["generated_text"]

        # "Cevap:" etiketinden sonrasını al
        answer = result.split("Cevap:")[-1].strip()
        print(f"\n🤖 Cevap: {answer}\n")

if __name__ == "__main__":
    main()
