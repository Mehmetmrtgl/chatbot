# scripts/generate_questions_from_pdfs.py

import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from model_inference import generate_answer

PDF_FOLDER = "../data/pdfs"
OUTPUT_PATH = "../data/faiss/questions.json"

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def load_and_split_texts():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(os.path.join(PDF_FOLDER, filename))
            chunks = splitter.split_text(raw_text)
            all_chunks.extend(chunks)

    return all_chunks

def generate_questions():
    chunks = load_and_split_texts()
    qa_pairs = []

    print(f"🔍 {len(chunks)} metin parçası bulundu. Soru üretimi başlatılıyor...\n")
    for chunk in tqdm(chunks, desc="Soru Üretimi"):
        prompt = f"Aşağıdaki metne göre bir soru üret:\n\n{chunk}\n\nSoru:"
        question = generate_answer(prompt)
        if question:
            qa_pairs.append({"question": question, "context": chunk})

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(qa_pairs)} soru üretildi ve '{OUTPUT_PATH}' dosyasına kaydedildi.")

if __name__ == "__main__":
    generate_questions()
