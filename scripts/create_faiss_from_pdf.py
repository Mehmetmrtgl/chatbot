import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

PDF_FOLDER = "../data/pdfs"
INDEX_FOLDER = "../data/embeddings/faiss_index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def load_pdfs_and_split():
    documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, filename)
            raw_text = extract_text_from_pdf(path)
            metadata = {"source": filename}

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.create_documents([raw_text], metadatas=[metadata])
            documents.extend(chunks)
    return documents

def create_faiss_index():
    print("📂 PDF'ler yükleniyor ve bölünüyor...")
    docs = load_pdfs_and_split()

    print(f"🧠 {len(docs)} metin parçası bulundu. Embedding başlatılıyor...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(INDEX_FOLDER)
    print(f"✅ FAISS dizini '{INDEX_FOLDER}' klasörüne kaydedildi.")

if __name__ == "__main__":
    create_faiss_index()
