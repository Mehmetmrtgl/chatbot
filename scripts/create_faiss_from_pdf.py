import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER = os.path.join(BASE_DIR, "../data/pdfs") 
INDEX_FOLDER = os.path.join(BASE_DIR, "../data/embeddings/faiss_index")

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text() + "\n"
    except Exception as e:
        print(f"❌ Hata ({pdf_path}): {e}")
    return text


def load_pdfs_and_split():

    documents = []
    
    if not os.path.exists(PDF_FOLDER):
        print(f"⚠️ PDF Klasörü bulunamadı: {PDF_FOLDER}")
        print("   Lütfen 'data' klasörünün içine 'pdfs' klasörü açıp dosyaları oraya koyun.")
        os.makedirs(PDF_FOLDER, exist_ok=True)
        return []

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    print(f"📄 Bulunan PDF Sayısı: {len(pdf_files)}")

    if not pdf_files:
        print("⚠️ Klasör boş. Lütfen PDF dosyalarınızı 'data/pdfs' klasörüne yükleyin.")
        return []

    for filename in pdf_files:
        path = os.path.join(PDF_FOLDER, filename)
        raw_text = extract_text_from_pdf(path)
        
        if not raw_text.strip():
            print(f"⚠️ Boş içerik veya okunamadı: {filename}")
            continue

        metadata = {"source": filename}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        chunks = splitter.create_documents([raw_text], metadatas=[metadata])
        documents.extend(chunks)
        print(f"   ✅ {filename}: {len(chunks)} parçaya bölündü.")
        
    return documents


def create_faiss_index():
    print("🚀 PDF İşleme Başlatılıyor...")
    
    docs = load_pdfs_and_split()

    if not docs:
        print("❌ Hiçbir belge işlenemedi. Klasör boş olabilir.")
        return

    print(f"🧠 Toplam {len(docs)} metin parçası vektöre çevriliyor... (Biraz sürebilir)")
    
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        
        vectorstore = FAISS.from_documents(docs, embedding_model)
        
        if not os.path.exists(INDEX_FOLDER):
            os.makedirs(INDEX_FOLDER)

        vectorstore.save_local(INDEX_FOLDER)
        print(f"✅ FAISS İndeksi başarıyla oluşturuldu ve kaydedildi:\n   📁 {INDEX_FOLDER}")
        
    except Exception as e:
        print(f"❌ Kritik Hata: {e}")

if __name__ == "__main__":
    create_faiss_index()
