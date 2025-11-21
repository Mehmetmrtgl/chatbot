from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

INDEX_FOLDER = "data/embeddings/faiss_index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def test_faiss(query):
    print(f"\n🔍 Soru: {query}\n")

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(INDEX_FOLDER, embedding_model, allow_dangerous_deserialization=True)

    results = vectorstore.similarity_search(query, k=3)

    print("✅ En benzer metin parçaları:\n")
    for i, doc in enumerate(results):
        print(f"--- [{i+1}] ---")
        print(doc.page_content[:500])  # ilk 500 karakteri göster
        print()

if __name__ == "__main__":
    sample_question = "şifre sıfırlama nasıl yapılır?"
    test_faiss(sample_question)
