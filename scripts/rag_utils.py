# scripts/faiss_utils.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


from langchain_text_splitters import CharacterTextSplitter
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("data/embeddings/faiss_index", embedding_model, allow_dangerous_deserialization=True)


def get_context_from_faiss(query: str) -> str:
    docs = db.similarity_search(query, k=1)
    return "\n\n".join([doc.page_content for doc in docs]) if docs else ""
