from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Union, Dict, Any

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("data/embeddings/faiss_index", embedding_model, allow_dangerous_deserialization=True)

def get_context_from_faiss(query: str) -> Union[str, Dict[str, Any]]:

    docs = db.similarity_search(query, k=3)
    
    if not docs:
        return ""

    combined_text = "\n\n".join([doc.page_content for doc in docs])


    primary_metadata = docs[0].metadata

    return {
        "text": combined_text,
        "metadata": primary_metadata
    }
