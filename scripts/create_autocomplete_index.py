import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


from langchain.schema import Document
import os

SOURCE = "../data/finetune/so.jsonl"
INDEX_PATH = "../data/embeddings/autocomplete_index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def main():
    with open(SOURCE, "r", encoding="utf-8") as f:
        prompts = [json.loads(line)["prompt"] for line in f if "prompt" in json.loads(line)]

    docs = [Document(page_content=p) for p in prompts]
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(INDEX_PATH)

    with open(os.path.join(INDEX_PATH, "texts.json"), "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(prompts)} soru başarıyla FAISS autocomplete index'e işlendi.")
    print(f"📄 texts.json dosyası da {INDEX_PATH}/ içinde kaydedildi.")

if __name__ == "__main__":
    main()
