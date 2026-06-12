from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FAISS_PATH = PROJECT_ROOT / "indexes" / "faiss_llm_book"


def get_embeddings_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def load_faiss(folder_path: str | Path = DEFAULT_FAISS_PATH):
    embeddings = get_embeddings_model()
    vector_store = FAISS.load_local(
        str(folder_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def search_docs(query: str, k: int = 3, folder_path: str | Path = DEFAULT_FAISS_PATH):
    db = load_faiss(folder_path)
    results = db.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    query = "what is the main usage of RAG?"
    results = search_docs(query, k=3)

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print(doc.page_content[:400])
        print(doc.metadata)