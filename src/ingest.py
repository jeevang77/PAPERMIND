from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def load_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    return documents


def chunk_documents(documents, chunk_size: int = 800, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def get_embeddings_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def build_faiss(chunks, embeddings):
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def save_faiss(vector_store, folder_path: str):
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(folder_path)
    print(f"Saved FAISS index to {folder_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "hand on llms modified.pdf"
    index_path = project_root / "indexes" / "faiss_llm_book"

    docs = load_pdf(pdf_path)
    chunks = chunk_documents(docs)
    embeddings = get_embeddings_model()
    vector_store = build_faiss(chunks, embeddings)
    save_faiss(vector_store, index_path)