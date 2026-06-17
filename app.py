import streamlit as st
import sys
import os
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ingest import load_pdf, chunk_documents, get_embeddings_model, build_faiss, save_faiss
from retrieve import load_faiss
from agent import ask_agent

st.set_page_config(
    page_title="PaperMind",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PaperMind")
st.subheader("Research Paper Assistant")
st.write("Upload any documents and ask questions from it.")
st.divider()

# ── Session state ─────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "paper_loaded" not in st.session_state:
    st.session_state.paper_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "paper_name" not in st.session_state:
    st.session_state.paper_name = ""

# ── Upload section ────────────────────────────────────────────
st.header("Step 1: Upload your paper")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.paper_name:
        with st.spinner("Processing your paper... please wait"):
            # Save to temp file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
                dir="data"
            ) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # Build fresh index
            docs = load_pdf(tmp_path)
            chunks = chunk_documents(docs)
            embeddings = get_embeddings_model()
            vector_store = build_faiss(chunks, embeddings)
            save_faiss(vector_store, "indexes/faiss_uploaded")

            # Store in session
            st.session_state.vector_store = vector_store
            st.session_state.paper_loaded = True
            st.session_state.paper_name = uploaded_file.name
            st.session_state.chat_history = []

        st.success(f"Ready: {uploaded_file.name}")

# ── Chat section ──────────────────────────────────────────────
if st.session_state.paper_loaded and st.session_state.vector_store is not None:
    st.divider()
    st.header("Step 2: Ask questions")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    question = st.chat_input("Ask a question about your paper...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_agent(
                    question,
                    st.session_state.vector_store  # ← pass uploaded index
                )
            st.write(answer)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })