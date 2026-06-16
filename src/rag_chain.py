# V0 - Naive RAG pipeline (deprecated in V1)
# This file is kept for reference only.
# Use agent.py for the current implementation


import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from retrieve import load_faiss


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def build_rag_chain():
    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    vectorstore = load_faiss()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant for question answering. "
            "Answer the user's question only from the provided context. "
            "If the answer is not in the context, say you don't know. "
            "After the answer, rely only on the retrieved context.\n\n"
            "Context:\n{context}"
        ),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain


def ask_question(query: str):
    rag_chain = build_rag_chain()
    response = rag_chain.invoke({"input": query})

    answer = response["answer"]
    source_docs = response.get("context", [])

    citations = []
    for doc in source_docs:
        source = doc.metadata.get("source", "unknown_page")
        page = doc.metadata.get("page", "unknown_page")
        citations.append({
            "source" : source,
            "page" : page
        })

    return {
        "question": query ,
        "answer" : answer,
        "citations": citations,
        "source_docs": source_docs
    }


if __name__ == "__main__":
    query = "explain transformers?,what is RAG?"
    response = ask_question(query)

    print("\nQUESTION:\n", response["question"])
    print("\nANSWER:\n", response["answer"])
    print("\nCITATIONS:")
    for c in response["citations"]:
        print(f"- {c['source']}  | page {c["page"]}")
