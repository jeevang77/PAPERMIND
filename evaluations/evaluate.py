from rag_chain import ask_question

test_questions = [
    "What is RAG?",
    "Why do we use RAG?",
    "What are the components of a basic RAG pipeline?",
    "How does RAG reduce hallucinations?",
    "What is semantic search?",
    "What are advanced RAG techniques?",
    "What is query rewriting in RAG?",
    "What is the difference between generation and retrieval?",
    "What is chunking?",
    "Why are embeddings used in RAG?",
    "What is FAISS used for?",
    "What is the role of the retriever?",
    "What is the role of the LLM in RAG?",
    "How does RAG help with private data?",
    "What is Transformers?",
    "explain transformers architecture?",
    "how transformers evolve?",
    "who came up with transformer architecture first?",
    "what is the capability of transformers?"
    "which type of data can pass through transformers?"
]


def evaluation():
    for i , question in enumerate(test_questions, start=1):
        result = ask_question(question)

        print("=" * 80)
        print(f"Q{i}: {question}")
        print("\nAnswer")
        print(result["answer"])

        print("\nCITATIONS:")
        for c in result["citations"]:
            print(f"- {c["source"]}   |   page  {c["page"]}")

if __name__ == "__main__":
    evaluation()