import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag_chain import ask_question


if __name__ == "__main__":
    query = input("Enter your question: ")
    response = ask_question(query)

    print("\nANSWER:\n")
    print(response["answer"])