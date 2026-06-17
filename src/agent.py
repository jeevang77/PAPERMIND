import os
import json
from groq import Groq
from dotenv import load_dotenv
from retrieve import load_faiss

load_dotenv()

# ── Setup ────────────────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Tool schema ──────────────────────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_paper",
            "description": (
                "Search the research paper for relevant information. "
                "Call this when the question is about the paper content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant chunks"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ── Actual function ──────────────────────────────────────────────
def search_paper(query: str, vector_store) -> str:
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant information found in the paper."

    context = ""
    for i, doc in enumerate(results, start=1):
        page = doc.metadata.get("page", "unknown")
        context += f"\n[Chunk {i} | Page {page}]:\n{doc.page_content}\n"
    return context

# ── Agent function ───────────────────────────────────────────────
def ask_agent(question: str, vector_store) -> str:
    print(f"\nQuestion: {question}")
    print("-" * 50)

    messages = [
       {
    "role": "system",
    "content": (
        "You are a helpful research paper assistant. "
        "A user has uploaded a document and you have access to the search_paper tool "
        "that searches the content of that uploaded document. "
        "ALWAYS use the search_paper tool for ANY question about: "
        "- the document, paper, book, or file content "
        "- the title, author, topic, or subject of the document "
        "- summaries, explanations, or details from the document "
        "- any specific information the user is asking about from the document "
        "The ONLY time you should answer directly WITHOUT the tool is for "
        "completely unrelated questions like math calculations or general knowledge "
        "that have nothing to do with the uploaded document. "
        "Always cite page numbers from tool results in your answer."
    )
},
        {
            "role": "user",
            "content": question
        }
    ]

    try:
        # Turn 1: send to model
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            clean_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in tool_calls
                ]
            }
            messages.append(clean_message)

            tool_result = ""
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"Tool called: {function_name}")
                print(f"Query sent to tool: {function_args}")

                if function_name == "search_paper":
                    tool_result = search_paper(
                        query=function_args.get("query"),
                        vector_store= vector_store
                    )
                    if "No relevant information" in tool_result:
                        tool_result = "The paper does not contain relevant information about this topic."
                    print(f"Retrieved context preview: {tool_result[:300]}...")

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_result
                })

            # Turn 2: get final answer
            final_response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            return final_response.choices[0].message.content

        # No tool called — direct answer
        print("No tool called. Model answered directly.")
        return response_message.content

    except Exception as e:
        return f"Error: {str(e)}"


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("PaperMind V1 - Tool-augmented RAG Agent")
    print("=" * 50)
    vs = load_faiss()

    while True:
        question = input("Ask a question (q to quit): ")
        if question.lower() == "q":
            break
        answer = ask_agent(question)
        print(f"\nAnswer:\n{answer}")
        print("=" * 50)