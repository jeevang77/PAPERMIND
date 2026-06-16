import os 
import json
from groq import Groq
from dotenv import load_dotenv
from retrieve import load_faiss

load_dotenv()

## setup 
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"
vector_store = load_faiss()

## Tool schema

tools = [
    {
        "type" : "function",
        "function" : {
            "name" : "search_paper",
            "description" : ("search for papers for relevant information."
                            "Call this when you need answers to questions about the paper content.") ,
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "query" : {"type" : "string", "description": "The search query to find relevant chunks"}
                },
                "required" : ["query"]
            }  

        }
    }

]

## tool actual function

def search_paper(query: str) -> str:
    results = vector_store.similarity_search(query, k=3)
    if not results :
        return "No relevant information found in the paper."

    context = ""
    for i , doc in enumerate(results , start=1):
        page = doc.metadata.get("page", "unknown")
        context += f"\n[chunk{i}  | Page {page}] : \n{doc.page_content}\n"
    return context



# Agent Function

def ask_agent(question: str) -> str:
    print(f"\nQuestion : {question}")
    print("-" * 25)

    messages = [
        {"role" : "system", "content" : (
            "You are a helpful research paper assitant. "
            "You have access to search_paper tool that searches an indexed papers."
            "if tool returns no relevant information, respond with: I could not find relevant information about this in the paper, \nanswer them directly."
            "Only answer directly WITHOUT the tool for completely unrelated questions."
            "Always cite page numbers when using tool results."
        ) 
        },
        {
            "role" : "user",
            "content" : question
        }
    ]

    try:
        # send to model
        response = client.chat.completions.create(
            model = MODEL,
            messages = messages,
            tools = tools,
            tool_choice = "auto"
                
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # checking if model wants to call a tool
        if tool_calls:
            messages.append(response_message)

            for tool_call in tool_calls :
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "search_paper":
                    tool_result = search_paper(query = function_args.get("query"))
                    print(f" Tool called: {function_name}")
                    print(f"Query sent to tool : {function_args}")                
                    print(f" Retrived context preview : {tool_result[:500]}....")

                messages.append({
                    "tool_call_id" : tool_call.id,
                    "role" : "tool",
                    "name" : function_name,
                    "content" : tool_result
                })

            # get final response 
            final_response = client.chat.completions.create(
                model = MODEL,
                messages= messages        
            )

            return final_response.choices[0].message.content

        # No tool calls needed - direct answer
        print("No tool calls needed. Providing direct answer.")
        return response_message.content
    
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    print("PAPERMIND v1 - Tool-Augmented RAG Agent")
    print("=" * 25)

    while True:
        question = input("Ask a question (q to quit): ")
        if question.lower() == "q":
            break
        answer = ask_agent(question)
        print(f"\nAnswer : \n{answer}")
 


    
    


