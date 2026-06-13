# PAPERMIND

PAPERMIND is a personal **Research Paper Assistant** built to learn Retrieval-Augmented Generation (RAG). It loads a PDF, splits it into chunks, converts those chunks into embeddings, stores them in a FAISS vector index, retrieves relevant context for a question, and generates grounded answers with source citations.

## What it does

- Loads a PDF using `PyPDFLoader`
- Splits the document into chunks using `RecursiveCharacterTextSplitter`
- Creates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores embeddings in a local FAISS index
- Retrieves the most relevant chunks for a question
- Uses an LLM to answer only from the retrieved context
- Returns citations such as source file and page number
- Includes a small evaluation script for testing answer quality

## Project goal

This project was built as a personal learning project to understand:

- Semantic search
- Embeddings
- Vector databases
- Retrieval-Augmented Generation (RAG)
- Citation-based answering
- Basic RAG evaluation

## Output
<img width="824" height="403" alt="Screenshot 2026-06-12 191728" src="https://github.com/user-attachments/assets/27e9c606-774a-42c7-be7b-fe92a97b2835" />


## How the pipeline works

1. **Load PDF**  
   The PDF is loaded page by page.

2. **Chunk text**  
   The document is split into smaller overlapping chunks.

3. **Create embeddings**  
   Each chunk is converted into a vector representation.

4. **Store in FAISS**  
   The vectors are stored in a local FAISS index.

5. **Retrieve relevant chunks**  
   For a user question, the most similar chunks are retrieved.

6. **Generate answer**  
   The LLM answers using only the retrieved context.

7. **Return citations**  
   The response includes source metadata such as file name and page number.




**python src/ingest.py**  This will:
- load the PDF
- split it into chunks
- create embeddings
- save the FAISS index locally

`python src/retrieve.py`  This checks whether relevant chunks are being retrieved from the index.

`python src/rag_chain.py`  This runs question answering with citations.

Run the app interface `python app/run_rag.py`

Run evaluation  `python eval/evaluate.py` This runs a small question set to inspect:
- answer quality
- citation relevance
- whether the system says "I don't know" when needed
## Tech stack

- Python
- LangChain
- FAISS
- Hugging Face sentence-transformers
- PyPDF
- Gemini API
- dotenv

## Future improvements

- Multi-document support
- Better citation formatting
- Query rewriting before retrieval
- Conversational memory
- Local LLM support for zero API cost evaluation(Maybe)
- Lightweight UI

## Learning outcome

This project demonstrates the complete RAG V1 workflow:

- document ingestion
- chunking
- embeddings
- vector search
- retrieval-based prompting
- grounded answer generation
- citation return
- evaluation

