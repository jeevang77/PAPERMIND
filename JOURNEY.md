# PaperMind — Journey

This document explains how i am building PaperMind from a 
naive basic RAG pipeline to a tool-augmented agent, and why 
each upgrade was made.

---

## V0 — Naive RAG

### What I built
- PDF loading → chunking → embeddings → FAISS index
- Fixed retrieve → answer pipeline
- Used LangChain + FAISS + Gemini 2.5 Flash

### How it worked
Every question followed the same fixed path:
1. User asks question
2. System always retrieves top-k chunks from FAISS
3. LLM answers using those chunks

### What was wrong with it
- Always retrieved even for general questions
- No decision making
- No intelligence — just a fixed pipeline
- Called naive RAG for a reason

---

## V1 — Tool-augmented RAG Agent | added `agent.py`

### What I built
- Exposed search_paper() as a callable tool
- Model decides when to call the tool
- Used Groq (Llama 3) + FAISS + function calling

### How it works
1. User asks question
2. Model thinks: does this need the paper?
3. If yes → calls search_paper() tool → retrieves → answers with citations
4. If no → answers directly from knowledge

### What improved
- Model makes decisions instead of following fixed pipeline
- General questions answered without unnecessary retrieval
- Answers include page citations
- No more tool_use_failed errors with proper prompt engineering

### What I learned
- Function calling / tool calling concepts
- Stateless vs stateful function calling
- How to define tools in JSON schema
- How to handle tool results and send them back to model
- Difference between Gemini SDK, OpenAI Responses API, 
  Chat Completions API, and Groq

  ## Key concepts learned

| Concept | What it is |
|---------|-----------|
| Naive RAG | Fixed retrieve → answer, no decisions |
| Tool calling | Model decides when to call your functions |
| Stateless function calling | You carry conversation history manually |
| Function calling vs built-in tools | Custom functions vs provider abstractions |
| Agentic RAG | RAG where model controls retrieval decisions |

---

## V2 — ReAct Agent (currently working)

### What I plan to build
- Multi-step reasoning loop
- Thought → Action → Observation → repeat
- Model searches multiple times if first result is insufficient

### Why
Some complex questions need multiple retrievals.
V1 only calls the tool once. V2 will loop until 
the model has enough information to answer confidently.


---

## Tech stack 

| Version | LLM | Framework | Retrieval |
|---------|-----|-----------|-----------|
| V0 | Gemini 2.5 Flash | LangChain | FAISS |
| V1 | Groq Llama 3 | Groq SDK | FAISS |
