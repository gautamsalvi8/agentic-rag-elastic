# Agentic RAG on Elasticsearch

A production-grade AI Search Agent built with **FastAPI**, **Elasticsearch**, and **LangChain**.

## 🚀 Key Features
- **Hybrid Search**: Combines BM25 and Vector Search for maximum relevance.
- **Agentic Logic**: The system "reasons" about the search results before answering.
- **Re-ranking**: Uses advanced precision to filter the best document chunks.

## 🛠️ Tech Stack
- **Database**: Elasticsearch (Vector Store)
- **Framework**: FastAPI (Python)
- **Orchestration**: LangChain & LangGraph
- **Infrastructure**: Docker & Docker Compose

## 🚦 How to Run
1. Start the database: `docker compose up -d`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the backend: `uvicorn backend.app:app --reload`