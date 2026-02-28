---
title: ElasticNode AI Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# ElasticAI – Agentic RAG on Elasticsearch

ElasticAI is a production-style, multi-document RAG (Retrieval-Augmented Generation) system built on top of Elasticsearch, Groq LLMs, and a modern **Streamlit** UI.

It focuses on:
- Robust multi‑PDF ingestion and retrieval
- Smart document routing (per‑chat, per‑query)
- Explainable, source‑aware answers (with scores and chunks)
- Practical performance optimizations suitable for real‑world use

> _ElasticAI can make mistakes, so proper filename-based retrieval is recommended while writing._

---

## Run locally (for judges)

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/gautamsalvi8/agentic-rag-elastic.git
cd agentic-rag-elastic
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root with your **Elastic Cloud** URL and keys (get these from [elastic.co/cloud](https://www.elastic.co/cloud)):

```env
ELASTIC_URL=https://your-deployment.es.region.gcp.elastic.cloud:443
ELASTIC_API_KEY=your_elastic_api_key
USE_GROQ_API=true
GROQ_API_KEY=your_groq_api_key
```

Run the app:

```bash
streamlit run frontend/front.py
```

Open **http://localhost:8501** → upload a PDF → ask questions.

**Optional:** To run Elasticsearch on your machine instead of Elastic Cloud, use `docker-compose up -d` and set `ELASTIC_URL=http://localhost:9200` in `.env`.

---

## 🔍 What ElasticAI Does

- **Upload multiple PDFs**: Resumes, job descriptions, practical files, research papers, etc.
- **Ask natural language questions**: Hinglish + English, short forms (e.g. “qualifications”, “exp”, “smrz”).
- **Agentic retrieval**:
  - Hybrid BM25 + vector search in Elasticsearch
  - Cross‑encoder reranking (with smart fallbacks)
  - Per‑chat document scoping (no cross‑chat contamination)
- **Source transparency**:
  - Shows top chunks, filenames, scores
  - Lets you see exactly _why_ a given answer was produced
- **Resilient behavior**:
  - Avoids hard thresholds that cause “No relevant documents found”
  - Has multiple fallbacks (resume routing, PDF raw‑text fallback, cache)

---

## 🧱 High‑Level Architecture

- **Frontend**: `Streamlit`
  - Main UI in `frontend/front.py`
  - Chat interface, document upload, metric panels
  - Per‑user conversation history (with optional Supabase / local storage)
- **Backend / Retrieval**:
  - `backend/hybrid_search.py`  
    - Sentence‑Transformer embeddings (`all-MiniLM-L6-v2`)
    - Hybrid BM25 + vector similarity
    - Cross‑encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
    - Resume‑aware filename routing for “resume”/“cv” queries
  - `backend/bulk_ingest.py`
    - Chunking via `backend/chunker.py`
    - Batch embeddings + bulk indexing into `rag-docs` index
- **Generation**:
  - `backend/generator.py`
    - Groq API client (LLM: `llama-3.3-70b-versatile`)
    - Stream + non‑stream answer generation
    - Structured prompts with context windows & deduplication
- **Supporting modules**:
  - `backend/retriever.py` (simple BM25 fetcher)
  - `backend/memory_router.py`, `backend/router.py`
  - `backend/metrics_logger.py` for retrieval timing and metrics
  - `backend/metrics.py`, `frontend/Metrics dashboard.py` (analysis views)

---

## 🛠 Tech Stack

- **Core**
  - Python 3.10+
  - FastAPI (backend API entrypoint if needed)
  - Streamlit (interactive UI)
- **Search & Storage**
  - Elasticsearch 8.x (vector + BM25 hybrid search)
- **ML / RAG**
  - Sentence-Transformers (embeddings)
  - HuggingFace transformers (where needed)
  - CrossEncoder (`ms-marco-MiniLM-L-6-v2`) for reranking
  - Groq LLMs for answer generation
- **Other**
  - PyPDF2, pypdf (PDF text extraction)
  - Supabase (optional, for user/session storage)
  - Docker Compose (optional local stack)

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/agentic-rag-elastic.git
cd agentic-rag-elastic
```

### 2. Create and activate virtualenv

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a `.env` file in the project root (copy from your local `.env`):

```env
# Elasticsearch
ELASTIC_URL=http://localhost:9200
ELASTIC_API_KEY=...
# or
ELASTIC_PASSWORD=changeme

# Groq
USE_GROQ_API=true
GROQ_API_KEY=...

# Optional: Supabase, etc.
SUPABASE_URL=...
SUPABASE_KEY=...
```

Spin up Elasticsearch locally (via Docker or your own cluster). Example with Docker:

```bash
docker-compose up -d
```

---

## ▶️ Running the App Locally

Everything is wired through Streamlit; a helper script is included.

```bash
# From project root
.\run_streamlit.bat
# or (cross‑platform)
streamlit run frontend/front.py
```

By default the app runs at `http://localhost:8501`.

---

## 💡 Usage Flow

1. **Open the app**  
   Go to `http://localhost:8501`.

2. **Upload documents**
   - Click the 📎 icon and upload one or more PDFs (resume, JD, reports, etc.).
   - The app:
     - Extracts text with PyPDF2
     - Chunks it
     - Embeds and indexes chunks into Elasticsearch (`rag-docs`)

3. **Ask questions**
   - Example queries:
     - “what are my qualifications and work experience?”
     - “summarise this document”
     - “tell me more about the projects”
     - “highlight the education section”
   - For queries containing “resume” / “cv”:
     - `hybrid_search` biases results towards `*resume*.pdf` / `*cv*.pdf` / `salvi gautam` style filenames.

4. **Inspect sources**
   - Expand **📚 Sources** to see:
     - Filename
     - Chunk id
     - Retrieved text preview
     - Score

5. **Metrics**
   - **⚡ Performance Metrics** shows:
     - Total latency
     - Search, Rerank, Embedding times
     - Cached vs fresh

---

## 🎯 Retrieval & Routing Design

- **Hybrid Search**
  - BM25 + dense vector via `script_score`.
  - Fallback to pure vector search when hybrid returns 0 hits.
- **Reranking**
  - CrossEncoder only runs when `len(docs) > 3`.
  - Top‑k for reranking capped at 5 (to keep latency under ~2s).
  - If reranker returns 0 docs, fallback to original ES top‑k.
- **Per‑chat isolation**
  - Each chat has its own `docs` list.
  - Filters ensure that results come **only** from documents uploaded in the current chat.
- **Filename‑aware routing**
  - For queries mentioning **“resume”** / **“cv”**, `hybrid_search` filters candidates to filenames that look like resumes.
- **PDF raw‑text fallback**
  - If retrieval fails post‑filters but PDFs exist in this chat:
    - App extracts text from `doc_files` directly.
    - Concatenates texts and sends them straight to the LLM.
    - Still shows pseudo‑chunks as sources.

---

## 🚧 Limitations & Notes

- PDF text extraction currently relies on PyPDF2:
  - Image‑only or heavily formatted PDFs may need OCR integration to work perfectly.
- Groq API key is required for generation:
  - Without it, the app returns a clear “Groq client not initialized” message.
- This repo is focused on **retrieval and quality**:
  - Security (auth, multi‑tenant isolation, rate‑limits) is minimal and should be hardened before production.

> _ElasticAI can make mistakes, so proper filename-based retrieval is recommended while writing._

---

## 🧪 Testing & Benchmarks

- `backend/benchmark.py`, `backend/benchmark_embeddings.py`  
  Benchmark retrieval + generation across multiple queries.
- `backend/test_chunker.py`  
  Validates chunking logic on sample text.
- `TEST_CASES_CHECKLIST.md`  
  Manual scenarios for multi‑doc retrieval, routing, and UX.

---

## 📜 License

This project is provided for educational and experimental purposes.  
Please review and update the license according to your needs (MIT / Apache‑2.0 / etc.).
