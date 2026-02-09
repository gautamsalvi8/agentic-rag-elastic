from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import time

from router import SearchRouter
from generator import Generator
from hybrid_search import HybridSearch
from reranker import Reranker
from benchmark import Timer, run_query_benchmark
from retriever import retrieve_chunks
from llm import ask_llm
from memory_router import MemoryRouter


# =====================================
# INIT COMPONENTS (load once only)
# =====================================

app = FastAPI(title="Agentic RAG API")

memory = MemoryRouter()
router = SearchRouter()
search = HybridSearch()
reranker = Reranker()
gen = Generator(prompt_style="STRICT")


# =====================================
# REQUEST MODELS
# =====================================

class ChatRequest(BaseModel):
    query: str


class BenchmarkRequest(BaseModel):
    queries: list[str]


# =====================================
# CHAT ENDPOINT (MAIN ONE)
# =====================================

@app.post("/chat")
def chat(req: ChatRequest):

    query = req.query

    # -------------------------
    # MEMORY vs SEARCH ROUTING
    # -------------------------
    if memory.should_search(query):
        route = "search"

        start = time.time()
        chunks, _ = retrieve_chunks(query)
        context = "\n\n".join(list(dict.fromkeys(chunks)))[:3000]
        search_time = round(time.time() - start, 3)

    else:
        route = "memory"
        context = memory.get_context()
        search_time = 0


    # -------------------------
    # LLM GENERATION
    # -------------------------
    answer, llm_time = ask_llm(query, context)

    memory.save(query, answer)


    return {
        "route": route,
        "answer": answer,
        "search_latency": search_time,
        "llm_latency": llm_time
    }


# =====================================
# FULL RAG PIPELINE ENDPOINT
# (optional advanced)
# =====================================

@app.post("/rag")
def rag(req: ChatRequest):

    query = req.query
    timer = Timer()

    timer.start("Search")
    docs = search.search(query, k=10)
    timer.stop("Search")

    timer.start("Rerank")
    docs = reranker.rerank(query, docs, top_k=5)
    timer.stop("Rerank")

    timer.start("LLM")
    answer = gen.generate(query, docs)
    timer.stop("LLM")

    return {
        "answer": answer,
        "metrics": timer.times
    }


# =====================================
# BENCHMARK (NOW USER DEFINED)
# =====================================

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    results = run_query_benchmark(search, gen, req.queries)
    return results


# =====================================
# FILE UPLOAD (future indexing)
# =====================================

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    content = file.file.read().decode("utf-8")

    # later → index to elastic
    return {
        "filename": file.filename,
        "size": len(content),
        "status": "uploaded"
    }
