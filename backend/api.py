from pydantic import BaseModel
from typing import List
import time

# =========================
# Request Model
# =========================
class BenchmarkRequest(BaseModel):
    queries: List[str]


# =========================
# Benchmark Endpoint
# =========================
@front.post("/benchmark")
def benchmark(req: BenchmarkRequest):

    results = []

    total_search = 0
    total_rerank = 0
    total_llm = 0
    total_time_all = 0

    for q in req.queries:

        start_total = time.time()

        # -------------------------
        # MEMORY vs SEARCH routing
        # -------------------------
        if memory.should_search(q):

            route = "search"

            # 🔍 Retrieve
            chunks, search_time = retrieve_chunks(q)

            # ensure unique + limit
            chunks = list(dict.fromkeys(chunks))[:5]

            # context
            context = "\n\n".join(chunks)

            # fake sources (or use real metadata if available)
            sources = [f"chunk_{i}" for i in range(len(chunks))]

            rerank_time = 0  # if not using reranker here

        else:
            route = "memory"

            context = memory.get_context()
            sources = ["memory"]
            search_time = 0
            rerank_time = 0

        # -------------------------
        # LLM
        # -------------------------
        answer, llm_time = ask_llm(q, context)

        # save conversation to memory
        memory.save(q, answer)

        # -------------------------
        # Metrics
        # -------------------------
        total_time = round(time.time() - start_total, 3)

        total_search += search_time
        total_rerank += rerank_time
        total_llm += llm_time
        total_time_all += total_time

        results.append({
            "query": q,
            "route": route,
            "answer": answer,
            "sources": sources,
            "metrics": {
                "search": round(search_time, 3),
                "rerank": round(rerank_time, 3),
                "llm": round(llm_time, 3),
                "total": total_time
            }
        })

    n = len(req.queries)

    return {
        "total_queries": n,

        "averages": {
            "search": round(total_search / n, 3),
            "rerank": round(total_rerank / n, 3),
            "llm": round(total_llm / n, 3),
            "total": round(total_time_all / n, 3)
        },

        "results": results
    }
    
    
@front.post("/chat")
def chat(req: ChatRequest):
    try:
        query = req.query

        start = time.time()

        if memory.should_search(query):
            route = "search"
            chunks, search_time = retrieve_chunks(query)
            context = "\n".join(chunks)
        else:
            route = "memory"
            context = memory.get_context()
            search_time = 0

        answer, llm_time = ask_llm(query, context)

        memory.save(query, answer)

        total = round(time.time() - start, 3)

        return {
            "answer": answer,
            "route": route,
            "metrics": {
                "search": search_time,
                "llm": llm_time,
                "total": total
            },
            "sources": chunks[:3] if route == "search" else []
        }

    except Exception as e:
        return {"error": str(e)}  # 🔥 NEVER crash

