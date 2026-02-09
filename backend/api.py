from pydantic import BaseModel
from typing import List
import time


class BenchmarkRequest(BaseModel):
    queries: List[str]


@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    results = []

    for q in req.queries:
        start = time.time()

        # reuse your existing pipeline
        chunks, search_time = retrieve_chunks(q)
        context = "\n".join(chunks)
        answer, llm_time = ask_llm(q, context)

        total_time = round(time.time() - start, 3)

        results.append({
            "query": q,
            "answer": answer,
            "search_latency": search_time,
            "llm_latency": llm_time,
            "total_latency": total_time
        })

    return {
        "total_queries": len(req.queries),
        "results": results
    }
