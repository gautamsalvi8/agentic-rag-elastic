from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import time

# Local modules
from router import SearchRouter
from generator import Generator
from hybrid_search import HybridSearch
from reranker import Reranker
from benchmark import Timer, run_query_benchmark
from retriever import retrieve_chunks
from memory_router import MemoryRouter

# RAGAS imports - commented out until needed for evaluation
# Uncomment these if you want to use RAGAS evaluation:
# from ragas import EvaluationDataset, evaluate
# from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
# from ragas.llms import LangchainLLMWrapper
# from langchain_community.llms import HuggingFaceEndpoint


# def run_ragas_eval(query, retrieved_docs, generated_response, ground_truth=None):
#     """
#     Evaluates a single interaction.
#     retrieved_docs: List of strings from Elasticsearch
#     generated_response: String from your LLM
#     """
#     
#     # 1. Prepare the data structure
#     data = [{
#         "user_input": query,
#         "retrieved_contexts": retrieved_docs, # Ragas expects a list of strings
#         "response": generated_response,
#         "reference": ground_truth if ground_truth else "" # Optional: The "perfect" answer
#     }]
#     
#     dataset = EvaluationDataset.from_list(data)
#     
#     # 2. Run the evaluation
#     # Note: Ragas works best if you use a strong model (like GPT-4 or Qwen-72B) 
#     # as the 'Judge', but you can use your local Qwen-3B too.
#     result = evaluate(
#         dataset=dataset,
#         metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()]
#     )
#     
#     return result


# =====================================
# INIT COMPONENTS
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


class QueryRequest(BaseModel):  # ← ADDED THIS - Was missing!
    """Request model for streaming endpoint"""
    query: str
    docs: list[dict] = []  # List of {text: str, score: float}
    history: str = ""


# =====================================
# STREAMING ENDPOINT
# =====================================

@app.post("/stream")
def stream_answer(request: QueryRequest):
    """
    Streams the LLM response token by token.
    Integrates with the existing memory and retrieval logic.
    """
    # 1. Extract data from request model
    docs = request.docs
    query = request.query
    history = request.history
    
    # 2. Determine top score for logic/thresholds
    top_score = docs[0]["score"] if docs else 0
    
    # 3. Get context
    if docs:
        context = [d["text"] for d in docs]
    elif memory.should_search(query):
        chunks, _ = retrieve_chunks(query)
        context = list(dict.fromkeys(chunks))
    else:
        # Fallback to existing memory context
        context = [memory.get_context()]

    def event_stream():
        full_response = ""
        # Stream tokens
        for token in gen.stream_generate(query, context, history, top_score):
            full_response += token
            yield token
        
        # Save to memory once stream finishes
        memory.save(query, full_response)

    return StreamingResponse(event_stream(), media_type="text/plain")


# =====================================
# EXISTING ENDPOINTS
# =====================================

@app.post("/chat")
def chat(req: ChatRequest):
    query = req.query

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

    # Generate answer (non-streaming)
    answer = gen.generate(query, [context])
    llm_time = 0  # Timer would go here
    
    memory.save(query, answer)

    return {
        "route": route,
        "answer": answer,
        "search_latency": search_time,
        "llm_latency": llm_time
    }


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
    # Extract text from docs for generator
    doc_texts = [d.get("text", d.get("_source", {}).get("text", "")) for d in docs]
    answer = gen.generate(query, doc_texts)
    timer.stop("LLM")

    return {
        "answer": answer,
        "metrics": timer.times
    }


@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    results = run_query_benchmark(search, gen, req.queries)
    return results


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    content = file.file.read().decode("utf-8")
    return {
        "filename": file.filename,
        "size": len(content),
        "status": "uploaded"
    }


# =====================================
# HEALTH CHECK
# =====================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "components": {
            "generator": "loaded",
            "search": "ready",
            "reranker": "ready",
            "memory": "ready"
        }
    }


# =====================================
# RUN SERVER
# =====================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)