from router import SearchRouter
from generator import Generator
from hybrid_search import HybridSearch
from reranker import Reranker
from benchmark import Timer


router = SearchRouter()
search = HybridSearch()
reranker = Reranker()
gen = Generator(prompt_style="STRICT")
timer = Timer()


query = input("Enter query: ")


# =====================================================
# PART 1 — SHOW RAW SEARCH RESULTS (your original logic)
# =====================================================

print("\n--- HYBRID ONLY ---")
results = router.hybrid.search(query, use_reranker=False)

print("\n--- HYBRID + RERANKER ---")
results = router.hybrid.search(query, use_reranker=True)


seen = set()

for r in results:
    if r not in seen:
        print("-", r)
        seen.add(r)


# =====================================================
# PART 2 — FULL RAG PIPELINE (ADDED, not removed)
# =====================================================

print("\n==============================")
print("🚀 Running Full RAG Pipeline")
print("==============================")


# ---- SEARCH ----
timer.start("Search")
docs = search.search(query, k=10)
timer.stop("Search")


# ---- RERANK ----
timer.start("Rerank")
docs = reranker.rerank(query, docs, top_k=5)
timer.stop("Rerank")


# ---- GENERATE ----
timer.start("LLM")
answer = gen.generate(query, docs)
timer.stop("LLM")


# ---- FINAL ANSWER ----
print("\n🧠 Answer:\n")
print(answer)


# ---- LATENCY REPORT ----
timer.report()
