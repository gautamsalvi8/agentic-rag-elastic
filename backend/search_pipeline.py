import time

class SearchPipeline:
    def __init__(self, hybrid_search, reranker, embedder, cache):
        self.hybrid = hybrid_search
        self.reranker = reranker
        self.embedder = embedder
        self.cache = cache
        # Define the threshold as a class constant or config variable
        self.TOP_SCORE_THRESHOLD = 0.35

    def query(self, query):
        total_start = time.time()

        # 1. Cache check
        if query in self.cache:
            return {
                "cached": True,
                **self.cache[query]
            }

        # 2. Embedding Generation
        embed_start = time.time()
        embedding = self.embedder.encode(query)
        embed_time = time.time() - embed_start

        # 3. Hybrid Search (Elasticsearch)
        search_start = time.time()
        results = self.hybrid.search(query, embedding)
        search_time = time.time() - search_start

        # 4. Reranking (Cross-Encoder)
        rerank_start = time.time()
        reranked = self.reranker.rerank(query, results)
        rerank_time = time.time() - rerank_start

        total_time = time.time() - total_start

        # --- GUARDRAIL LOGIC START ---
        # Check if we have results and if the top result meets our quality bar
        if not reranked or reranked[0]["score"] < self.TOP_SCORE_THRESHOLD:
            return {
                "answer": "Not found in document.",
                "guard_triggered": True,
                "results": [],
                "total_latency": total_time,
                "cached": False,
                "top_score": reranked[0]["score"] if reranked else 0
            }
        # --- GUARDRAIL LOGIC END ---

        response = {
            "results": reranked,
            "embedding_time": embed_time,
            "search_time": search_time,
            "rerank_time": rerank_time,
            "total_latency": total_time,
            "cached": False,
            "guard_triggered": False
        }

        # Update cache with valid response
        self.cache[query] = response
        return response