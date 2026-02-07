import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from cache import get_cached_query, set_cached_query
import time
from reranker import Reranker


load_dotenv()


class HybridSearch:

    def __init__(self):
        self.es = Elasticsearch(
            "http://localhost:9200",
            basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD"))
        )

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = Reranker()

        self.index = "rag-docs"

        # weights
        self.alpha = 0.6
        self.beta = 0.4

    def search(self, query, k=5, use_reranker=True):
        
        start_total = time.time()
        
        # -----------------
        # 1️⃣ cache check
        # -----------------
        cached = get_cached_query(query)
        if cached:
            print("⚡ cache hit")
            return cached
        
        # -----------------
        # 2️⃣ embedding
        # -----------------
        query_vector = self.model.encode(query).tolist()
        
        # -----------------
        # 3️⃣ hybrid scoring
        # -----------------
        start_es = time.time()
        
        body = {
            "size": 20,  # fetch more for reranker
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": """
                            double bm25 = _score;
                            double vector = cosineSimilarity(params.query_vector, 'embedding') + 1.0;
                            return params.alpha * bm25 + params.beta * vector;
                        """,
                        "params": {
                            "query_vector": query_vector,
                            "alpha": self.alpha,
                            "beta": self.beta
                        }
                    }
                }
            }
        }
        
        res = self.es.search(index=self.index, body=body)
        docs = [hit["_source"]["text"] for hit in res["hits"]["hits"]]
        
        es_time = time.time() - start_es
        
        # -----------------
        # 4️⃣ rerank (optional)
        # -----------------
        if use_reranker and docs:
            start_rr = time.time()
            docs = self.reranker.rerank(query, docs, top_k=k)
            rr_time = time.time() - start_rr
        else:
            docs = docs[:k]  # just take top k if no reranking
            rr_time = 0
        
        total_time = time.time() - start_total
        
        # -----------------
        # 5️⃣ timing logs
        # -----------------
        print(f"\n⏱ ES time: {es_time:.3f}s")
        print(f"⏱ Rerank time: {rr_time:.3f}s")
        print(f"⏱ Total latency: {total_time:.3f}s\n")
        
        # -----------------
        # 6️⃣ cache save
        # -----------------
        set_cached_query(query, docs)
        
        return docs