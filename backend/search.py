import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_root, ".env"))
load_dotenv()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Elasticsearch client (hosts required; support cloud + local)
_url = os.getenv("ELASTIC_URL") or "http://localhost:9200"
_api_key = os.getenv("ELASTIC_API_KEY")
if _api_key:
    es = Elasticsearch(hosts=[_url], api_key=_api_key)
else:
    es = Elasticsearch(hosts=[_url], basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD", "changeme")))

def semantic_search(question, top_k=2):
    query_vector = model.encode(question).tolist()

    response = es.search(
        index="rag-docs",
        knn={
            "field": "vector",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": 10
        }
    )

    return response["hits"]["hits"]

if __name__ == "__main__":
    results = semantic_search("What is RAG?")

    for hit in results:
        print("→", hit["_source"]["text"])
        
        

