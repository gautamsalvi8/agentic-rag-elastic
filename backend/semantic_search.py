import os
from dotenv import load_dotenv

# REMOVE the dots from these two:
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch


_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_root, ".env"))
load_dotenv()

def _es_client():
    url = os.getenv("ELASTIC_URL") or "http://localhost:9200"
    api_key = os.getenv("ELASTIC_API_KEY")
    if api_key:
        return Elasticsearch(hosts=[url], api_key=api_key)
    return Elasticsearch(hosts=[url], basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD", "changeme")))

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.es = _es_client()
        self.index = "rag-docs"

    def search(self, query, k=5):

        query_vector = self.model.encode(query).tolist()

        body = {
            "size": k,
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": 50
            }
        }

        res = self.es.search(index=self.index, body=body)

        return [h["_source"]["text"] for h in res["hits"]["hits"]]
