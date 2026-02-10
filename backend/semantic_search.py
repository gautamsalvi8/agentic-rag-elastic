import os
from dotenv import load_dotenv

# REMOVE the dots from these two:
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch


load_dotenv()

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.es = Elasticsearch(
            "http://localhost:9200",
            basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD"))
        )

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
