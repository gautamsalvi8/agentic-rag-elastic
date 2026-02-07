import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

class KeywordSearch:

    def __init__(self):
        self.es = Elasticsearch(
            "http://localhost:9200",
            basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD"))
        )
        self.index = "rag-docs"

    def search(self, query, k=5):

        body = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text"],
                    "fuzziness": "AUTO"   # 🔥 important
                }
            }
        }

        res = self.es.search(index=self.index, body=body)
        return [h["_source"]["text"] for h in res["hits"]["hits"]]
