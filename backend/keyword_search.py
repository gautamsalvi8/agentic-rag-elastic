import os
from dotenv import load_dotenv
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

class KeywordSearch:

    def __init__(self):
        self.es = _es_client()
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
