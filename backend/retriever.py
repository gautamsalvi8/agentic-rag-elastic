import os
import time
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, exceptions

# ===============================
# Load .env variables (from project root when run from frontend)
# ===============================
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(_root, ".env"))
load_dotenv()

ELASTIC_URL = os.getenv("ELASTIC_URL") or "http://localhost:9200"
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")

# ===============================
# Elasticsearch client (hosts required; never pass None)
# ===============================
if ELASTIC_API_KEY:
    es = Elasticsearch(hosts=[ELASTIC_URL], api_key=ELASTIC_API_KEY)
else:
    ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD", "changeme")
    es = Elasticsearch(hosts=[ELASTIC_URL], basic_auth=("elastic", ELASTIC_PASSWORD))


def retrieve_chunks(query, index="rag-docs", top_k=5, retries=3):

    for attempt in range(retries):
        try:
            start = time.time()

            body = {
                "size": top_k,
                "query": {
                    "match": {
                        "text": query
                    }
                }
            }

            res = es.search(index=index, body=body)

            chunks = [
                hit["_source"]["text"]
                for hit in res["hits"]["hits"]
            ]

            latency = round(time.time() - start, 3)

            return chunks, latency

        except exceptions.ConnectionError:
            print(f"⚠️ Elasticsearch connection failed (attempt {attempt+1})")
            time.sleep(1)

    raise Exception("❌ Elasticsearch failed after retries")
