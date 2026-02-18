# create_index.py

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()

ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")

# ---------------------------------
# Connect to Elasticsearch
# ---------------------------------
# Supports both API key (Elastic Cloud) and basic auth (local)
if ELASTIC_API_KEY:
    es = Elasticsearch(ELASTIC_URL, api_key=ELASTIC_API_KEY)
else:
    ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD", "changeme")
    es = Elasticsearch(ELASTIC_URL, basic_auth=("elastic", ELASTIC_PASSWORD))

INDEX_NAME = "rag-docs"


def create_index():
    """
    Create elastic index for storing:
    1. original text
    2. embedding vector

    MiniLM model → 384 dimensions
    """

    mapping = {
        "mappings": {
            "properties": {
                "text": {
                    "type": "text"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine"
                },
                "chunk_id": {
                    "type": "integer"
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "total_chunks": {"type": "integer"}
                    }
                }
            }
        }
    }


    # ignore error if index already exists
    es.indices.create(
        index=INDEX_NAME,
        body=mapping,
        ignore=400
    )

    print(f"Index '{INDEX_NAME}' is ready.")


# ---------------------------------
# run file directly
# ---------------------------------
if __name__ == "__main__":
    create_index()
