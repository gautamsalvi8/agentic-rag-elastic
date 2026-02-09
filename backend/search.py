import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Elasticsearch client
es = Elasticsearch(
    hosts="http://localhost:9200",
    basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD"))
)

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
        
        

