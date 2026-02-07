"""
Day 3 – Generate embeddings locally

Goal:
Text -> vector numbers

"""

from sentence_transformers import SentenceTransformer


def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def generate_embeddings(texts, model):
    embeddings = model.encode(texts)
    return embeddings


def main():
    texts = [
        "Elasticsearch supports vector search",
        "RAG combines retrieval with language models",
        "Dogs are cute animals"
    ]

    model = load_model()

    vectors = generate_embeddings(texts, model)

    for text, vec in zip(texts, vectors):
        print("\nTEXT:", text)
        print("VECTOR LENGTH:", len(vec))
        print("FIRST 5 VALUES:", vec[:5])


if __name__ == "__main__":
    main()


import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize embedding model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Elasticsearch
es = Elasticsearch(
    hosts="http://localhost:9200",
    basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD"))
)

# Sample documents to index
documents = [
    "Elasticsearch supports vector search",
    "RAG combines retrieval with language models"
]

def index_documents(text_list):
    for text in text_list:
        vector = embedding_model.encode(text)

        payload = {
            "text": text,
            "vector": vector.tolist()
        }

        es.index(index="rag-docs", document=payload)

    print("✅ Vector documents successfully indexed")

if __name__ == "__main__":
    index_documents(documents)
