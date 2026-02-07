"""
parallel_ingest.py
---------------------------------------
High-speed ingestion using:
✔ chunking
✔ embeddings
✔ bulk indexing
✔ parallel workers

Designed for large PDFs / many files.
"""

import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer

from chunker import chunk_text

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD"))
)


# ------------------------
# Worker function
# ------------------------
def index_batch(batch, filename):
    actions = []

    for chunk in batch:
        vec = model.encode(chunk["text"]).tolist()

        actions.append({
            "_index": "rag-docs",
            "_source": {
                "text": chunk["text"],
                "vector": vec,
                "chunk_id": chunk["chunk_id"],
                "source_file": filename
            }
        })

    helpers.bulk(es, actions)


# ------------------------
# Parallel ingestion
# ------------------------
def parallel_index(text, filename, batch_size=10, workers=4):
    chunks = chunk_text(text)

    batches = [
        chunks[i:i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    print("Total chunks:", len(chunks))
    print("Total batches:", len(batches))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch in batches:
            executor.submit(index_batch, batch, filename)

    print("✅ Parallel indexing complete")


# quick test
if __name__ == "__main__":
    sample = "Elasticsearch scales beautifully. " * 300
    parallel_index(sample, "sample.txt")
