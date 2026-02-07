"""
bulk_ingest.py
---------------------------------
Bulk indexes documents into Elasticsearch with:
✔ chunking
✔ embeddings
✔ batch vectorization (faster)
✔ secure password via .env
✔ clean structure
"""

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from chunker import chunk_text

load_dotenv()


class BulkIngest:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.es = Elasticsearch(
            "http://localhost:9200",
            basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD"))
        )

        self.index = "rag-docs"

    def bulk_index_document(self, text: str, filename: str):

        chunks = chunk_text(text)  # returns list[dict]

        # ✅ encode ALL at once (10x faster)
        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(texts).tolist()

        actions = []

        for c, vec in zip(chunks, vectors):
            actions.append({
                "_index": self.index,
                "_source": {
                    "text": c["text"],
                    "embedding": vec,
                    "chunk_id": c["chunk_id"],
                    "source_file": filename
                }
            })

        helpers.bulk(self.es, actions)

        print(f"✅ Indexed {len(actions)} chunks from {filename}")


# quick test
if __name__ == "__main__":
    sample = "Elasticsearch powers modern search systems. " * 200
    ingestor = BulkIngest()
    ingestor.bulk_index_document(sample, "sample.txt")
