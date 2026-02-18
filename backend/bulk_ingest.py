"""
bulk_ingest.py
---------------------------------
Bulk indexes documents into Elasticsearch with:
✔ chunking
✔ embeddings
✔ batch vectorization (faster)
✔ secure password via .env
✔ clean structure
✔ performance tracking
✔ timeout handling
✔ batch processing
✔ PROPER METADATA STRUCTURE (FIXED!)
"""

import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from chunker import chunk_text
import time

load_dotenv()


class BulkIngest:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.es = Elasticsearch(
    os.getenv("ELASTIC_URL"),
    api_key=os.getenv("ELASTIC_API_KEY"),
    request_timeout=60,
    max_retries=3,
    retry_on_timeout=True
)

        self.index = "rag-docs"

    def bulk_index_document(self, text: str, filename: str):
        
        start_total = time.time()
        
        # 0️⃣ Optional: clear any previous chunks for this file to avoid duplicates
        try:
            print(f"\n🧹 Clearing old chunks for '{filename}' (if any)...")
            self.es.delete_by_query(
                index=self.index,
                body={
                    "query": {
                        "term": {
                            "metadata.filename.keyword": filename
                        }
                    }
                },
                conflicts="proceed",
            )
        except Exception as e:
            # Non-fatal — just log and continue with fresh indexing
            print(f"   ⚠️ Could not clear previous chunks for {filename}: {e}")
        
        # ===========================
        # 1️⃣ CHUNKING
        # ===========================
        start_chunk = time.time()
        chunks = chunk_text(text)
        chunk_time = time.time() - start_chunk
        
        if not chunks:
            print(f"⚠️ No chunks created from {filename}")
            return

        # ===========================
        # 2️⃣ EMBEDDING (Batch)
        # ===========================
        start_embed = time.time()
        texts = [c["text"] for c in chunks]
        vectors = self.model.encode(texts, show_progress_bar=False).tolist()
        embed_time = time.time() - start_embed

        # ===========================
        # 3️⃣ BATCH BULK OPERATIONS
        # ===========================
        batch_size = 50
        total_indexed = 0
        total_failed = 0
        
        start_bulk = time.time()
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_vectors = vectors[i:i+batch_size]
            
            actions = []
            for idx, (c, vec) in enumerate(zip(batch_chunks, batch_vectors)):
                actions.append({
                    "_index": self.index,
                    "_source": {
                        "text": c["text"],
                        "embedding": vec,
                        "chunk_id": c["chunk_id"],
                        # 🔥 FIX: Proper metadata structure
                        "metadata": {
                            "filename": filename,
                            "chunk_index": i + idx,  # Global chunk index
                            "total_chunks": len(chunks)
                        }
                    }
                })
            
            try:
                # ========================================
                # FIX: Use options() to avoid deprecation
                # ========================================
                es_client = self.es.options(request_timeout=30)
                
                success, failed = helpers.bulk(
                    es_client,
                    actions,
                    raise_on_error=False
                )
                
                # ========================================
                # FIX: failed is a list, not int
                # ========================================
                total_indexed += success
                total_failed += len(failed) if isinstance(failed, list) else failed
                
            except Exception as e:
                print(f"   ⚠️ Batch {i//batch_size + 1} failed: {e}")
                total_failed += len(actions)
        
        bulk_time = time.time() - start_bulk
        total_time = time.time() - start_total

        # ===========================
        # 4️⃣ PERFORMANCE REPORT
        # ===========================
        print(f"\n✅ Indexed {filename}")
        print(f"   📄 Total chunks: {len(chunks)}")
        print(f"   ✅ Indexed: {total_indexed}")
        if total_failed:
            print(f"   ❌ Failed: {total_failed}")
        print(f"   ⏱ Chunking: {chunk_time:.3f}s")
        print(f"   ⏱ Embedding: {embed_time:.3f}s ({embed_time/len(chunks)*1000:.2f}ms per chunk)")
        print(f"   ⏱ Bulk index: {bulk_time:.3f}s")
        print(f"   ⏱ Total: {total_time:.3f}s")
        
        if total_indexed > 0:
            print(f"   🚀 Throughput: {total_indexed/total_time:.1f} chunks/sec")

        return {
            "chunks": len(chunks),
            "indexed": total_indexed,
            "failed": total_failed,
            "time": round(total_time, 2),
        }


# quick test
if __name__ == "__main__":
    
    print("\n=== TEST 1: Repeated Text ===")
    sample_repeated = "Elasticsearch powers modern search systems. " * 200
    ingestor = BulkIngest()
    ingestor.bulk_index_document(sample_repeated, "test_repeated.txt")
    
    print("\n=== TEST 2: Real Content ===")
    sample_real = """
    Elasticsearch is a distributed, RESTful search and analytics engine 
    capable of addressing a growing number of use cases. As the heart of 
    the Elastic Stack, it centrally stores your data for lightning fast 
    search, fine‑tuned relevancy, and powerful analytics that scale with ease.
    
    Built on Apache Lucene, Elasticsearch provides a distributed, multitenant-capable 
    full-text search engine with an HTTP web interface and schema-free JSON documents.
    
    Elasticsearch is developed in Java and is dual-licensed under the source-available 
    Server Side Public License and the Elastic license, while other parts fall under 
    the proprietary Elastic License. Official clients are available in Java, .NET (C#), 
    PHP, Python, Ruby and many other languages.
    
    The Elastic Stack includes Kibana for visualization, Logstash for data processing, 
    and Beats for lightweight data shipping. Together they provide a complete solution 
    for search, logging, and analytics.
    """
    
    ingestor.bulk_index_document(sample_real, "test_real.txt")