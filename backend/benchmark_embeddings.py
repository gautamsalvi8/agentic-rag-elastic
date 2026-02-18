"""
benchmark_embeddings.py
---------------------------------
Measures embedding speed + memory usage.
Provides real metrics for blog benchmarks.
"""

import time
import psutil
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = ["Elastic vector search is awesome"] * 1000

process = psutil.Process()

start_mem = process.memory_info().rss / 1024**2
start = time.time()

vectors = model.encode(texts)

end = time.time()
end_mem = process.memory_info().rss / 1024**2

print("\n📊 EMBEDDING BENCHMARK")
print("----------------------")
print("Texts:", len(texts))
print("Time taken:", round(end - start, 3), "seconds")
print("Avg per text:", round((end - start) / len(texts) * 1000, 3), "ms")
print("Memory used:", round(end_mem - start_mem, 2), "MB")
