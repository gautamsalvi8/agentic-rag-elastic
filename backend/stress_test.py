import time
from hybrid_search import HybridSearch

search = HybridSearch()

query = "what is elastic"

sizes = [10, 100, 500, 1000]

for size in sizes:

    print(f"\n--- Testing with {size} docs ---")

    start = time.time()

    for _ in range(size):
        search.search(query)

    t = time.time() - start

    print(f"Total time: {round(t,2)}s")
    print(f"Avg per search: {round(t/size,4)}s")
