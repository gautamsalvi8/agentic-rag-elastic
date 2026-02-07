from router import SearchRouter

router = SearchRouter()

query = input("Enter query: ")

print("\n--- HYBRID ONLY ---")
results = router.hybrid.search(query, use_reranker=False)

print("\n--- HYBRID + RERANKER ---")
results = router.hybrid.search(query, use_reranker=True)


seen = set()   # ✅ FIX — define this

for r in results:
    if r not in seen:
        print("-", r)
        seen.add(r)
