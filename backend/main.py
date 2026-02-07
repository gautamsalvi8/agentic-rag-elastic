from router import SearchRouter

router = SearchRouter()

query = input("Enter query: ")

results = router.route(query)

print("\nResults:\n")

seen = set()   # ✅ FIX — define this

for r in results:
    if r not in seen:
        print("-", r)
        seen.add(r)
