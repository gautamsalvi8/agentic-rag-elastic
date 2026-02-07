class HybridSearch:

    def __init__(self, keyword, semantic):
        self.keyword = keyword
        self.semantic = semantic

    def search(self, query, k=5):

        kw = self.keyword.search(query, k)
        sem = self.semantic.search(query, k)

        seen = set()
        results = []

        for r in kw + sem:
            if r not in seen:
                seen.add(r)
                results.append(r)

        return results[:k]
