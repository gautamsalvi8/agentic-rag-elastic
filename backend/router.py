
from semantic_search import SemanticSearch
from keyword_search import KeywordSearch
# --- ADD THIS IMPORT ---
from hybrid_search import HybridSearch 

class SearchRouter:
    def __init__(self):
        # Option A: If HybridSearch needs the instances passed in:
        kw = KeywordSearch()
        sem = SemanticSearch()
        self.hybrid = HybridSearch(kw, sem)
        
        # Option B: If HybridSearch handles its own setup:
        # self.hybrid = HybridSearch()

    def route(self, query):
        return self.hybrid.search(query)
    

class SearchRouter:
    def __init__(self):
        self.hybrid = HybridSearch()

    def route(self, query):
        return self.hybrid.search(query)

