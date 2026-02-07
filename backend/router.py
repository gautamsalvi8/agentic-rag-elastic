from keyword_search import KeywordSearch
from semantic_search import SemanticSearch
from hybrid_search import HybridSearch

class SearchRouter:

    def __init__(self):
        kw = KeywordSearch()
        sem = SemanticSearch()
        self.hybrid = HybridSearch(kw, sem)

    def route(self, query):
        return self.hybrid.search(query)
