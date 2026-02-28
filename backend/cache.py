"""
Simple in-memory cache for search results
"""

cache_store = {}


def get_cached_query(query):
    """Get cached search results for a query"""
    return cache_store.get(query)


def set_cached_query(query, results):
    """Cache search results for a query"""
    cache_store[query] = results

class QueryCache:
    def __init__(self):
        self.store = {}
        self.hits = 0
        self.total = 0

    def get(self, key):
        self.total += 1
        if key in self.store:
            self.hits += 1
            return self.store[key]
        return None

    def set(self, key, value):
        self.store[key] = value

    def stats(self):
        ratio = (self.hits / self.total) * 100 if self.total else 0
        return {
            "total_queries": self.total,
            "cache_hits": self.hits,
            "hit_ratio": round(ratio, 2)
        }
