from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_query(query: str):
    return query

cache_store = {}


def get_cached_query(query):
    return cache_store.get(query)


def set_cached_query(query, results):
    cache_store[query] = results
