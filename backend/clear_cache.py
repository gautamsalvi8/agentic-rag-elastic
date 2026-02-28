from cache import *

# Clear all cached queries
if hasattr(cache, 'clear'):
    cache.clear()
    print("✅ Cache cleared")
else:
    # If cache is a dict
    cache = {}
    print("✅ Cache reset")