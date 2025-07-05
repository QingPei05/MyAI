from functools import lru_cache
import hashlib

def cached_analysis(func):
    @lru_cache(maxsize=100)
    def wrapper(image_data: bytes, *args, **kwargs):
        return func(image_data, *args, **kwargs)
    return wrapper
