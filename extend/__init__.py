import os

from pathlib import Path
from transformers import TRANSFORMERS_CACHE

transformers_cache = Path(TRANSFORMERS_CACHE)

cache_parent = transformers_cache.parent
if "huggingface" in str(cache_parent.name):
    cache_parent = cache_parent.parent

cache_root = os.getenv("EXTEND_CACHE", cache_parent / "extend")


