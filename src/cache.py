"""
Smart caching: stores previous query results to avoid re-running the crew for similar searches.
"""

import os
import json
import hashlib

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")


def _get_cache_key(symptoms: str, language: str) -> str:
    """Generate a hash key from symptoms + language."""
    raw = f"{symptoms.strip().lower()}|{language}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached_result(symptoms: str, language: str) -> str | None:
    """Check if a result exists in cache. Returns the result or None."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _get_cache_key(symptoms, language)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            data = json.load(f)
            return data.get("result")

    return None


def save_to_cache(symptoms: str, language: str, result: str):
    """Save a result to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _get_cache_key(symptoms, language)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")

    data = {
        "symptoms": symptoms,
        "language": language,
        "result": result,
    }

    with open(cache_file, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
