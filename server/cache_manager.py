import os
import json
import re
from datetime import datetime
import logging
from custom_logger import CustomLogger


logger = CustomLogger("cache_manager", "logs/cache_manager_log.txt")

CACHE_DIR = "cache/forecasts"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_file_path(cache_key):
    """Generate a sanitized file path for the cache key"""
    safe_key = re.sub(r"[^\w\-_]", "_", cache_key)
    return os.path.join(CACHE_DIR, f"{safe_key}.json")


def load_from_file_cache(cache_key, max_age_seconds=1800):
    """Load data from file cache if it exists and is not expired"""
    file_path = get_cache_file_path(cache_key)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                cache_data = json.load(f)

            cache_time = datetime.fromisoformat(
                cache_data.get("cached_at", "2000-01-01T00:00:00")
            )
            if (datetime.now() - cache_time).total_seconds() < max_age_seconds:
                logger.info(f"Using file cache for {cache_key}")
                logger.info(f"Cache loaded successfully for key: {cache_key}")
                return cache_data.get("data")
        except Exception as e:
            logger.error(f"Error loading from file cache: {e}")

    return None


def save_to_file_cache(cache_key, data):
    """Save data to file cache"""
    file_path = get_cache_file_path(cache_key)
    try:
        cache_entry = {
            "data": data,
            "cached_at": datetime.now().isoformat(),
            "key": cache_key,
        }
        with open(file_path, "w") as f:
            json.dump(cache_entry, f, indent=2)
        logger.info(f"Saved to file cache: {cache_key}")
        logger.debug(f"Cache entry saved: {cache_entry}")
        return True
    except Exception as e:
        logger.error(f"Error saving to file cache: {e}")
        return False
