# ai_cache.py
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict

import aiofiles

logger = logging.getLogger(__name__)


class AICache:
    """
    A high-performance, asynchronous, file-based cache with time-based expiration.
    Designed to be used as an async context manager.
    """

    def __init__(
        self,
        cache_file="cache/ai_analysis_cache.json",
        cache_duration_seconds: int = 6 * 3600,
    ):
        self.cache_file = cache_file
        self.cache_duration = cache_duration_seconds
        self.cache: dict[str, Any] = {}
        # Ensure the cache directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

    async def _load_cache(self):
        """Asynchronously loads the cache from the file."""
        if os.path.exists(self.cache_file):
            try:
                async with aiofiles.open(self.cache_file, "r", encoding="utf-8") as f:
                    content = await f.read()
                    self.cache = json.loads(content)
                logger.info(
                    f"✅ Loaded {len(self.cache)} items from AI cache at {self.cache_file}"
                )
            except Exception as e:
                logger.error(
                    f"Error loading cache file: {e}. Starting with an empty cache."
                )
                self.cache = {}

    async def _save_cache(self):
        """Asynchronously saves the entire cache to the file."""
        try:
            async with aiofiles.open(self.cache_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(self.cache, indent=2))
            logger.info(
                f"💾 Saved {len(self.cache)} items to AI cache at {self.cache_file}"
            )
        except Exception as e:
            logger.error(f"Error saving cache file: {e}")

    def _generate_key(self, game_id: str, model_version: str) -> str:
        """Generates a consistent cache key, now including model version."""
        # Adding a model version to the key prevents using stale data if you update the AI prompt/model
        key_string = f"{game_id}-{model_version}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, game_id: str, model_version: str = "1.0") -> Any:
        """
        Retrieves an item from the cache if it exists and is not expired.
        This operation is synchronous as it only reads from memory.
        """
        cache_key = self._generate_key(game_id, model_version)
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry["timestamp"] < self.cache_duration:
                logger.info(f"CACHE HIT for game {game_id}")
                return entry["result"]
            else:
                logger.info(f"CACHE EXPIRED for game {game_id}")
                del self.cache[cache_key]  # Remove expired entry

        logger.info(f"CACHE MISS for game {game_id}")
        return None

    def set(self, game_id: str, result: Any, model_version: str = "1.0"):
        """
        Adds or updates an item in the cache in memory.
        Does not write to disk immediately for performance.
        """
        cache_key = self._generate_key(game_id, model_version)
        self.cache[cache_key] = {"timestamp": time.time(), "result": result}
        logger.info(f"CACHE SET for game {game_id}")

    # --- Context Manager Methods ---
    async def __aenter__(self):
        """Called when entering the 'async with' block. Loads the cache."""
        await self._load_cache()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting the 'async with' block. Saves the cache."""
        await self._save_cache()


# --- Example Usage ---
async def main():
    """Demonstrates how to use the asynchronous AICache."""
    logging.basicConfig(level=logging.INFO)

    game1_id = "test_game_123"

    # The 'async with' statement handles loading and saving automatically
    async with AICache() as cache:
        # First time, it will be a cache miss
        cached_result = cache.get(game1_id)
        if not cached_result:
            # Simulate getting a new result from an AI
            new_result = {"confidence": 0.75, "team": "Yankees"}
            cache.set(game1_id, new_result)

    print("-" * 20)

    # In a separate run or later in the code, it will be a cache hit
    async with AICache() as cache:
        cached_result = cache.get(game1_id)
        print("Retrieved from cache:", cached_result)


if __name__ == "__main__":
    asyncio.run(main())
