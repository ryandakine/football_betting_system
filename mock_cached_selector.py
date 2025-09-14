#!/usr/bin/env python3
"""
Mock Cached Game Selector - Test caching logic without AI dependencies
Demonstrates the caching architecture before installing AI libraries
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from api_config import get_api_keys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MockGameSelector:
    """
    Mock version demonstrating caching logic without requiring AI libraries.
    Perfect for testing the architecture before installing dependencies.
    """

    def __init__(
        self, alert_manager=None, cache_duration_hours=6, cache_dir="testing_logs"
    ):
        """Initialize Mock GameSelector with caching capabilities."""
        logger.info(
            "🧪 Initializing Mock GameSelector (no AI dependencies required)..."
        )
        self.alert_manager = alert_manager

        # Caching Configuration
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "mock_ai_cache.json")
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self._ensure_cache_directory()
        self._load_cache()

        # Mock AI providers
        self.mock_providers = ["MockClaude", "MockGPT", "MockGrok"]

        logger.info(
            f"🧪 Mock GameSelector ready with {len(self.mock_providers)} mock providers and {cache_duration_hours}h caching"
        )

    def _ensure_cache_directory(self):
        """Ensure cache directory exists."""
        os.makedirs(self.cache_dir, exist_ok=True)

    # --- Same Caching Logic as Real Version ---

    def _load_cache(self):
        """Load the AI analysis cache from persistent storage."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file) as f:
                    self.cache = json.load(f)
                self._clean_expired_cache()
                logger.info(f"✅ Loaded {len(self.cache)} valid items from mock cache")
            else:
                self.cache = {}
                logger.info("📝 Created new mock cache")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"⚠️ Cache load failed ({e}), starting fresh")
            self.cache = {}

    def _save_cache(self):
        """Save the current cache to persistent storage."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")

    def _clean_expired_cache(self):
        """Remove expired entries from cache."""
        current_time = datetime.now(timezone.utc)
        expired_keys = []

        for game_id, cached_item in self.cache.items():
            try:
                timestamp = datetime.fromisoformat(cached_item["timestamp"])
                if current_time - timestamp > self.cache_duration:
                    expired_keys.append(game_id)
            except (KeyError, ValueError):
                expired_keys.append(game_id)

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"🧹 Cleaned {len(expired_keys)} expired cache entries")

    def _get_cache_key(self, game_info: dict) -> str:
        """Generate a unique cache key for a game."""
        game_id = game_info.get("game_id", "unknown")
        commence_time = game_info.get("commence_time", "")
        return f"{game_id}_{commence_time}"

    def _get_from_cache(self, cache_key: str) -> dict | None:
        """Retrieve a valid result from cache if not expired."""
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            try:
                timestamp = datetime.fromisoformat(cached_item["timestamp"])
                if datetime.now(timezone.utc) - timestamp < self.cache_duration:
                    logger.info(f"✅ Using cached analysis for {cache_key}")
                    return cached_item["data"]
            except (KeyError, ValueError):
                del self.cache[cache_key]
        return None

    def _update_cache(self, cache_key: str, data: dict):
        """Update cache with new analysis result."""
        self.cache[cache_key] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        self._save_cache()

    # --- Mock AI Analysis ---

    def _format_game_data_for_ai(self, game_info: dict) -> dict:
        """Format game data for AI consumption."""
        home_team = game_info.get("home_team", "Unknown")
        away_team = game_info.get("away_team", "Unknown")

        return {
            "game_id": game_info.get("game_id", "unknown"),
            "matchup": f"{away_team} @ {home_team}",
            "home_team": home_team,
            "away_team": away_team,
            "context": f"Mock analysis: {away_team} visiting {home_team}",
        }

    async def _mock_ai_analysis(self, provider: str, game_data: dict) -> dict:
        """Mock AI analysis that simulates real AI providers."""
        # Simulate API call delay
        await asyncio.sleep(random.uniform(0.5, 1.5))

        # Generate deterministic but varied results based on team names
        home_team = game_data.get("home_team", "")
        away_team = game_data.get("away_team", "")

        # Create pseudo-random but consistent confidence based on team names
        seed = hash(f"{provider}_{home_team}_{away_team}") % 1000
        random.seed(seed)

        confidence = round(random.uniform(0.55, 0.88), 3)

        # Generate mock reasoning
        factors = [
            "strong starting pitcher matchup",
            "favorable home field advantage",
            "recent offensive hot streak",
            "opponent struggling on road",
            "bullpen depth advantage",
            "key player returning from injury",
            "weather conditions favor home team",
        ]

        reasoning = (
            f"Mock {provider}: {random.choice(factors)} gives edge to {home_team}."
        )

        return {
            "provider": f"Mock {provider}",
            "confidence": confidence,
            "reasoning": reasoning,
            "home_team_edge": confidence > 0.65,
        }

    async def analyze_game(self, game_info: dict) -> dict:
        """Mock game analysis with full caching logic."""
        cache_key = self._get_cache_key(game_info)
        game_id = game_info.get("game_id", "unknown")

        # Step 1: Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result

        # Step 2: Format data and run mock analyses
        formatted_data = self._format_game_data_for_ai(game_info)
        logger.info(f"🔍 Mock analyzing {formatted_data['matchup']} (cache miss)")

        # Run all mock AI analyses in parallel
        analysis_tasks = [
            self._mock_ai_analysis(provider, formatted_data)
            for provider in self.mock_providers
        ]

        individual_analyses = await asyncio.gather(*analysis_tasks)

        # Calculate consensus
        confidences = [a["confidence"] for a in individual_analyses]
        consensus_confidence = sum(confidences) / len(confidences)

        # Log individual results
        for analysis in individual_analyses:
            provider = analysis["provider"]
            confidence = analysis["confidence"]
            reasoning = analysis["reasoning"]
            logger.info(f"   {provider}: {confidence:.1%} - {reasoning}")

        final_result = {
            "game_id": game_id,
            "game_info": game_info,
            "consensus_confidence": round(consensus_confidence, 3),
            "individual_analyses": individual_analyses,
            "providers_count": len(individual_analyses),
            "cache_status": "fresh_mock_analysis",
        }

        logger.info(
            f"📊 Mock consensus for {game_id}: {consensus_confidence:.1%} from {len(individual_analyses)} providers"
        )

        # Step 3: Cache the result
        self._update_cache(cache_key, final_result)

        return final_result

    async def analyze_games(self, games_list: list[dict]) -> dict[str, Any]:
        """Analyze multiple games with mock AI and caching."""
        if not games_list:
            logger.warning("No games provided for analysis")
            return {}

        logger.info(f"🎯 Mock analyzing {len(games_list)} games")

        # Analyze all games in parallel
        analysis_tasks = [self.analyze_game(game) for game in games_list]
        analysis_results = await asyncio.gather(*analysis_tasks)

        # Convert to dictionary and gather stats
        results_dict = {}
        cache_hits = 0
        fresh_analyses = 0

        for result in analysis_results:
            game_id = result.get("game_id", "unknown")
            results_dict[game_id] = result

            cache_status = result.get("cache_status", "unknown")
            if "fresh" in cache_status:
                fresh_analyses += 1
            else:
                cache_hits += 1

        # Summary statistics
        if results_dict:
            confidences = [r["consensus_confidence"] for r in results_dict.values()]
            avg_confidence = sum(confidences) / len(confidences)

            logger.info(f"✅ Mock analysis complete:")
            logger.info(f"   Games analyzed: {len(results_dict)}")
            logger.info(f"   Average confidence: {avg_confidence:.1%}")
            logger.info(
                f"   Cache hits: {cache_hits}, Fresh analyses: {fresh_analyses}"
            )

        return results_dict

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_entries = len(self.cache)
        cache_size_mb = 0

        try:
            if os.path.exists(self.cache_file):
                cache_size_mb = os.path.getsize(self.cache_file) / 1024 / 1024
        except:
            pass

        return {
            "total_cached_games": total_entries,
            "cache_duration_hours": self.cache_duration.total_seconds() / 3600,
            "cache_file_size_mb": round(cache_size_mb, 3),
            "mock_providers": len(self.mock_providers),
        }


# --- Testing ---


async def test_mock_cached_selector():
    """Test the mock cached selector to demonstrate caching logic."""
    sample_games = [
        {
            "game_id": "mock_game_1",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "commence_time": "2025-06-16T19:00:00Z",
        },
        {
            "game_id": "mock_game_2",
            "home_team": "Los Angeles Dodgers",
            "away_team": "San Francisco Giants",
            "commence_time": "2025-06-16T22:00:00Z",
        },
        {
            "game_id": "mock_game_3",
            "home_team": "Chicago Cubs",
            "away_team": "Milwaukee Brewers",
            "commence_time": "2025-06-16T20:00:00Z",
        },
    ]

    # Test with short cache duration for demo
    selector = MockGameSelector(cache_duration_hours=0.1)  # 6 minutes for testing

    print("=" * 70)
    print("🧪 MOCK CACHED GAME SELECTOR TEST")
    print("=" * 70)
    print("This demonstrates the caching architecture without requiring AI libraries!")
    print()

    # First run - should be fresh analyses
    logger.info("🧪 FIRST RUN (should be fresh mock analyses):")
    start_time = time.time()
    results1 = await selector.analyze_games(sample_games)
    first_run_time = time.time() - start_time

    print(f"\n⏱️ First run took {first_run_time:.2f} seconds")

    # Second run - should use cache
    logger.info("\n🧪 SECOND RUN (should use cache):")
    start_time = time.time()
    results2 = await selector.analyze_games(sample_games)
    second_run_time = time.time() - start_time

    print(f"⏱️ Second run took {second_run_time:.2f} seconds")
    print(f"🚀 Speedup: {first_run_time/second_run_time:.1f}x faster with cache!")

    # Display results
    print(f"\n📊 ANALYSIS RESULTS:")
    for game_id, analysis in results1.items():
        home_team = analysis["game_info"]["home_team"]
        away_team = analysis["game_info"]["away_team"]
        consensus = analysis["consensus_confidence"]

        print(f"\n🏈 {away_team} @ {home_team}")
        print(f"   Consensus: {consensus:.1%}")

        for individual in analysis["individual_analyses"][:2]:  # Show first 2
            provider = individual["provider"]
            confidence = individual["confidence"]
            print(f"   {provider}: {confidence:.1%}")

    # Cache statistics
    cache_stats = selector.get_cache_stats()
    print(f"\n📊 CACHE PERFORMANCE:")
    for key, value in cache_stats.items():
        print(f"   {key}: {value}")

    print(f"\n✅ Mock test complete! Install AI libraries and run real version:")
    print(f"   pip install anthropic openai")
    print(f"   python cached_game_selection.py")


if __name__ == "__main__":
    asyncio.run(test_mock_cached_selector())
