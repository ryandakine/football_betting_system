#!/usr/bin/env python3
"""
Advanced optimizations for Gold Standard MLB System
Handling 15+ concurrent games with maximum efficiency
"""

import asyncio
import json
import logging
import multiprocessing as mp
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


# ── Performance Monitoring ──────────────────────────────────────
@dataclass
class PerformanceMetrics:
    total_games: int
    processing_time_ms: float
    games_per_second: float
    memory_usage_mb: float
    cache_hit_rate: float


class PerformanceMonitor:
    def __init__(self):
        self.metrics_history: list[PerformanceMetrics] = []
        self.start_time: float | None = None

    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self, total_games: int) -> PerformanceMetrics:
        if not self.start_time:
            raise ValueError("Timer not started")

        processing_time = time.time() - self.start_time
        games_per_second = total_games / processing_time if processing_time > 0 else 0

        metrics = PerformanceMetrics(
            total_games=total_games,
            processing_time_ms=processing_time * 1000,
            games_per_second=games_per_second,
            memory_usage_mb=0,  # Would need psutil for real measurement
            cache_hit_rate=0,
        )

        self.metrics_history.append(metrics)
        return metrics


# ── Intelligent Caching System ──────────────────────────────────
class GameCache:
    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, dict] = {}
        self.max_size = max_size
        self.access_count: dict[str, int] = defaultdict(int)

    def get(self, game_id: str) -> dict | None:
        if game_id in self.cache:
            self.access_count[game_id] += 1
            return self.cache[game_id]
        return None

    def set(self, game_id: str, data: dict):
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]

        self.cache[game_id] = data
        self.access_count[game_id] = 1

    def get_hit_rate(self) -> float:
        total_accesses = sum(self.access_count.values())
        return len(self.cache) / total_accesses if total_accesses > 0 else 0


# ── Game Prioritization ─────────────────────────────────────────
class GamePrioritizer:
    def __init__(self):
        self.priority_weights = {
            "edge_cents": 0.4,
            "value_rating": 0.3,
            "time_until_game": 0.2,
            "historical_performance": 0.1,
        }

    def calculate_priority_score(self, game_data: dict) -> float:
        score = 0.0

        # Edge cents (higher is better)
        edge_cents = game_data.get("edge_cents", 0)
        score += edge_cents * self.priority_weights["edge_cents"]

        # Value rating
        value_rating = game_data.get("value_rating", "UNKNOWN")
        rating_scores = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
        score += (
            rating_scores.get(value_rating, 0) * self.priority_weights["value_rating"]
        )

        # Time until game (closer games get higher priority)
        commence_time = game_data.get("commence_time")
        if commence_time:
            try:
                from datetime import datetime

                game_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                now = datetime.utcnow()
                hours_until_game = (game_time - now).total_seconds() / 3600
                # Closer games get higher scores (max 24 hours)
                time_score = max(0, 24 - hours_until_game) / 24
                score += time_score * self.priority_weights["time_until_game"]
            except:
                pass

        return score


# ── Batch Processing with Load Balancing ────────────────────────
class LoadBalancedProcessor:
    def __init__(self, max_workers: int = 4, batch_size: int = 50):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.cache = GameCache()
        self.prioritizer = GamePrioritizer()
        self.monitor = PerformanceMonitor()

    async def process_games_optimized(self, games: list[dict]) -> list[dict]:
        """Process games with advanced optimizations."""
        self.monitor.start_timer()

        # Step 1: Pre-filter and cache
        filtered_games = await self._prefilter_and_cache(games)

        # Step 2: Prioritize games
        prioritized_games = await self._prioritize_games(filtered_games)

        # Step 3: Process in optimized batches
        results = await self._process_optimized_batches(prioritized_games)

        # Step 4: Calculate metrics
        metrics = self.monitor.end_timer(len(games))
        logging.info(
            f"🚀 Processed {len(games)} games in {metrics.processing_time_ms:.2f}ms "
            f"({metrics.games_per_second:.1f} games/sec)"
        )

        return results

    async def _prefilter_and_cache(self, games: list[dict]) -> list[dict]:
        """Pre-filter games and update cache."""
        filtered = []

        for game in games:
            game_id = game.get("game_id")
            if not game_id:
                continue

            # Check cache first
            cached_result = self.cache.get(game_id)
            if cached_result:
                filtered.append(cached_result)
                continue

            # Apply basic filters
            if (
                game.get("bookmaker", "").lower() == "fanduel"
                and game.get("edge_cents", 0) >= 8
            ):
                self.cache.set(game_id, game)
                filtered.append(game)

        return filtered

    async def _prioritize_games(self, games: list[dict]) -> list[dict]:
        """Sort games by priority score."""
        scored_games = []

        for game in games:
            score = self.prioritizer.calculate_priority_score(game)
            scored_games.append((score, game))

        # Sort by score (highest first)
        scored_games.sort(key=lambda x: x[0], reverse=True)

        return [game for score, game in scored_games]

    async def _process_optimized_batches(self, games: list[dict]) -> list[dict]:
        """Process games in optimized batches with load balancing."""
        if not games:
            return []

        # Create batches with load balancing
        batches = self._create_balanced_batches(games)

        # Process batches concurrently
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch_async(batch))
            tasks.append(task)

        # Wait for all batches
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_results = []
        for result in batch_results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logging.error(f"❌ Batch processing error: {result}")

        return all_results

    def _create_balanced_batches(self, games: list[dict]) -> list[list[dict]]:
        """Create batches with balanced workload."""
        if len(games) <= self.batch_size:
            return [games]

        batches = []
        current_batch = []
        current_batch_score = 0

        for game in games:
            score = self.prioritizer.calculate_priority_score(game)

            # If adding this game would make the batch too heavy, start a new batch
            if (
                len(current_batch) >= self.batch_size or current_batch_score > 100
            ):  # Arbitrary threshold
                if current_batch:
                    batches.append(current_batch)
                current_batch = [game]
                current_batch_score = score
            else:
                current_batch.append(game)
                current_batch_score += score

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _process_batch_async(self, batch: list[dict]) -> list[dict]:
        """Process a batch asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool, self._process_batch_sync, batch
        )

    def _process_batch_sync(self, batch: list[dict]) -> list[dict]:
        """Synchronous batch processing."""
        picks = []

        for game in batch:
            # Extract key fields
            game_id = game.get("game_id", "unknown")
            home_team = game.get("home_team", "Unknown")
            away_team = game.get("away_team", "Unknown")
            home_price = game.get("home_price")
            away_price = game.get("away_price")
            edge_cents = game.get("edge_cents", 0)
            value_rating = game.get("value_rating", "UNKNOWN")
            commence_time = game.get("commence_time")

            # Create picks for valid value ratings
            if value_rating in ["HIGH", "MEDIUM", "LOW"]:
                if home_price is not None:
                    home_pick = {
                        "game_id": game_id,
                        "team": home_team,
                        "side": "home",
                        "fanduel_odds": home_price,
                        "best_other_odds": away_price,
                        "best_other_book": "Other",
                        "edge_cents": edge_cents,
                        "value_rating": value_rating,
                        "commence_time": commence_time,
                        "game": f"{home_team} vs {away_team}",
                        "detected_at": game.get("detected_at"),
                        "priority_score": self.prioritizer.calculate_priority_score(
                            game
                        ),
                    }
                    picks.append(home_pick)

                if away_price is not None and away_price != home_price:
                    away_pick = {
                        "game_id": game_id + "_away",
                        "team": away_team,
                        "side": "away",
                        "fanduel_odds": away_price,
                        "best_other_odds": home_price,
                        "best_other_book": "Other",
                        "edge_cents": edge_cents,
                        "value_rating": value_rating,
                        "commence_time": commence_time,
                        "game": f"{home_team} vs {away_team}",
                        "detected_at": game.get("detected_at"),
                        "priority_score": self.prioritizer.calculate_priority_score(
                            game
                        ),
                    }
                    picks.append(away_pick)

        return picks


# ── Memory-Efficient Data Structures ────────────────────────────
class MemoryOptimizedProcessor:
    def __init__(self):
        self.game_sets: set[str] = set()  # Track processed games
        self.edge_thresholds = [8, 15, 25, 50]  # Different edge tiers

    def process_with_memory_optimization(self, games: list[dict]) -> list[dict]:
        """Process games with memory optimization."""

        # Use generators to avoid loading everything into memory
        def game_generator():
            for game in games:
                if self._should_process_game(game):
                    yield game

        picks = []
        for game in game_generator():
            game_picks = self._process_single_game(game)
            picks.extend(game_picks)

            # Track processed games to avoid duplicates
            game_id = game.get("game_id")
            if game_id:
                self.game_sets.add(game_id)

        return picks

    def _should_process_game(self, game: dict) -> bool:
        """Determine if a game should be processed."""
        game_id = game.get("game_id")
        if not game_id or game_id in self.game_sets:
            return False

        return game.get("bookmaker", "").lower() == "fanduel" and game.get(
            "edge_cents", 0
        ) >= min(self.edge_thresholds)

    def _process_single_game(self, game: dict) -> list[dict]:
        """Process a single game efficiently."""
        picks = []

        # Extract only needed fields
        game_id = game.get("game_id")
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        home_price = game.get("home_price")
        away_price = game.get("away_price")
        edge_cents = game.get("edge_cents", 0)
        value_rating = game.get("value_rating")

        if value_rating in ["HIGH", "MEDIUM", "LOW"] and home_price is not None:
            picks.append(
                {
                    "game_id": game_id,
                    "team": home_team,
                    "side": "home",
                    "fanduel_odds": home_price,
                    "edge_cents": edge_cents,
                    "value_rating": value_rating,
                    "game": f"{home_team} vs {away_team}",
                }
            )

        return picks


# ── Usage Example ───────────────────────────────────────────────
async def example_usage():
    """Example of using the optimized processors."""

    # Create sample game data
    sample_games = [
        {
            "game_id": "game_1",
            "bookmaker": "fanduel",
            "home_team": "Yankees",
            "away_team": "Red Sox",
            "home_price": -110,
            "away_price": +100,
            "edge_cents": 15,
            "value_rating": "HIGH",
            "commence_time": "2025-01-15T19:00:00Z",
        },
        # Add more sample games...
    ]

    # Use load-balanced processor
    processor = LoadBalancedProcessor(max_workers=4, batch_size=25)
    results = await processor.process_games_optimized(sample_games)

    print(f"Processed {len(results)} picks from {len(sample_games)} games")

    # Use memory-optimized processor
    mem_processor = MemoryOptimizedProcessor()
    mem_results = mem_processor.process_with_memory_optimization(sample_games)

    print(f"Memory-optimized processing found {len(mem_results)} picks")


if __name__ == "__main__":
    asyncio.run(example_usage())
