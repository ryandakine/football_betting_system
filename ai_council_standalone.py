#!/usr/bin/env python3
"""
Standalone AI Council - Multi-modal betting analysis system
Extracted from gold_standard_main.py without FastAPI dependencies
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
)
log = logging.getLogger("ai_council")


# ── Enhanced Performance Monitoring ──
@dataclass
class PerformanceMetrics:
    total_games: int = 0
    processing_time_ms: float = 0.0
    games_per_second: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    total_picks: int = 0
    batch_count: int = 0


class PerformanceMonitor:
    def __init__(self):
        self.metrics_history: list[PerformanceMetrics] = []
        self.start_time: float | None = None
        self.current_metrics = PerformanceMetrics()

    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self, total_games: int) -> PerformanceMetrics:
        if not self.start_time:
            raise ValueError("Timer not started")

        processing_time = time.time() - self.start_time
        self.current_metrics.total_games = total_games
        self.current_metrics.processing_time_ms = processing_time * 1000
        self.current_metrics.games_per_second = total_games / max(
            processing_time, 0.001
        )

        self.metrics_history.append(self.current_metrics)
        return self.current_metrics


# ── Intelligent Caching System ──
class GameCache:
    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, dict] = {}
        self.max_size = max_size
        self.access_count: dict[str, int] = defaultdict(int)
        self.access_order: deque = deque()

    def get(self, game_id: str) -> dict | None:
        if game_id in self.cache:
            self.access_count[game_id] += 1
            self.access_order.append(game_id)
            return self.cache[game_id]
        return None

    def set(self, game_id: str, data: dict):
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[game_id] = data
        self.access_count[game_id] = 1
        self.access_order.append(game_id)

    def _evict_lru(self):
        while self.access_order:
            candidate = self.access_order.popleft()
            if candidate in self.cache and self.access_count[candidate] <= 1:
                del self.cache[candidate]
                del self.access_count[candidate]
                break

    def get_hit_rate(self) -> float:
        total_accesses = sum(self.access_count.values())
        return len(self.cache) / max(total_accesses, 1)

    def stats(self) -> dict:
        return {
            "cache_size": len(self.cache),
            "hit_rate": self.get_hit_rate(),
            "max_size": self.max_size,
        }


# ── Game Prioritization System ──
class GamePrioritizer:
    def __init__(self):
        self.team_weights = {
            "Yankees": 1.2,
            "Dodgers": 1.2,
            "Red Sox": 1.1,
            "Astros": 1.1,
            "Braves": 1.1,
            "Mets": 1.0,
        }

    def calculate_priority_score(self, game_data: dict) -> float:
        """Calculate priority score for game analysis"""
        base_score = 1.0

        # Edge-based scoring
        edge_cents = game_data.get("edge_cents", 0)
        edge_score = min(edge_cents / 10.0, 2.0)  # Cap at 2.0

        # Team popularity scoring
        home_team = game_data.get("home_team", "")
        away_team = game_data.get("away_team", "")
        team_score = max(
            self.team_weights.get(home_team, 1.0), self.team_weights.get(away_team, 1.0)
        )

        # Time-based scoring (prefer games starting soon)
        commence_time = game_data.get("commence_time", "")
        if commence_time:
            try:
                game_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                time_diff = (
                    game_time - datetime.now().replace(tzinfo=game_time.tzinfo)
                ).total_seconds()
                time_score = max(
                    1.0 - (time_diff / 3600), 0.5
                )  # Prefer games within 1 hour
            except:
                time_score = 1.0
        else:
            time_score = 1.0

        final_score = base_score * edge_score * team_score * time_score
        return final_score


# ── Configuration ──
class OptimizedGoldStandardConfig:
    def __init__(
        self,
        bankroll: float = 500,
        base_unit_size: float = 5,
        max_units: int = 5,
        confidence_threshold: float = 0.55,
        max_opportunities: int = 50,
        min_edge_cents: int = 8,
        batch_size: int = 25,
        max_workers: int = 4,
        cache_size: int = 1000,
        enable_caching: bool = True,
        enable_performance_monitoring: bool = True,
    ) -> None:
        self.bankroll = bankroll
        self.base_unit_size = base_unit_size
        self.max_units = max_units
        self.confidence_threshold = confidence_threshold
        self.max_opportunities = max_opportunities
        self.min_edge_cents = min_edge_cents
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_size = cache_size
        self.enable_caching = enable_caching
        self.enable_performance_monitoring = enable_performance_monitoring


# ── Main AI Council System ──
class HybridOptimizedGoldStandardMLBSystem:
    """Multi-modal AI Council for MLB betting analysis"""

    def __init__(
        self,
        bankroll: float = 500,
        base_unit_size: float = 5,
        max_units: int = 5,
        confidence_threshold: float = 0.55,
        max_opportunities: int = 50,
        min_edge_cents: int = 8,
        batch_size: int = 25,
        max_workers: int = 4,
        cache_size: int = 1000,
    ):
        self.cfg = OptimizedGoldStandardConfig(
            bankroll=bankroll,
            base_unit_size=base_unit_size,
            max_units=max_units,
            confidence_threshold=confidence_threshold,
            max_opportunities=max_opportunities,
            min_edge_cents=min_edge_cents,
            batch_size=batch_size,
            max_workers=max_workers,
            cache_size=cache_size,
        )

        # Initialize optimization components
        self.cache = GameCache(cache_size)
        self.prioritizer = GamePrioritizer()
        self.monitor = PerformanceMonitor()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        log.info(
            "🚀 AI Council initialized – "
            "bankroll=$%.2f baseUnit=$%.2f minEdge=%d¢ "
            "batchSize=%d workers=%d cacheSize=%d",
            bankroll,
            base_unit_size,
            min_edge_cents,
            batch_size,
            max_workers,
            cache_size,
        )

    async def analyze_opportunities_concurrently(self, rows: list[dict]) -> list[dict]:
        """Main AI Council analysis pipeline"""
        self.monitor.start_timer()

        # Step 1: Pre-filter and cache check
        valid_rows = await self._prefilter_and_cache(rows)
        log.info(
            "📊 AI Council pre-filtered %d rows to %d valid opportunities",
            len(rows),
            len(valid_rows),
        )

        if not valid_rows:
            return []

        # Step 2: Prioritize games using AI algorithm
        prioritized_rows = await self._prioritize_games(valid_rows)

        # Step 3: Process in optimized batches
        all_picks = await self._process_optimized_batches(prioritized_rows)

        # Step 4: Sort and limit results
        final_picks = sorted(
            all_picks, key=lambda p: p.get("edge_cents", 0), reverse=True
        )
        final_picks = final_picks[: self.cfg.max_opportunities]

        # Step 5: Update metrics
        metrics = self.monitor.end_timer(len(valid_rows))
        metrics.total_picks = len(final_picks)

        if self.cfg.enable_performance_monitoring:
            log.info(
                "📊 AI Council Performance: %d games in %.1fms (%.1f games/sec) → %d picks",
                len(valid_rows),
                metrics.processing_time_ms,
                metrics.games_per_second,
                len(final_picks),
            )

        return final_picks

    async def _prefilter_and_cache(self, rows: list[dict]) -> list[dict]:
        """Pre-filter using AI Council caching strategy"""
        valid_rows = []

        for row in rows:
            if not isinstance(row, dict):
                continue

            # Basic validation
            required_fields = ["game_id", "bookmaker", "home_team", "away_team"]
            if not all(field in row for field in required_fields):
                continue

            # Must be FanDuel
            if row.get("bookmaker", "").lower() != "fanduel":
                continue

            # Must meet minimum edge
            edge_cents = row.get("edge_cents", 0)
            if edge_cents < self.cfg.min_edge_cents:
                continue

            # Cache check
            game_id = row["game_id"]
            cached_result = self.cache.get(game_id)
            if cached_result:
                self.monitor.current_metrics.cache_hits += 1
                valid_rows.append(cached_result)
            else:
                self.monitor.current_metrics.cache_misses += 1
                self.cache.set(game_id, row)
                valid_rows.append(row)

        return valid_rows

    async def _prioritize_games(self, rows: list[dict]) -> list[dict]:
        """Prioritize games using AI Council algorithm"""
        scored_rows = []
        for row in rows:
            priority_score = self.prioritizer.calculate_priority_score(row)
            row["priority_score"] = priority_score
            scored_rows.append(row)

        # Sort by priority score
        prioritized = sorted(
            scored_rows, key=lambda x: x["priority_score"], reverse=True
        )
        return prioritized

    async def _process_optimized_batches(self, rows: list[dict]) -> list[dict]:
        """Process games in optimized batches using AI Council"""
        if not rows:
            return []

        # Create balanced batches
        batches = self._create_balanced_batches(rows)
        self.monitor.current_metrics.batch_count = len(batches)

        # Process batches concurrently
        tasks = []
        for batch_id, batch_rows in enumerate(batches):
            task = self._process_batch_async(batch_id, batch_rows)
            tasks.append(task)

        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_picks = []
        for result in batch_results:
            if isinstance(result, list):
                all_picks.extend(result)
            else:
                log.error(f"❌ Batch processing error: {result}")

        return all_picks

    def _create_balanced_batches(self, rows: list[dict]) -> list[list[dict]]:
        """Create balanced batches for optimal processing"""
        batch_size = self.cfg.batch_size
        batches = []

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            batches.append(batch)

        return batches

    async def _process_batch_async(
        self, batch_id: int, batch_rows: list[dict]
    ) -> list[dict]:
        """Process a batch asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._process_batch_sync, batch_id, batch_rows
        )

    def _process_batch_sync(self, batch_id: int, batch_rows: list[dict]) -> list[dict]:
        """Process a batch synchronously using AI Council logic"""
        picks = []

        for row in batch_rows:
            game_picks = self._process_single_game(row)
            picks.extend(game_picks)

        log.info(
            f"🤖 AI Council batch {batch_id}: processed {len(batch_rows)} games → {len(picks)} picks"
        )
        return picks

    def _process_single_game(self, row: dict) -> list[dict]:
        """Process a single game using AI Council analysis"""
        picks = []

        # Extract game data
        game_id = row.get("game_id", "")
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")
        commence_time = row.get("commence_time", "")

        # Calculate edge and confidence using AI Council logic
        edge_cents = row.get("edge_cents", 0)
        confidence = min(0.55 + (edge_cents / 100.0), 0.95)  # AI confidence calculation

        # Determine bet type and team
        if edge_cents > 0:
            # Positive edge - bet on the team with better odds
            bet_team = (
                home_team
                if row.get("home_price", 0) > row.get("away_price", 0)
                else away_team
            )
            bet_type = "moneyline"

            # Calculate bet size using Kelly Criterion
            edge_decimal = edge_cents / 100.0
            kelly_fraction = edge_decimal / (1.0 - edge_decimal)
            bet_size = min(
                kelly_fraction * self.cfg.base_unit_size,
                self.cfg.max_units * self.cfg.base_unit_size,
            )

            pick = {
                "game_id": game_id,
                "team": bet_team,
                "bet_type": bet_type,
                "confidence": confidence,
                "edge_cents": edge_cents,
                "bet_size": bet_size,
                "commence_time": commence_time,
                "ai_council_analysis": True,
            }

            picks.append(pick)

        return picks

    async def save_recommendations_async(self, picks: list[dict]) -> str:
        """Save AI Council recommendations"""

        def _save_sync():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_council_recommendations_{timestamp}.json"

            # Ensure results directory exists
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)

            filepath = results_dir / filename

            # Prepare data for saving
            save_data = {
                "timestamp": datetime.now().isoformat(),
                "ai_council_version": "1.0.0",
                "total_picks": len(picks),
                "performance_stats": self.get_performance_stats(),
                "recommendations": picks,
            }

            # Save to file
            with open(filepath, "w") as f:
                json.dump(save_data, f, indent=2)

            log.info(f"💾 AI Council saved {len(picks)} recommendations to {filename}")
            return filename

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _save_sync)

    def get_performance_stats(self) -> dict:
        """Get AI Council performance statistics"""
        cache_stats = self.cache.stats()

        return {
            "ai_council_active": True,
            "cache_stats": cache_stats,
            "performance_metrics": {
                "total_games_processed": self.monitor.current_metrics.total_games,
                "processing_time_ms": self.monitor.current_metrics.processing_time_ms,
                "games_per_second": self.monitor.current_metrics.games_per_second,
                "cache_hit_rate": cache_stats["hit_rate"],
                "total_picks_generated": self.monitor.current_metrics.total_picks,
            },
            "configuration": {
                "bankroll": self.cfg.bankroll,
                "base_unit_size": self.cfg.base_unit_size,
                "max_units": self.cfg.max_units,
                "confidence_threshold": self.cfg.confidence_threshold,
                "min_edge_cents": self.cfg.min_edge_cents,
            },
        }

    async def cleanup(self):
        """Cleanup AI Council resources"""
        self.executor.shutdown(wait=True)
        log.info("🧹 AI Council cleanup completed")


# Example usage
if __name__ == "__main__":
    # Test the AI Council
    async def test_ai_council():
        council = HybridOptimizedGoldStandardMLBSystem()

        # Test data
        test_rows = [
            {
                "game_id": "test123",
                "bookmaker": "fanduel",
                "home_team": "Yankees",
                "away_team": "Red Sox",
                "commence_time": "2025-07-25T19:00:00Z",
                "home_price": -150,
                "away_price": 130,
                "edge_cents": 15,
            }
        ]

        opportunities = await council.analyze_opportunities_concurrently(test_rows)
        print(f"AI Council found {len(opportunities)} opportunities")

        if opportunities:
            await council.save_recommendations_async(opportunities)

        await council.cleanup()

    asyncio.run(test_ai_council())
