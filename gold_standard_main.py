#!/usr/bin/env python3
# gold_standard_main.py – HYBRID OPTIMIZED version with LINE MONITORING INTEGRATION
"""
Best of both worlds: Cursor's advanced optimization classes + FastAPI integration + Line Monitoring Intelligence
Handles 15+ concurrent games with maximum efficiency and modularity + Real-time betting intelligence
"""

from __future__ import annotations

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

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Import the enhanced line monitoring system
from line_monitoring import TestingUtilities, line_tracker, monitor_lines_periodically
from line_monitoring import router as line_monitoring_router


# ── Enhanced Performance Monitoring (from Cursor) ──
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


# ── Intelligent Caching System (from Cursor) ──
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
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": self.get_hit_rate(),
            "total_accesses": sum(self.access_count.values()),
        }


# ── Game Prioritization (enhanced from Cursor) ──
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

        # Edge cents (normalized to 0-100 scale)
        edge_cents = game_data.get("edge_cents", 0)
        edge_score = min(edge_cents / 5.0, 100)  # Cap at 100 for 500¢ edges
        score += edge_score * self.priority_weights["edge_cents"]

        # Value rating
        value_rating = game_data.get("value_rating", "UNKNOWN")
        rating_scores = {"HIGH": 100, "MEDIUM": 60, "LOW": 30, "UNKNOWN": 0}
        rating_score = rating_scores.get(value_rating, 0)
        score += rating_score * self.priority_weights["value_rating"]

        # Time until game (closer games get higher priority)
        commence_time = game_data.get("commence_time")
        time_score = 50  # Default
        if commence_time:
            try:
                game_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                now = datetime.now(game_time.tzinfo)
                hours_until = (game_time - now).total_seconds() / 3600
                # Games starting soon get higher scores
                time_score = max(10, 100 - (hours_until * 2))
            except:
                pass
        score += time_score * self.priority_weights["time_until_game"]

        # Historical performance (placeholder for future ML integration)
        historical_score = 50
        score += historical_score * self.priority_weights["historical_performance"]

        return score


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


# ── Logging ──
log = logging.getLogger("gold-main-hybrid")
log.setLevel(logging.INFO)
if not logging.getLogger().handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logging.getLogger().addHandler(h)


# ── HYBRID Gold Standard MLB System ──
class HybridOptimizedGoldStandardMLBSystem:
    """Combines Cursor's optimization classes with production FastAPI integration."""

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

        # Initialize Cursor's optimization components
        self.cache = GameCache(cache_size)
        self.prioritizer = GamePrioritizer()
        self.monitor = PerformanceMonitor()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        log.info(
            "🚀 HybridOptimizedGoldStandardMLBSystem initialized – "
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
        """Main processing pipeline using Cursor's optimization strategies."""
        self.monitor.start_timer()

        # Step 1: Pre-filter and cache check
        valid_rows = await self._prefilter_and_cache(rows)
        log.info(
            "📊 Pre-filtered %d rows to %d valid opportunities",
            len(rows),
            len(valid_rows),
        )

        if not valid_rows:
            return []

        # Step 2: Prioritize games using Cursor's algorithm
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
                "📊 Performance: %d games in %.1fms (%.1f games/sec) → %d picks",
                len(valid_rows),
                metrics.processing_time_ms,
                metrics.games_per_second,
                len(final_picks),
            )

        return final_picks

    async def _prefilter_and_cache(self, rows: list[dict]) -> list[dict]:
        """Pre-filter using Cursor's caching strategy."""
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
            if self.cfg.enable_caching:
                cached_result = self.cache.get(game_id)
                if cached_result:
                    self.monitor.current_metrics.cache_hits += 1
                    valid_rows.append(cached_result)
                    continue
                else:
                    self.monitor.current_metrics.cache_misses += 1
                    self.cache.set(game_id, row)

            valid_rows.append(row)

        return valid_rows

    async def _prioritize_games(self, rows: list[dict]) -> list[dict]:
        """Prioritize games using Cursor's scoring algorithm."""
        scored_games = []

        for row in rows:
            priority = self.prioritizer.calculate_priority_score(row)
            scored_games.append((priority, row))

        # Sort by priority score (highest first)
        scored_games.sort(key=lambda x: x[0], reverse=True)

        return [row for _, row in scored_games]

    async def _process_optimized_batches(self, rows: list[dict]) -> list[dict]:
        """Process using Cursor's load-balanced batching."""
        if not rows:
            return []

        # Create balanced batches
        batches = self._create_balanced_batches(rows)
        log.info(
            "🔄 Processing %d games in %d balanced batches", len(rows), len(batches)
        )

        # Process batches concurrently
        tasks = [
            self._process_batch_async(batch_id, batch)
            for batch_id, batch in enumerate(batches)
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_picks = []
        for result in batch_results:
            if isinstance(result, Exception):
                log.error("❌ Batch processing error: %s", result)
                continue
            if isinstance(result, list):
                all_picks.extend(result)

        return all_picks

    def _create_balanced_batches(self, rows: list[dict]) -> list[list[dict]]:
        """Create load-balanced batches based on priority scores."""
        if len(rows) <= self.cfg.batch_size:
            return [rows]

        batches = []
        current_batch = []
        current_batch_priority = 0

        for row in rows:
            priority = self.prioritizer.calculate_priority_score(row)

            # Start new batch if current one is full or too heavy
            if (
                len(current_batch) >= self.cfg.batch_size
                or current_batch_priority > 200
            ):  # Priority threshold
                if current_batch:
                    batches.append(current_batch)
                current_batch = [row]
                current_batch_priority = priority
            else:
                current_batch.append(row)
                current_batch_priority += priority

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _process_batch_async(
        self, batch_id: int, batch_rows: list[dict]
    ) -> list[dict]:
        """Process a single batch asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._process_batch_sync, batch_id, batch_rows
        )

    def _process_batch_sync(self, batch_id: int, batch_rows: list[dict]) -> list[dict]:
        """Synchronous batch processing."""
        picks = []

        for row in batch_rows:
            try:
                game_picks = self._process_single_game(row)
                picks.extend(game_picks)
            except Exception as e:
                log.error("❌ Error processing game %s: %s", row.get("game_id"), e)
                continue

        log.debug("✅ Batch %d completed: %d picks generated", batch_id, len(picks))
        return picks

    def _process_single_game(self, row: dict) -> list[dict]:
        """Process individual game with enhanced data."""
        game_id = row.get("game_id", "unknown")
        home_team = row.get("home_team", "Unknown")
        away_team = row.get("away_team", "Unknown")
        home_price = row.get("home_price")
        away_price = row.get("away_price")
        edge_cents = row.get("edge_cents", 0)
        value_rating = row.get("value_rating", "UNKNOWN")
        commence_time = row.get("commence_time")
        detected_at = row.get("detected_at")

        picks = []
        priority_score = self.prioritizer.calculate_priority_score(row)

        # Create picks for valid sides
        if home_price is not None and value_rating in ["HIGH", "MEDIUM", "LOW"]:
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
                "detected_at": detected_at,
                "priority_score": priority_score,
                "home_team": home_team,
                "away_team": away_team,
            }
            picks.append(home_pick)

        if (
            away_price is not None
            and away_price != home_price
            and value_rating in ["HIGH", "MEDIUM", "LOW"]
        ):
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
                "detected_at": detected_at,
                "priority_score": priority_score,
                "home_team": home_team,
                "away_team": away_team,
            }
            picks.append(away_pick)

        return picks

    async def save_recommendations_async(self, picks: list[dict]) -> str:
        """Async file saving with enhanced metrics."""

        def _save_sync():
            date_str = datetime.utcnow().date().isoformat()
            out_dir = Path("data/picks")
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = out_dir / f"{date_str}_picks.json"

            save_data = {
                "date": date_str,
                "picks": picks,
                "performance_metrics": {
                    "total_games": self.monitor.current_metrics.total_games,
                    "processing_time_ms": self.monitor.current_metrics.processing_time_ms,
                    "games_per_second": self.monitor.current_metrics.games_per_second,
                    "cache_hits": self.monitor.current_metrics.cache_hits,
                    "cache_misses": self.monitor.current_metrics.cache_misses,
                    "cache_hit_rate": self.cache.get_hit_rate(),
                    "total_picks": len(picks),
                },
            }

            with fname.open("w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2)

            return str(fname)

        loop = asyncio.get_event_loop()
        filename = await loop.run_in_executor(self.executor, _save_sync)
        log.info(
            "💾 Saved %d picks with performance metrics to %s", len(picks), filename
        )
        return filename

    def get_performance_stats(self) -> dict:
        """Get comprehensive performance statistics."""
        cache_stats = self.cache.stats()

        return {
            "processing": {
                "total_games": self.monitor.current_metrics.total_games,
                "processing_time_ms": self.monitor.current_metrics.processing_time_ms,
                "games_per_second": self.monitor.current_metrics.games_per_second,
                "total_picks": self.monitor.current_metrics.total_picks,
                "cache_hits": self.monitor.current_metrics.cache_hits,
                "cache_misses": self.monitor.current_metrics.cache_misses,
            },
            "caching": cache_stats,
            "configuration": {
                "batch_size": self.cfg.batch_size,
                "max_workers": self.cfg.max_workers,
                "min_edge_cents": self.cfg.min_edge_cents,
                "max_opportunities": self.cfg.max_opportunities,
                "cache_size": self.cfg.cache_size,
            },
        }

    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)
        self.cache.cache.clear()


# FastAPI app setup
app = FastAPI(
    title="Hybrid Optimized Gold Standard MLB System with Line Monitoring Intelligence",
    description="Best of both worlds: Advanced optimizations + Production ready + Real-time betting intelligence",
)

# Initialize hybrid system
system = HybridOptimizedGoldStandardMLBSystem(
    bankroll=500.0,
    base_unit_size=5.0,
    max_units=5,
    confidence_threshold=0.55,
    max_opportunities=50,
    min_edge_cents=8,
    batch_size=25,  # Optimal for 15+ games
    max_workers=4,  # Adjust based on CPU
    cache_size=1000,  # Large cache for repeated games
)

# Add the line monitoring router
print("Including router, id:", id(line_monitoring_router))
app.include_router(line_monitoring_router)


# Serve the dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    try:
        dashboard_path = Path("templates/dashboard.html")
        return dashboard_path.read_text()
    except FileNotFoundError:
        return HTMLResponse(
            """
            <html>
            <head><title>Dashboard Setup Required</title></head>
            <body>
                <h1>🚀 MLB Betting Intelligence Dashboard</h1>
                <h2>Setup Required</h2>
                <p>To enable the dashboard, save the HTML dashboard as <code>templates/dashboard.html</code></p>
                <p>Your line monitoring system is running at:</p>
                <ul>
                    <li><a href="/line-monitoring/health">System Health</a></li>
                    <li><a href="/line-monitoring/recommendations">Betting Recommendations</a></li>
                    <li><a href="/line-monitoring/dashboard">Line Monitoring Dashboard</a></li>
                </ul>
            </body>
            </html>
            """,
            status_code=200,
        )


@app.post("/opportunities")
async def opportunities(data: dict):
    """Process betting opportunities with hybrid optimization."""
    start_time = time.time()
    log.info("📨 Received API request with data keys: %s", list(data.keys()))

    # Handle different data structures
    rows = []
    if "rows" in data:
        rows = data["rows"]
    elif isinstance(data, list):
        rows = data
    else:
        for key, value in data.items():
            if isinstance(value, list) and value:
                rows = value
                break

    log.info("📊 Processing %d opportunity rows with hybrid system", len(rows))

    if not rows:
        log.warning("⚠️ No rows found in request data")
        return {"status": "empty", "opportunities": [], "message": "No data received"}

    try:
        # Process with hybrid optimization
        picks = await system.analyze_opportunities_concurrently(rows)

        # Record line monitoring snapshots for each pick
        for pick in picks:
            game_id = pick.get("game_id")
            if not isinstance(game_id, str) or not game_id:
                continue  # skip invalid game_id
            snapshot = {
                "game_id": game_id,
                "edge_cents": pick.get("edge_cents"),
                "fanduel_odds": pick.get("fanduel_odds"),
                "commence_time": pick.get("commence_time"),
                "value_rating": pick.get("value_rating"),
                "home_team": pick.get("home_team"),
                "away_team": pick.get("away_team"),
                "timestamp": datetime.utcnow().isoformat(),
            }
            line_tracker.record_snapshot(game_id, snapshot)

        # Save picks asynchronously
        filename = None
        if picks:
            filename = await system.save_recommendations_async(picks)

        # Get performance stats
        perf_stats = system.get_performance_stats()

        response = {
            "status": "success",
            "opportunities": picks,
            "count": len(picks),
            "timestamp": datetime.utcnow().isoformat(),
            "file": filename,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "performance": perf_stats["processing"],
            "cache_stats": perf_stats["caching"],
        }

        log.info(
            "✅ Returning %d opportunities (%.1fms processing)",
            len(picks),
            (time.time() - start_time) * 1000,
        )
        return response

    except Exception as e:
        log.error("❌ Error processing opportunities: %s", str(e))
        return {
            "status": "error",
            "opportunities": [],
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
        }


@app.get("/performance")
async def get_performance_stats():
    """Get detailed performance statistics."""
    return {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        **system.get_performance_stats(),
    }


@app.get("/health")
async def health():
    """Enhanced health check."""
    perf_stats = system.get_performance_stats()

    return {
        "status": "healthy",
        "service": "Hybrid Optimized Gold Standard MLB System with Line Monitoring",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0 - Hybrid (Cursor optimizations + FastAPI + Line Monitoring Intelligence)",
        "config": perf_stats["configuration"],
        "performance_summary": {
            "games_per_second": perf_stats["processing"]["games_per_second"],
            "cache_hit_rate": perf_stats["caching"]["hit_rate"],
            "total_cache_size": perf_stats["caching"]["size"],
        },
    }


@app.on_event("startup")
async def startup_event():
    """Enhanced startup with line monitoring integration."""
    print("🚀 Starting Enhanced MLB Betting Intelligence System...")
    print(
        "⚡ Features: Advanced caching, Game prioritization, Load balancing, Line monitoring"
    )

    # Run basic system tests
    print("🧪 Running line monitoring system tests...")
    try:
        test_results = TestingUtilities.run_basic_tests()

        passed_tests = sum(1 for test in test_results if test["passed"])
        total_tests = len(test_results)

        print(f"✅ System tests: {passed_tests}/{total_tests} passed")

        if passed_tests >= total_tests - 1:  # Allow 1 test failure
            print("📊 Starting line monitoring intelligence...")
            # Start the background monitoring task
            asyncio.create_task(monitor_lines_periodically(system, interval_minutes=15))
            print("✅ Line monitoring system active!")
            print("🎯 Your betting intelligence system is now tracking line movements!")
        else:
            print(
                "⚠️ Some tests failed, but system will continue. Check logs for details."
            )
            for test in test_results:
                if not test["passed"]:
                    print(f"❌ {test['test']}: {test.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"⚠️ Line monitoring tests failed: {e}")
        print("✅ Main system will continue without line monitoring")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("🔄 Shutting down systems...")
    await system.cleanup()
    print("✅ Cleanup complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Hybrid Optimized Gold Standard MLB System with Line Monitoring Intelligence",
        "version": "3.0 - Complete Betting Intelligence Platform",
        "features": [
            "Cursor's advanced optimization algorithms",
            "Production-ready FastAPI integration",
            "Intelligent caching with LRU eviction",
            "Game prioritization with weighted scoring",
            "Load-balanced batch processing",
            "Comprehensive performance monitoring",
            "Real-time line movement tracking",
            "Intelligent betting timing advice",
            "Email alert system",
            "Professional web dashboard",
            "Background monitoring system",
        ],
        "endpoints": {
            "/opportunities": "POST - Submit betting opportunities",
            "/performance": "GET - Detailed performance stats",
            "/health": "GET - Health check with metrics",
            "/dashboard": "GET - Professional web dashboard",
            "/line-monitoring/*": "GET - Line monitoring endpoints",
            "/docs": "GET - API documentation",
        },
        "line_monitoring_endpoints": {
            "/line-monitoring/dashboard": "GET - Line monitoring dashboard data",
            "/line-monitoring/recommendations": "GET - Current betting recommendations",
            "/line-monitoring/health": "GET - Line monitoring health check",
            "/line-monitoring/alerts/recent": "GET - Recent alerts",
            "/line-monitoring/test-email": "POST - Test email alerts",
        },
    }


if __name__ == "__main__":
    log.info("🚀 Starting Complete MLB Betting Intelligence System on port 8767")
    log.info(
        "⚡ Features: Advanced optimization + Line monitoring + Real-time intelligence"
    )
    log.info("🎯 Dashboard available at: http://localhost:8767/dashboard")
    log.info("📊 Line monitoring at: http://localhost:8767/line-monitoring/dashboard")
    uvicorn.run(app, host="0.0.0.0", port=8767, reload=True)
