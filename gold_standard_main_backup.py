#!/usr/bin/env python3
# gold_standard_main.py – "real edge" edition
"""
Scans flattened Odds-API rows, finds FanDuel money-line edges, returns picks via API.
Runs as a FastAPI server on port 8767.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI

# ── logging ──────────────────────────────────────────────────────
log = logging.getLogger("gold-main")
log.setLevel(logging.INFO)
if not logging.getLogger().handlers:  # avoid double-handlers in reloads
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logging.getLogger().addHandler(h)


# ── configuration dataclass ──────────────────────────────────────
class GoldStandardConfig:
    def __init__(
        self,
        bankroll: float = 500,
        base_unit_size: float = 5,
        max_units: int = 5,
        confidence_threshold: float = 0.55,
        max_opportunities: int = 50,
        min_edge_cents: int = 8,
        batch_size: int = 100,  # Process games in batches
        max_workers: int = 4,  # Thread pool size for I/O operations
    ) -> None:
        self.bankroll = bankroll
        self.base_unit_size = base_unit_size
        self.max_units = max_units
        self.confidence_threshold = confidence_threshold
        self.max_opportunities = max_opportunities
        self.min_edge_cents = min_edge_cents
        self.batch_size = batch_size
        self.max_workers = max_workers


# ╔══════════════════════════════════════════════════════════════╗
# ║  Core helper: choose best (most-profitable) American odds    ║
# ╚══════════════════════════════════════════════════════════════╝
def _better(price_a: int | None, price_b: int | None) -> bool:
    """
    Return True if A is strictly *better* for the bettor than B
    (i.e. pays more on a win).
    +EV ordering for American odds:
        • any positive number beats any negative
        • among positives, higher is better  (+150 > +130)
        • among negatives, *less negative* is better (-105 > -120)
    """
    if price_a is None:
        return False
    if price_b is None:
        return True
    # A positive & B negative → A better
    if price_a > 0 >= price_b:
        return True
    if price_b > 0 >= price_a:
        return False
    # both positive: greater is better
    if price_a > 0 and price_b > 0:
        return price_a > price_b
    # both negative: -105 beats -120  (numerically larger)
    return price_a > price_b


# ╔══════════════════════════════════════════════════════════════╗
# ║        G o l d S t a n d a r d   M L B   S y s t e m         ║
# ╚══════════════════════════════════════════════════════════════╝
class GoldStandardMLBSystem:
    """Single-file 'edge finder' that the API calls."""

    def __init__(
        self,
        bankroll: float,
        base_unit_size: float,
        max_units: int,
        confidence_threshold: float,
        max_opportunities: int,
        min_edge_cents: int = 8,
        batch_size: int = 100,
        max_workers: int = 4,
    ):
        self.cfg = GoldStandardConfig(
            bankroll=bankroll,
            base_unit_size=base_unit_size,
            max_units=max_units,
            confidence_threshold=confidence_threshold,
            max_opportunities=max_opportunities,
            min_edge_cents=min_edge_cents,
            batch_size=batch_size,
            max_workers=max_workers,
        )
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._cache = {}  # Simple cache for repeated data
        log.info(
            "📊 GoldStandardMLBSystem initialised – "
            "bankroll=$%.2f  baseUnit=$%.2f conf≥%.2f  maxOpp=%d  minEdge=%d¢  batchSize=%d",
            bankroll,
            base_unit_size,
            confidence_threshold,
            max_opportunities,
            min_edge_cents,
            batch_size,
        )
        self.recommendations: list[dict] = []

    # ─────────────── PUBLIC API expected by launcher ─────────────
    async def analyze_opportunities_concurrently(self, rows: list[dict]) -> list[dict]:
        """Process rows and return value picks via API with parallel processing."""
        return await self._find_edges_parallel(rows)

    async def save_recommendations(self, picks: list[dict]) -> str:
        """Write to data/picks/YYYY-MM-DD_picks.json & return its filename (async)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool, self._save_recommendations_sync, picks
        )

    def _save_recommendations_sync(self, picks: list[dict]) -> str:
        """Synchronous version of save_recommendations for thread pool execution."""
        date_str = datetime.utcnow().date().isoformat()
        out_dir = Path("data/picks")
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{date_str}_picks.json"
        json.dump(
            {"date": date_str, "picks": picks},
            fname.open("w", encoding="utf-8"),
            indent=2,
        )
        log.info("💾 Saved %d picks to %s", len(picks), fname)
        return str(fname)

    # ──────────────── internal workers ───────────────────────────
    def _load_latest_odds(self, file_path: str | Path) -> list[dict]:
        """Read latest_odds.json and return a flat list of row-dicts."""
        p = Path(file_path)
        if not p.exists():
            log.warning("Odds file %s not found – returning empty list", p)
            return []

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # ① already correct
        if isinstance(data, list):
            return data

        # ② wrapped: {"odds": [...]}, {"rows": [...]}, {"data": [...]}
        for key in ("odds", "rows", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]

        # ③ bad shape → fail soft
        log.error("latest_odds.json has unexpected shape: %s", type(data))
        return []

    async def execute_today(self) -> None:
        """Load odds file, find edges, and fill self.recommendations."""
        log.info(
            "▶️  execute_today called (%s)",
            datetime.utcnow().isoformat(timespec="seconds"),
        )

        rows = self._load_latest_odds("latest_odds.json")
        log.info("📄 latest_odds.json loaded (%d rows)", len(rows))

        if not rows:
            log.warning("No odds rows found – aborting execute_today()")
            return

        # filter out anything that isn't a dict
        clean_rows = [r for r in rows if isinstance(r, dict)]
        if len(clean_rows) != len(rows):
            log.warning("Skipped %d malformed rows", len(rows) - len(clean_rows))

        self.recommendations = await self._find_edges_parallel(clean_rows)
        self.recommendations = self.recommendations[: self.cfg.max_opportunities]

    def _prefilter_rows(self, rows: list[dict]) -> list[dict]:
        """Quick pre-filter to remove obviously invalid rows."""
        filtered = []
        for r in rows:
            if not isinstance(r, dict):
                continue

            # Quick checks for FanDuel and minimum edge
            bookmaker = r.get("bookmaker", "").lower()
            edge_cents = r.get("edge_cents", 0)

            if bookmaker == "fanduel" and edge_cents >= self.cfg.min_edge_cents:
                filtered.append(r)

        return filtered

    def _process_game_batch(self, batch: list[dict]) -> list[dict]:
        """Process a batch of games and return picks."""
        picks = []

        for r in batch:
            # Extract key fields
            game_id = r.get("game_id", "unknown")
            home_team = r.get("home_team", "Unknown")
            away_team = r.get("away_team", "Unknown")
            home_price = r.get("home_price")
            away_price = r.get("away_price")
            edge_cents = r.get("edge_cents", 0)
            value_rating = r.get("value_rating", "UNKNOWN")
            commence_time = r.get("commence_time")

            # Determine which side has the edge based on value_rating
            if value_rating in ["HIGH", "MEDIUM", "LOW"]:
                # Create picks for both sides if they have value
                # For home team
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
                        "detected_at": r.get("detected_at"),
                    }
                    picks.append(home_pick)

                # For away team (if different odds/edge)
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
                        "detected_at": r.get("detected_at"),
                    }
                    picks.append(away_pick)

        return picks

    async def _find_edges_parallel(self, rows: list[dict]) -> list[dict]:
        """Find value betting opportunities with parallel processing."""
        start_time = time.time()
        log.info("🔍 Received %d rows for edge analysis", len(rows))

        # Pre-filter to reduce workload
        filtered_rows = self._prefilter_rows(rows)
        log.info("📊 Pre-filtered to %d valid rows", len(filtered_rows))

        if not filtered_rows:
            return []

        # Group by game_id for better organization
        games_by_id = defaultdict(list)
        for row in filtered_rows:
            game_id = row.get("game_id", "unknown")
            games_by_id[game_id].append(row)

        log.info("🎮 Processing %d unique games", len(games_by_id))

        # Process games in batches
        all_picks = []
        game_batches = list(games_by_id.values())

        # Process batches concurrently
        tasks = []
        for i in range(0, len(game_batches), self.cfg.batch_size):
            batch = game_batches[i : i + self.cfg.batch_size]
            task = asyncio.create_task(self._process_batch_async(batch))
            tasks.append(task)

        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        for result in batch_results:
            if isinstance(result, list):
                all_picks.extend(result)
            elif isinstance(result, Exception):
                log.error("❌ Batch processing error: %s", result)

        # Sort by biggest edge first
        all_picks.sort(key=lambda p: p["edge_cents"], reverse=True)

        # Log summary
        processing_time = time.time() - start_time
        log.info(
            "🏆 Found %d total value bets ≥ %d¢ in %.2fs",
            len(all_picks),
            self.cfg.min_edge_cents,
            processing_time,
        )

        # Log top picks for debugging
        for i, pick in enumerate(all_picks[:5]):
            log.info(
                "📊 Pick #%d: %s (%s) - %s edge %d¢",
                i + 1,
                pick["team"],
                pick["side"],
                pick["value_rating"],
                pick["edge_cents"],
            )

        if len(all_picks) > 5:
            log.info("📊 ... and %d more picks", len(all_picks) - 5)

        return all_picks

    async def _process_batch_async(self, batch: list[list[dict]]) -> list[dict]:
        """Process a batch of games asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._process_game_batch,
            [game for games in batch for game in games],
        )

    def _find_edges(self, rows: list[dict]) -> list[dict]:
        """Legacy synchronous method - kept for compatibility."""
        return asyncio.run(self._find_edges_parallel(rows))


# FastAPI app setup
app = FastAPI(title="Gold Standard MLB System", description="Edge detection API")
system = GoldStandardMLBSystem(
    bankroll=500.0,
    base_unit_size=5.0,
    max_units=5,
    confidence_threshold=0.55,
    max_opportunities=50,
    min_edge_cents=8,
    batch_size=50,  # Increase for more games
    max_workers=4,  # Increase for more CPU cores
)


@app.post("/opportunities")
async def opportunities(data: dict):
    """Process betting opportunities and return value picks."""
    log.info("📨 Received API request with data keys: %s", list(data.keys()))

    # Handle different possible data structures
    rows = []
    if "rows" in data:
        rows = data["rows"]
    elif isinstance(data, list):
        rows = data
    else:
        # Try to find array in any top-level key
        for key, value in data.items():
            if isinstance(value, list) and value:
                rows = value
                break

    log.info("📊 Processing %d opportunity rows", len(rows))

    if not rows:
        log.warning("⚠️ No rows found in request data")
        return {"status": "empty", "opportunities": [], "message": "No data received"}

    try:
        start_time = time.time()
        picks = await system.analyze_opportunities_concurrently(rows)
        processing_time = time.time() - start_time

        # Save picks to file asynchronously
        filename = None
        if picks:
            filename = await system.save_recommendations(picks)
            log.info("💾 Saved %d picks to %s", len(picks), filename)

        response = {
            "status": "success",
            "opportunities": picks,
            "count": len(picks),
            "processing_time_ms": round(processing_time * 1000, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "file": filename,
        }

        log.info(
            "✅ Returning %d opportunities in %.2fms",
            len(picks),
            processing_time * 1000,
        )
        return response

    except Exception as e:
        log.error("❌ Error processing opportunities: %s", str(e))
        return {
            "status": "error",
            "opportunities": [],
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Gold Standard MLB System",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "min_edge_cents": system.cfg.min_edge_cents,
            "max_opportunities": system.cfg.max_opportunities,
            "bankroll": system.cfg.bankroll,
            "batch_size": system.cfg.batch_size,
            "max_workers": system.cfg.max_workers,
        },
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Gold Standard MLB System",
        "version": "1.0",
        "endpoints": {
            "/opportunities": "POST - Submit betting opportunities for analysis",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        },
    }


if __name__ == "__main__":
    log.info("🚀 Starting Gold Standard MLB System on port 8767")
    uvicorn.run(app, host="0.0.0.0", port=8767, reload=True)
