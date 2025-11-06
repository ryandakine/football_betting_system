# fixed_gold_standard_mlb_system.py
"""
Gold-Standard MLB – production pipeline
• Accepts ODDS_API_KEY or THE_ODDS_API_KEY
• Accepts CLAUDE_API_KEY or ANTHROPIC_API_KEY
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
from datetime import date, datetime, timezone
from typing import Any, Dict, List

import certifi

import aiohttp
from gold_standard_analyzer import GoldStandardAIAnalyzer

logger = logging.getLogger(__name__)


class FixedGoldStandardMLBSystem:
    # ───────────────────────── constructor ──────────────────────────
    def __init__(
        self,
        bankroll: float,
        base_unit_size: float,
        *,
        slate_date: date,
        max_opportunities: int | None = None,
        batch_size: int | None = None,
        parallel_requests: int | None = None,
    ) -> None:
        self.bankroll = bankroll
        self.base_unit_size = base_unit_size
        self.slate_date = slate_date
        self.max_opportunities = max_opportunities
        self.batch_size = batch_size or 20
        self.parallel_requests = parallel_requests or 5

        # API keys -----------------------------------------------------------
        self.odds_api_key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
        if not self.odds_api_key:
            raise RuntimeError("Missing ODDS_API_KEY (or THE_ODDS_API_KEY)")

        self.ai_keys = {
            "claude_api_key": (
                os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            ),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "grok_api_key": os.getenv("GROK_API_KEY"),
        }

        logger.info(
            "System ready • date=%s  max=%s  batch=%s  parallel=%s",
            self.slate_date,
            self.max_opportunities or "∞",
            self.batch_size,
            self.parallel_requests,
        )

    # ─────────────────────── public pipeline ─────────────────────────
    async def run_fixed_pipeline(self) -> dict[str, Any]:
        """Fetch odds → AI analysis → return recommendations."""
        opportunities = await self._fetch_opportunities()
        if not opportunities:
            logger.warning("No markets pulled for %s", self.slate_date)
            return {"total_opportunities": 0, "recommendations": []}

        if self.max_opportunities:
            opportunities = opportunities[: self.max_opportunities]

        async with GoldStandardAIAnalyzer(self.ai_keys) as analyzer:
            recs = await analyzer.analyze_opportunities_concurrently(opportunities)

        return {
            "total_opportunities": len(opportunities),
            "recommendations": recs,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "slate_date": self.slate_date.isoformat(),
        }

    # ───────────────────── fetch opportunities ──────────────────────
    async def _fetch_opportunities(self) -> list[dict]:
        """Download MLB odds for the slate date and flatten into dicts."""
        logger.info("Fetching odds for %s …", self.slate_date)

        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = {
            "apiKey": self.odds_api_key,
            "sport": "baseball_mlb",
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }

        # Proper SSL context for aiohttp
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)

        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"Odds API error {resp.status}: {await resp.text()}"
                    )
                games = await resp.json()

        # ── Flatten games → opportunity dicts ─────────────────────────
        opps: list[dict] = []
        for g in games:
            gdate = datetime.fromisoformat(
                g["commence_time"].replace("Z", "+00:00")
            ).date()
            if gdate != self.slate_date:
                continue

            home, away = g["home_team"], g["away_team"]
            gid = g["id"]

            for book in g.get("bookmakers", []):
                bookmaker = book["title"]
                for mkt in book.get("markets", []):
                    key = mkt["key"]  # h2h / spreads / totals
                    for out in mkt["outcomes"]:
                        opps.append(
                            {
                                "game_id": gid,
                                "game_context": f"{away} @ {home}",
                                "commence_time": g["commence_time"],
                                "home_team": home,
                                "away_team": away,
                                "team": out["name"],
                                "bet_type": key if key != "spreads" else "spread",
                                "market": key,
                                "odds": out["price"],
                                "point": out.get("point"),
                                "selection": out["name"],
                                "bookmaker": bookmaker,
                            }
                        )

        logger.info("Pulled %d raw opportunities", len(opps))
        return opps
