#!/usr/bin/env python3
"""
ultimate_odds_fetcher.py
~~~~~~~~~~~~~~~~~~~~~~~~
Real-time / historical odds aggregation for the Ultimate AI Betting System.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path  # ← ADDED: This was missing from your original file
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# ─── logging ────────────────────────────────────────────────────────
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    log.addHandler(h)

# ─── config / env ───────────────────────────────────────────────────
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")  # primary
SPORTSDATA_API_KEY = os.getenv("SPORTSDATA_API_KEY")  # optional
ODDSJAM_API_KEY = os.getenv("ODDSJAM_API_KEY")  # optional

CACHE_DIR = "cache"  # local disk cache for dev / rate-limit protection
os.makedirs(CACHE_DIR, exist_ok=True)


# ─── helpers ────────────────────────────────────────────────────────
def _iso_to_local(iso: str) -> str:
    """Keep full ISO but slice ms for readability."""
    return iso.replace("Z", "")[:19]


# ─── dataclass ──────────────────────────────────────────────────────
@dataclass(slots=True)
class OddsData:
    game_id: str
    commence_time: str
    home_team: str
    away_team: str

    moneyline_home: int | None = None
    moneyline_away: int | None = None
    spread_home: float | None = None
    spread_home_odds: int | None = None
    spread_away: float | None = None
    spread_away_odds: int | None = None
    total: float | None = None
    over_odds: int | None = None
    under_odds: int | None = None

    bookmaker: str = ""
    source: str = ""
    last_update: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds")
    )

    # merge incoming update keeping *best* price
    def merge(self, other: OddsData) -> OddsData:
        def better(x: int | None, y: int | None, is_fave: bool) -> int | None:
            # More positive for dogs, more negative (larger abs) for faves
            if x is None:
                return y
            if y is None:
                return x
            return (
                y if ((x < 0 and y < 0 and y > x) or (x > 0 and y > 0 and y > x)) else x
            )

        self.moneyline_home = better(
            self.moneyline_home,
            other.moneyline_home,
            self.moneyline_home and self.moneyline_home < 0,
        )
        self.moneyline_away = better(
            self.moneyline_away,
            other.moneyline_away,
            self.moneyline_away and self.moneyline_away < 0,
        )
        # spreads / totals: keep first seen (could add logic later)
        return self

    # JSON-serialisable
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── tenacity wrapper for HTTP calls ────────────────────────────────
def _retryable(func):
    return retry(
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type(aiohttp.ClientError)
        | retry_if_exception_type(asyncio.TimeoutError),
        reraise=True,
    )(func)


# ─── main fetcher class ─────────────────────────────────────────────
class UltimateOddsFetcher:
    preferred_bookmakers = [
        "draftkings",
        "fanduel",
        "betmgm",
        "caesars",
        "betrivers",
        "pointsbet",
        "betus",
        "bovada",
    ]

    def __init__(self):
        ssl_ctx = ssl.create_default_context()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(ssl=ssl_ctx),
        )

    # ── context manager helpers ───────────────────
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self._session.close()

    # ── public API ────────────────────────────────
    async def fetch_mlb_odds(self, slate: date) -> list[OddsData]:
        # 0. check local cache
        cache_path = Path(CACHE_DIR) / f"odds_{slate}.json"
        if cache_path.exists():
            log.debug("Using cached odds file %s", cache_path)
            return [OddsData(**o) for o in json.loads(cache_path.read_text())]

        collectors = [
            self._from_the_odds_api,
            self._from_sportsdata_io,
            self._from_oddsjam,
        ]
        merged: dict[str, OddsData] = {}

        for fn in collectors:
            try:
                rows = await fn(slate)
                for o in rows:
                    merged.setdefault(o.game_id, o).merge(o)
                if merged:  # stop after first successful source
                    break
            except Exception as e:
                log.warning("Source %s failed: %s", fn.__name__, e)

        if not merged:
            log.warning("All sources failed – generating mock odds")
            merged = {o.game_id: o for o in await self._mock_odds(slate)}

        # write cache
        cache_path.write_text(
            json.dumps([o.to_dict() for o in merged.values()], indent=2)
        )
        return list(merged.values())

    # ── individual source methods ─────────────────
    @_retryable
    async def _from_the_odds_api(self, slate: date) -> list[OddsData]:
        if not THE_ODDS_API_KEY:
            raise RuntimeError("THE_ODDS_API_KEY missing")
        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = dict(
            apiKey=THE_ODDS_API_KEY,
            regions="us",
            markets="h2h,spreads,totals",
            oddsFormat="american",
            bookmakers=",".join(self.preferred_bookmakers[:6]),
        )
        log.debug("TheOddsAPI → %s", url)
        async with self._session.get(url, params=params) as r:
            if r.status != 200:
                raise RuntimeError(f"TheOddsAPI status {r.status}")
            raw = await r.json()
        out: list[OddsData] = []
        for g in raw:
            g_date = datetime.fromisoformat(
                g["commence_time"].replace("Z", "+00:00")
            ).date()
            if g_date != slate:
                continue
            for bm in g["bookmakers"]:
                o = OddsData(
                    game_id=f'{g["id"]}_{bm["key"]}',
                    commence_time=_iso_to_local(g["commence_time"]),
                    home_team=g["home_team"],
                    away_team=g["away_team"],
                    bookmaker=bm["title"],
                    source="the_odds_api",
                    last_update=_iso_to_local(bm["last_update"]),
                )
                for m in bm["markets"]:
                    if m["key"] == "h2h":
                        for oc in m["outcomes"]:
                            if oc["name"] == o.home_team:
                                o.moneyline_home = oc["price"]
                            if oc["name"] == o.away_team:
                                o.moneyline_away = oc["price"]
                    elif m["key"] == "spreads":
                        for oc in m["outcomes"]:
                            if oc["name"] == o.home_team:
                                o.spread_home = oc["point"]
                                o.spread_home_odds = oc["price"]
                            if oc["name"] == o.away_team:
                                o.spread_away = oc["point"]
                                o.spread_away_odds = oc["price"]
                    elif m["key"] == "totals":
                        for oc in m["outcomes"]:
                            if oc["name"] == "Over":
                                o.total = oc["point"]
                                o.over_odds = oc["price"]
                            if oc["name"] == "Under":
                                o.total = oc["point"]
                                o.under_odds = oc["price"]
                out.append(o)
        return out

    @_retryable
    async def _from_sportsdata_io(self, slate: date) -> list[OddsData]:
        if not SPORTSDATA_API_KEY:
            return []
        url = f"https://api.sportsdata.io/v3/mlb/odds/json/GameOddsByDate/{slate}"
        headers = {"Ocp-Apim-Subscription-Key": SPORTSDATA_API_KEY}
        log.debug("SportsDataIO → %s", url)
        async with self._session.get(url, headers=headers) as r:
            if r.status != 200:
                raise RuntimeError(f"SportsDataIO status {r.status}")
            raw = await r.json()
        rows: list[OddsData] = []
        for g in raw:
            rows.append(
                OddsData(
                    game_id=str(g["GameID"]),
                    commence_time=_iso_to_local(g["DateTime"]),
                    home_team=g["HomeTeam"],
                    away_team=g["AwayTeam"],
                    moneyline_home=g.get("HomeMoneyLine"),
                    moneyline_away=g.get("AwayMoneyLine"),
                    total=g.get("Total"),
                    over_odds=g.get("OverPayout"),
                    under_odds=g.get("UnderPayout"),
                    bookmaker="Consensus",
                    source="sportsdata_io",
                )
            )
        return rows

    async def _from_oddsjam(self, slate: date) -> list[OddsData]:
        # Placeholder – real call requires paid plan
        return []

    async def _mock_odds(self, slate: date) -> list[OddsData]:
        log.debug("Building mock odds for %s", slate)
        demo = [
            (
                "CHW",
                "TOR",
                +130,
                -150,
                8.5,
            ),  # White Sox @ Blue Jays (from today's actual games)
            ("BOS", "SF", +105, -115, 9.0),  # Red Sox @ Giants
        ]
        out = []
        for i, (a, h, aml, hml, tot) in enumerate(demo, 1):
            out.append(
                OddsData(
                    game_id=f"mock_{slate}_{i}",
                    commence_time=f"{slate}T19:05:00",
                    away_team=a,
                    home_team=h,
                    moneyline_away=aml,
                    moneyline_home=hml,
                    total=tot,
                    over_odds=-110,
                    under_odds=-110,
                    bookmaker="FanDuel",
                    source="mock",
                )
            )
        return out


# ─── orchestrator helper ────────────────────────────────────────────
async def fetch_and_integrate_odds(target: date) -> dict[str, Any]:
    async with UltimateOddsFetcher() as f:
        rows = await f.fetch_mlb_odds(target)
        # Save canonical file
        path = Path("data")
        path.mkdir(exist_ok=True)
        out_file = path / f"odds_{target}.json"
        out_file.write_text(json.dumps([o.to_dict() for o in rows], indent=2))
        log.info("Saved %s rows → %s", len(rows), out_file)
        return {"rows": rows, "file": str(out_file)}


# ─── quick manual test ─────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import sys

    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    result = asyncio.run(fetch_and_integrate_odds(target))
    print(f"✅ Fetched {len(result['rows'])} odds records")
    print(f"💾 Saved to: {result['file']}")
