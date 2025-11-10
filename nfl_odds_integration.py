#!/usr/bin/env python3
"""
NFL Odds Integration for TNF Predictions
Wires ultimate_odds_fetcher into the referee conspiracy pipeline
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Install with: pip install python-dotenv")

DATA_DIR = Path(os.getenv("REF_CONSPIRACY_DATA_DIR", "data/referee_conspiracy"))
ODDS_CACHE_DIR = DATA_DIR / "odds_cache"
ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")


@dataclass
class NFLGameOdds:
    """NFL game odds structure"""
    game_id: str
    commence_time: str
    home_team: str
    away_team: str
    
    # Moneylines
    moneyline_home: Optional[int] = None
    moneyline_away: Optional[int] = None
    
    # Spreads
    spread_home: Optional[float] = None
    spread_home_odds: Optional[int] = None
    spread_away: Optional[float] = None
    spread_away_odds: Optional[int] = None
    
    # Totals
    total: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    
    # Metadata
    bookmaker: str = ""
    source: str = ""
    week: Optional[int] = None
    season: Optional[int] = None
    last_update: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _retryable(func):
    """Decorator for retryable HTTP calls"""
    return retry(
        wait=wait_exponential(multiplier=1, min=2, max=20),
        stop=stop_after_attempt(4),
        retry=retry_if_exception_type(aiohttp.ClientError) | retry_if_exception_type(asyncio.TimeoutError),
        reraise=True,
    )(func)


class NFLOddsFetcher:
    """Fetches NFL odds for TNF integration"""
    
    PREFERRED_BOOKMAKERS = [
        "draftkings",
        "fanduel",
        "betmgm",
        "caesars",
        "betrivers",
        "pointsbet",
    ]
    
    NFL_TEAM_ABBR_MAP = {
        "Arizona Cardinals": "ARI",
        "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR",
        "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN",
        "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN",
        "Detroit Lions": "DET",
        "Green Bay Packers": "GB",
        "Houston Texans": "HOU",
        "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV",
        "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LA",
        "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN",
        "New England Patriots": "NE",
        "New Orleans Saints": "NO",
        "New York Giants": "NYG",
        "New York Jets": "NYJ",
        "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT",
        "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA",
        "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN",
        "Washington Commanders": "WAS",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or THE_ODDS_API_KEY
        if not self.api_key:
            logger.error("No Odds API key found - system will not work without it")
            logger.error("Please set THE_ODDS_API_KEY in your .env file")

        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self
    
    async def __aexit__(self, *exc):
        if self.session:
            await self.session.close()
    
    def _normalize_team_name(self, full_name: str) -> str:
        """Convert full team name to abbreviation"""
        return self.NFL_TEAM_ABBR_MAP.get(full_name, full_name)
    
    def _determine_week(self, commence_time: str) -> int:
        """Estimate NFL week from date (rough approximation)"""
        try:
            game_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
            # 2025 season starts ~Sept 5
            season_start = datetime(2025, 9, 5, tzinfo=game_date.tzinfo)
            if game_date < season_start:
                return 0
            
            days_since_start = (game_date - season_start).days
            return min(18, max(1, (days_since_start // 7) + 1))
        except Exception as e:
            logger.warning(f"Could not determine week: {e}")
            return 0
    
    @_retryable
    async def fetch_nfl_odds(self, target_date: Optional[date] = None) -> List[NFLGameOdds]:
        """Fetch NFL odds for given date (or today)"""
        target = target_date or date.today()

        # Check cache first
        cache_file = ODDS_CACHE_DIR / f"nfl_odds_{target}.json"
        if cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(minutes=15):
                logger.info(f"Using cached odds from {cache_file}")
                cached = json.loads(cache_file.read_text())
                return [NFLGameOdds(**o) for o in cached]

        if not self.api_key:
            logger.error("No API key found - cannot fetch odds")
            raise RuntimeError("THE_ODDS_API_KEY not found in environment. Please set it in .env file")

        # Fetch from real API only
        return await self._fetch_from_odds_api(target)
    
    async def _fetch_from_odds_api(self, target_date: date) -> List[NFLGameOdds]:
        """Fetch from The Odds API"""
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "bookmakers": ",".join(self.PREFERRED_BOOKMAKERS),
        }
        
        logger.info(f"Fetching NFL odds from The Odds API for {target_date}")
        
        async with self.session.get(url, params=params) as resp:
            if resp.status != 200:
                raise RuntimeError(f"API returned {resp.status}")
            
            raw = await resp.json()
        
        odds_list = []
        for game in raw:
            commence_dt = datetime.fromisoformat(game["commence_time"].replace("Z", "+00:00"))
            game_date = commence_dt.date()
            
            # Filter to target date
            if game_date != target_date:
                continue
            
            home_abbr = self._normalize_team_name(game["home_team"])
            away_abbr = self._normalize_team_name(game["away_team"])
            week = self._determine_week(game["commence_time"])
            
            for bookmaker in game.get("bookmakers", []):
                odds_obj = NFLGameOdds(
                    game_id=f"2025_{week:02d}_{away_abbr}_{home_abbr}",
                    commence_time=game["commence_time"],
                    home_team=home_abbr,
                    away_team=away_abbr,
                    bookmaker=bookmaker["title"],
                    source="the_odds_api",
                    week=week,
                    season=2025,
                    last_update=bookmaker.get("last_update", ""),
                )
                
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == game["home_team"]:
                                odds_obj.moneyline_home = outcome["price"]
                            elif outcome["name"] == game["away_team"]:
                                odds_obj.moneyline_away = outcome["price"]
                    
                    elif market["key"] == "spreads":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == game["home_team"]:
                                odds_obj.spread_home = outcome["point"]
                                odds_obj.spread_home_odds = outcome["price"]
                            elif outcome["name"] == game["away_team"]:
                                odds_obj.spread_away = outcome["point"]
                                odds_obj.spread_away_odds = outcome["price"]
                    
                    elif market["key"] == "totals":
                        for outcome in market["outcomes"]:
                            odds_obj.total = outcome["point"]
                            if outcome["name"] == "Over":
                                odds_obj.over_odds = outcome["price"]
                            elif outcome["name"] == "Under":
                                odds_obj.under_odds = outcome["price"]
                
                odds_list.append(odds_obj)
        
        # Cache results
        cache_file = ODDS_CACHE_DIR / f"nfl_odds_{target_date}.json"
        cache_file.write_text(json.dumps([o.to_dict() for o in odds_list], indent=2))
        logger.info(f"Cached {len(odds_list)} odds records to {cache_file}")

        return odds_list


async def fetch_and_integrate_nfl_odds(target_date: Optional[date] = None) -> Dict[str, Any]:
    """Main entry point - fetch NFL odds and save to conspiracy data dir"""
    target = target_date or date.today()
    
    async with NFLOddsFetcher() as fetcher:
        odds_list = await fetcher.fetch_nfl_odds(target)
        
        # Save to main data directory
        output_file = DATA_DIR / f"nfl_odds_{target}.json"
        output_file.write_text(json.dumps([o.to_dict() for o in odds_list], indent=2))
        
        logger.info(f"✅ Fetched {len(odds_list)} NFL odds records")
        logger.info(f"💾 Saved to: {output_file}")
        
        # Print summary
        if odds_list:
            logger.info("\n📊 GAMES FOUND:")
            for odds in odds_list[:5]:  # Show first 5
                logger.info(
                    f"  Week {odds.week}: {odds.away_team} @ {odds.home_team} "
                    f"| Total: {odds.total} | Spread: {odds.spread_home}"
                )
        
        return {
            "date": str(target),
            "games_count": len(odds_list),
            "file": str(output_file),
            "odds": odds_list,
        }


if __name__ == "__main__":
    import sys
    
    # Allow date as argument
    target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    
    result = asyncio.run(fetch_and_integrate_nfl_odds(target))
    
    print("\n" + "=" * 80)
    print(f"✅ NFL Odds Integration Complete")
    print(f"📅 Date: {result['date']}")
    print(f"🎮 Games: {result['games_count']}")
    print(f"💾 File: {result['file']}")
    print("=" * 80)
