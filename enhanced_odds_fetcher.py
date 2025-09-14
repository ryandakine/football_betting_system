# enhanced_odds_fetcher.py
"""
Enhanced Odds Fetcher with Intelligent Caching and Multi-Market Support
Production-ready odds collection system with comprehensive error handling
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp
from tri_model_api_config import get_odds_api_config, get_trimodel_api_keys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GameInfo:
    """Basic game information"""

    id: str
    home_team: str
    away_team: str
    commence_time: str
    sport_key: str = "baseball_mlb"


@dataclass
class H2HBet:
    """Head-to-head betting option"""

    game_id: str
    team: str
    odds: float
    bookmaker: str
    last_update: str


@dataclass
class PlayerPropBet:
    """Player proposition bet"""

    game_id: str
    player_name: str
    prop_type: str  # 'home_runs', 'strikeouts', etc.
    over_under: str  # 'over' or 'under'
    line: float
    odds: float
    bookmaker: str
    last_update: str


@dataclass
class StructuredOdds:
    """Complete structured odds data"""

    games: list[GameInfo] = field(default_factory=list)
    h2h_bets: list[H2HBet] = field(default_factory=list)
    prop_bets: list[PlayerPropBet] = field(default_factory=list)
    fetch_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    api_requests_used: int = 0

    def get_summary(self) -> dict:
        """Get summary statistics"""
        return {
            "total_games": len(self.games),
            "total_h2h_bets": len(self.h2h_bets),
            "total_prop_bets": len(self.prop_bets),
            "unique_bookmakers": len(
                {bet.bookmaker for bet in self.h2h_bets + self.prop_bets}
            ),
            "fetch_timestamp": self.fetch_timestamp,
            "api_requests_used": self.api_requests_used,
        }


class IntelligentOddsCache:
    """Smart caching system for odds data"""

    def __init__(self, cache_duration_minutes: int = 15):
        self.cache = {}
        self.cache_duration = cache_duration_minutes * 60

    def _generate_cache_key(self, endpoint: str, params: dict) -> str:
        """Generate cache key from endpoint and parameters"""
        param_str = json.dumps(sorted(params.items()))
        return hashlib.md5(f"{endpoint}:{param_str}".encode()).hexdigest()

    def get(self, endpoint: str, params: dict) -> Any | None:
        """Get cached data if valid"""
        cache_key = self._generate_cache_key(endpoint, params)

        if cache_key not in self.cache:
            return None

        cached_data, timestamp = self.cache[cache_key]

        # Check if cache is still valid
        if time.time() - timestamp > self.cache_duration:
            del self.cache[cache_key]
            return None

        logger.debug(f"Cache hit for {endpoint}")
        return cached_data

    def set(self, endpoint: str, params: dict, data: Any):
        """Cache data with timestamp"""
        cache_key = self._generate_cache_key(endpoint, params)
        self.cache[cache_key] = (data, time.time())

        # Clean old cache entries
        self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp > self.cache_duration
        ]

        for key in expired_keys:
            del self.cache[key]


class EnhancedOddsFetcher:
    """
    Production-ready odds fetcher with intelligent caching,
    multi-market support, and comprehensive error handling
    """

    def __init__(
        self,
        api_key: str | None = None,
        concurrency_limit: int = 5,
        cache_enabled: bool = True,
    ):
        """Initialize the enhanced odds fetcher"""
        self.api_key = api_key or get_trimodel_api_keys().get("odds_api")
        if not self.api_key:
            raise ValueError("Odds API key is required")

        self.config = get_odds_api_config()
        self.base_url = self.config["base_url"]
        self.concurrency_limit = concurrency_limit
        self.cache = IntelligentOddsCache() if cache_enabled else None

        # Request tracking
        self.requests_made = 0
        self.cache_hits = 0
        self.errors_encountered = 0

        # Session will be created when entering context
        self.session: aiohttp.ClientSession | None = None

        logger.info(f"📊 Enhanced Odds Fetcher initialized")
        logger.info(f"   API: {self.base_url}")
        logger.info(f"   Concurrency limit: {concurrency_limit}")
        logger.info(f"   Caching: {'Enabled' if cache_enabled else 'Disabled'}")

    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=self.concurrency_limit)

        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"User-Agent": "Ultimate-TriModel-System/4.0"},
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _make_api_request(self, endpoint: str, params: dict) -> dict | None:
        """Make API request with caching and error handling"""

        # Check cache first
        if self.cache:
            cached_data = self.cache.get(endpoint, params)
            if cached_data:
                self.cache_hits += 1
                return cached_data

        # Prepare request
        request_params = {"apiKey": self.api_key, **params}

        url = f"{self.base_url}/{endpoint}"

        try:
            logger.debug(f"Making API request: {endpoint}")

            async with self.session.get(url, params=request_params) as response:
                self.requests_made += 1

                if response.status == 200:
                    data = await response.json()

                    # Cache successful response
                    if self.cache:
                        self.cache.set(endpoint, params, data)

                    return data

                elif response.status == 429:
                    logger.warning(f"Rate limited by Odds API. Waiting 60 seconds...")
                    await asyncio.sleep(60)
                    return await self._make_api_request(endpoint, params)  # Retry once

                else:
                    error_text = await response.text()
                    logger.error(
                        f"API request failed: {response.status} - {error_text}"
                    )
                    self.errors_encountered += 1
                    return None

        except TimeoutError:
            logger.error(f"Timeout making request to {endpoint}")
            self.errors_encountered += 1
            return None

        except Exception as e:
            logger.error(f"Error making API request to {endpoint}: {e}")
            self.errors_encountered += 1
            return None

    async def get_comprehensive_odds(
        self,
        sport_key: str = "baseball_mlb",
        markets: list[str] = None,
        regions: list[str] = None,
        use_cache: bool = True,
        filter_bookmakers: bool = True,
    ) -> StructuredOdds:
        """
        Get comprehensive odds data for all available markets

        Args:
            sport_key: Sport to fetch odds for
            markets: List of markets to fetch (defaults to h2h, totals)
            regions: List of regions (defaults to 'us')
            use_cache: Whether to use caching
            filter_bookmakers: Whether to filter to preferred bookmakers

        Returns:
            StructuredOdds object with all fetched data
        """

        logger.info(f"🔄 Fetching comprehensive odds for {sport_key}")

        # Set defaults
        if markets is None:
            markets = ["h2h", "totals"]
        if regions is None:
            regions = ["us"]

        # Temporarily disable cache if requested
        original_cache = self.cache
        if not use_cache:
            self.cache = None

        try:
            # Create result object
            structured_odds = StructuredOdds()

            # Fetch main odds data
            odds_data = await self._fetch_sport_odds(sport_key, markets, regions)

            if not odds_data:
                logger.warning("No odds data returned from API")
                return structured_odds

            # Process the odds data
            self._process_odds_data(odds_data, structured_odds, filter_bookmakers)

            # Try to fetch player props if available
            try:
                await self._fetch_player_props(sport_key, structured_odds)
            except Exception as e:
                logger.warning(f"Player props fetch failed (non-critical): {e}")

            # Update request count
            structured_odds.api_requests_used = self.requests_made

            # Log summary
            summary = structured_odds.get_summary()
            logger.info(f"✅ Odds fetch complete: {summary}")

            return structured_odds

        finally:
            # Restore original cache setting
            self.cache = original_cache

    async def _fetch_sport_odds(
        self, sport_key: str, markets: list[str], regions: list[str]
    ) -> list[dict] | None:
        """Fetch odds data for a sport"""

        params = {
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }

        endpoint = f"sports/{sport_key}/odds"
        return await self._make_api_request(endpoint, params)

    def _process_odds_data(
        self,
        odds_data: list[dict],
        structured_odds: StructuredOdds,
        filter_bookmakers: bool,
    ):
        """Process raw odds data into structured format"""

        preferred_bookmakers = set(self.config.get("preferred_bookmakers", []))

        for game_data in odds_data:
            try:
                # Create game info
                game = GameInfo(
                    id=game_data["id"],
                    home_team=game_data["home_team"],
                    away_team=game_data["away_team"],
                    commence_time=game_data["commence_time"],
                    sport_key=game_data.get("sport_key", "baseball_mlb"),
                )
                structured_odds.games.append(game)

                # Process bookmaker odds
                for bookmaker in game_data.get("bookmakers", []):
                    bookmaker_key = bookmaker["key"]

                    # Filter bookmakers if requested
                    if (
                        filter_bookmakers
                        and preferred_bookmakers
                        and bookmaker_key not in preferred_bookmakers
                    ):
                        continue

                    # Process markets
                    for market in bookmaker.get("markets", []):
                        market_key = market["key"]

                        if market_key == "h2h":
                            self._process_h2h_market(
                                game.id, market, bookmaker, structured_odds
                            )
                        elif market_key in ["totals", "spreads"]:
                            # Could add spread/totals processing here
                            pass

            except KeyError as e:
                logger.warning(f"Missing key in game data: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing game data: {e}")
                continue

    def _process_h2h_market(
        self,
        game_id: str,
        market: dict,
        bookmaker: dict,
        structured_odds: StructuredOdds,
    ):
        """Process head-to-head market data"""

        try:
            for outcome in market.get("outcomes", []):
                bet = H2HBet(
                    game_id=game_id,
                    team=outcome["name"],
                    odds=float(outcome["price"]),
                    bookmaker=bookmaker["key"],
                    last_update=market.get("last_update", ""),
                )
                structured_odds.h2h_bets.append(bet)

        except (KeyError, ValueError) as e:
            logger.warning(f"Error processing H2H market: {e}")

    async def _fetch_player_props(
        self, sport_key: str, structured_odds: StructuredOdds
    ):
        """Fetch player proposition bets (if available)"""

        # Player props markets for MLB
        prop_markets = [
            "batter_home_runs",
            "pitcher_strikeouts",
            "batter_hits",
            "batter_total_bases",
        ]

        for market in prop_markets:
            try:
                params = {
                    "regions": "us",
                    "markets": market,
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                }

                endpoint = f"sports/{sport_key}/odds"
                prop_data = await self._make_api_request(endpoint, params)

                if prop_data:
                    self._process_prop_data(prop_data, market, structured_odds)

            except Exception as e:
                logger.debug(f"Could not fetch {market} props: {e}")
                continue

    def _process_prop_data(
        self, prop_data: list[dict], market_type: str, structured_odds: StructuredOdds
    ):
        """Process player prop data"""

        for game_data in prop_data:
            game_id = game_data["id"]

            for bookmaker in game_data.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == market_type:

                        for outcome in market.get("outcomes", []):
                            try:
                                prop_bet = PlayerPropBet(
                                    game_id=game_id,
                                    player_name=outcome.get("description", "Unknown"),
                                    prop_type=market_type,
                                    over_under=outcome["name"].lower(),
                                    line=float(outcome.get("point", 0)),
                                    odds=float(outcome["price"]),
                                    bookmaker=bookmaker["key"],
                                    last_update=market.get("last_update", ""),
                                )
                                structured_odds.prop_bets.append(prop_bet)

                            except (KeyError, ValueError) as e:
                                logger.debug(f"Error processing prop bet: {e}")
                                continue

    def get_performance_stats(self) -> dict:
        """Get fetcher performance statistics"""

        cache_hit_rate = self.cache_hits / max(1, self.requests_made + self.cache_hits)
        error_rate = self.errors_encountered / max(1, self.requests_made)

        return {
            "requests_made": self.requests_made,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "errors_encountered": self.errors_encountered,
            "error_rate": error_rate,
            "cache_enabled": self.cache is not None,
        }


# Convenience function for backward compatibility
async def fetch_mlb_odds(api_key: str | None = None) -> StructuredOdds:
    """Simple function to fetch MLB odds"""

    async with EnhancedOddsFetcher(api_key=api_key) as fetcher:
        return await fetcher.get_comprehensive_odds(
            sport_key="baseball_mlb",
            markets=["h2h", "totals"],
            use_cache=True,
            filter_bookmakers=True,
        )


# Testing and debugging function
async def test_odds_fetcher():
    """Test the odds fetcher functionality"""

    print("🧪 Testing Enhanced Odds Fetcher...")

    try:
        async with EnhancedOddsFetcher() as fetcher:
            print("✅ Fetcher initialized successfully")

            # Test basic odds fetch
            odds = await fetcher.get_comprehensive_odds()

            summary = odds.get_summary()
            print(f"📊 Fetch Results:")
            print(f"   Games found: {summary['total_games']}")
            print(f"   H2H bets: {summary['total_h2h_bets']}")
            print(f"   Prop bets: {summary['total_prop_bets']}")
            print(f"   Bookmakers: {summary['unique_bookmakers']}")
            print(f"   API requests: {summary['api_requests_used']}")

            # Performance stats
            stats = fetcher.get_performance_stats()
            print(f"📈 Performance Stats:")
            print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
            print(f"   Error rate: {stats['error_rate']:.1%}")

            return odds

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_odds_fetcher())
