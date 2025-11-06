#!/usr/bin/env python3
"""
Football Odds Fetcher - NFL & College Football
Adapted from MLB system for football-specific markets and data structures.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

import aiohttp
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GameInfo:
    """Football game information."""

    game_id: str
    home_team: str
    away_team: str
    commence_time: str
    sport_key: str
    sport_title: str
    bookmakers: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "commence_time": self.commence_time,
            "sport_key": self.sport_key,
            "sport_title": self.sport_title,
            "bookmakers": self.bookmakers,
        }


@dataclass
class H2HBet:
    """Head-to-head (moneyline) bet."""

    game_id: str
    home_team: str
    away_team: str
    home_odds: float
    away_odds: float
    bookmaker: str
    last_update: str

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_odds": self.home_odds,
            "away_odds": self.away_odds,
            "bookmaker": self.bookmaker,
            "last_update": self.last_update,
        }


@dataclass
class SpreadBet:
    """Point spread bet."""

    game_id: str
    home_team: str
    away_team: str
    home_spread: float
    away_spread: float
    home_odds: float
    away_odds: float
    bookmaker: str
    last_update: str

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_spread": self.home_spread,
            "away_spread": self.away_spread,
            "home_odds": self.home_odds,
            "away_odds": self.away_odds,
            "bookmaker": self.bookmaker,
            "last_update": self.last_update,
        }


@dataclass
class TotalBet:
    """Over/under total bet."""

    game_id: str
    home_team: str
    away_team: str
    total_points: float
    over_odds: float
    under_odds: float
    bookmaker: str
    last_update: str

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "total_points": self.total_points,
            "over_odds": self.over_odds,
            "under_odds": self.under_odds,
            "bookmaker": self.bookmaker,
            "last_update": self.last_update,
        }


@dataclass
class PlayerPropBet:
    """Player prop bet (passing yards, rushing yards, etc.)."""

    game_id: str
    player_name: str
    prop_type: str  # "passing_yards", "rushing_yards", "receiving_yards", "touchdowns"
    line_value: float
    over_odds: float
    under_odds: float
    bookmaker: str
    last_update: str

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "player_name": self.player_name,
            "prop_type": self.prop_type,
            "line_value": self.line_value,
            "over_odds": self.over_odds,
            "under_odds": self.under_odds,
            "bookmaker": self.bookmaker,
            "last_update": self.last_update,
        }


@dataclass
class StructuredOdds:
    """Structured odds data for football."""

    games: list[GameInfo] = field(default_factory=list)
    h2h_bets: list[H2HBet] = field(default_factory=list)
    spread_bets: list[SpreadBet] = field(default_factory=list)
    total_bets: list[TotalBet] = field(default_factory=list)
    prop_bets: list[PlayerPropBet] = field(default_factory=list)
    api_requests_used: int = 0
    fetch_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_summary(self) -> str:
        return (
            f"Games: {len(self.games)}, H2H: {len(self.h2h_bets)}, "
            f"Spreads: {len(self.spread_bets)}, Totals: {len(self.total_bets)}, "
            f"Props: {len(self.prop_bets)}"
        )


class FanDuelAPIClient:
    """Direct FanDuel API client for live odds data."""

    def __init__(self, api_key: str, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.fanduel.com"  # FanDuel API endpoint
        self.session = requests.Session()

        # Set headers for API authentication
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'Football-Betting-System/1.0'
        })

        logger.info("🎯 FanDuel API Client initialized")

    def get_live_odds(self, sport: str = "NFL") -> Dict[str, Any]:
        """Fetch live odds from FanDuel API."""
        try:
            # FanDuel API endpoints may vary - this is a template
            endpoint = f"/v1/odds/live/{sport.lower()}"
            url = f"{self.base_url}{endpoint}"

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            logger.info(f"📊 FanDuel live odds fetched for {sport}")
            return data

        except requests.RequestException as e:
            logger.error(f"❌ FanDuel API error: {e}")
            return {}

    def get_game_odds(self, game_id: str) -> Dict[str, Any]:
        """Get detailed odds for a specific game."""
        try:
            endpoint = f"/v1/odds/game/{game_id}"
            url = f"{self.base_url}{endpoint}"

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"❌ FanDuel game odds error: {e}")
            return {}

    def get_player_props(self, game_id: str) -> Dict[str, Any]:
        """Get player props for a specific game."""
        try:
            endpoint = f"/v1/props/game/{game_id}"
            url = f"{self.base_url}{endpoint}"

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"❌ FanDuel props error: {e}")
            return {}


class FootballOddsFetcher:
    """Fetches and processes football odds data."""

    def __init__(
        self,
        api_key: str,
        sport_key: str = "americanfootball_nfl",
        markets: list[str] | None = None,
        fanduel_api_key: str = None,
    ):
        self.api_key = api_key
        self.sport_key = sport_key
        self.markets = markets or ["h2h", "spreads", "totals", "player_props"]
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.session: aiohttp.ClientSession | None = None

        # Initialize FanDuel direct API if key provided
        self.fanduel_client = None
        if fanduel_api_key:
            self.fanduel_client = FanDuelAPIClient(fanduel_api_key)

        # Football-specific bookmaker preferences
        self.preferred_bookmakers = [
            "fanduel",  # Prioritize FanDuel
            "draftkings",
            "betmgm",
            "caesars",
            "betrivers",
            "pointsbet",
            "betus",
            "bovada",
        ]

        logger.info(f"🏈 Football Odds Fetcher initialized for {sport_key}")
        if self.fanduel_client:
            logger.info("🎯 FanDuel direct API integration active")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_all_odds_with_props(self) -> StructuredOdds:
        """Fetches all odds data including props."""
        structured_odds = StructuredOdds()

        try:
            # Try FanDuel direct API first if available
            if self.fanduel_client:
                fanduel_data = self._fetch_fanduel_odds()
                if fanduel_data:
                    self._process_fanduel_data(fanduel_data, structured_odds)
                    logger.info(f"✅ FanDuel direct API data processed")
                else:
                    logger.warning("FanDuel direct API failed, falling back to Odds API")

            # Fetch main odds data from Odds API as fallback/supplement
            odds_data = await self._fetch_odds()
            if not odds_data:
                logger.warning("No odds data returned from API")
                return structured_odds

            # Process the odds data
            self._process_odds_data(odds_data, structured_odds)

            # Try to fetch player props if available
            try:
                await self._fetch_player_props(structured_odds)
            except Exception as e:
                logger.warning(f"Player props fetch failed (non-critical): {e}")

            logger.info(f"✅ Football odds fetch complete: {structured_odds.get_summary()}")
            return structured_odds

        except Exception as e:
            logger.error(f"❌ Football odds fetch failed: {e}")
            return structured_odds

    def _fetch_fanduel_odds(self) -> Dict[str, Any]:
        """Fetch odds directly from FanDuel API."""
        if not self.fanduel_client:
            return {}

        try:
            sport_name = "NFL" if "nfl" in self.sport_key else "NCAAF"
            return self.fanduel_client.get_live_odds(sport_name)
        except Exception as e:
            logger.error(f"FanDuel direct fetch failed: {e}")
            return {}

    def _process_fanduel_data(self, fanduel_data: Dict[str, Any], structured_odds: StructuredOdds):
        """Process FanDuel API data into structured format."""
        try:
            # FanDuel data structure may vary - adapt as needed
            games = fanduel_data.get('games', [])

            for game in games:
                # Create game info from FanDuel data
                game_info = GameInfo(
                    game_id=str(game.get('id', '')),
                    home_team=game.get('home_team', ''),
                    away_team=game.get('away_team', ''),
                    commence_time=game.get('start_time', ''),
                    sport_key=self.sport_key,
                    sport_title=game.get('sport_title', 'Football'),
                    bookmakers=[{
                        'key': 'fanduel',
                        'title': 'FanDuel',
                        'last_update': game.get('last_update', ''),
                        'markets': []
                    }]
                )
                structured_odds.games.append(game_info)

                # Process FanDuel-specific odds
                self._process_fanduel_game_odds(game, game_info, structured_odds)

        except Exception as e:
            logger.error(f"Error processing FanDuel data: {e}")

    def _process_fanduel_game_odds(self, game: Dict[str, Any], game_info: GameInfo, structured_odds: StructuredOdds):
        """Process individual game odds from FanDuel."""
        game_id = str(game.get('id', ''))
        last_update = game.get('last_update', '')

        # Moneyline odds
        if 'moneyline' in game:
            ml = game['moneyline']
            h2h_bet = H2HBet(
                game_id=game_id,
                home_team=game_info.home_team,
                away_team=game_info.away_team,
                home_odds=float(ml.get('home', 0)),
                away_odds=float(ml.get('away', 0)),
                bookmaker='fanduel',
                last_update=last_update
            )
            structured_odds.h2h_bets.append(h2h_bet)

        # Spread odds
        if 'spread' in game:
            spread = game['spread']
            spread_bet = SpreadBet(
                game_id=game_id,
                home_team=game_info.home_team,
                away_team=game_info.away_team,
                home_spread=float(spread.get('home_spread', 0)),
                away_spread=float(spread.get('away_spread', 0)),
                home_odds=float(spread.get('home_odds', 0)),
                away_odds=float(spread.get('away_odds', 0)),
                bookmaker='fanduel',
                last_update=last_update
            )
            structured_odds.spread_bets.append(spread_bet)

        # Total odds
        if 'total' in game:
            total = game['total']
            total_bet = TotalBet(
                game_id=game_id,
                home_team=game_info.home_team,
                away_team=game_info.away_team,
                total_points=float(total.get('line', 0)),
                over_odds=float(total.get('over', 0)),
                under_odds=float(total.get('under', 0)),
                bookmaker='fanduel',
                last_update=last_update
            )
            structured_odds.total_bets.append(total_bet)

    async def _fetch_odds(self) -> list[dict] | None:
        """Fetches odds data from the API."""
        if self.session is None:
            raise RuntimeError(
                "HTTP session not initialized. "
                "Use 'async with FootballOddsFetcher(...) as fetcher'."
            )
        supported_markets = [m for m in self.markets if m in {"h2h", "spreads", "totals"}]
        params = {
            "regions": "us",
            "markets": ",".join(supported_markets),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "apiKey": self.api_key,
        }

        url = f"{self.base_url}/{self.sport_key}/odds"

        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                # Update request count
                remaining_requests = response.headers.get("x-requests-remaining", "unknown")
                logger.info(f"API requests remaining: {remaining_requests}")

                return data

        except aiohttp.ClientError as e:
            status = getattr(e, "status", "unknown")
            logger.error(f"API request failed (status={status}): {e.__class__.__name__}")
            return None

    def _process_odds_data(self, odds_data: list[dict], structured_odds: StructuredOdds):
        """Processes raw odds data into structured format."""
        for game_data in odds_data:
            # Create game info
            game_info = GameInfo(
                game_id=game_data.get("id", ""),
                home_team=game_data.get("home_team", ""),
                away_team=game_data.get("away_team", ""),
                commence_time=game_data.get("commence_time", ""),
                sport_key=game_data.get("sport_key", ""),
                sport_title=game_data.get("sport_title", ""),
                bookmakers=game_data.get("bookmakers", []),
            )
            structured_odds.games.append(game_info)

            # Process bookmaker data
            for bookmaker in game_data.get("bookmakers", []):
                if bookmaker.get("key") not in self.preferred_bookmakers:
                    continue

                self._process_bookmaker_markets(bookmaker, game_info, structured_odds)

                # Record odds snapshot for movement tracking (if GUI is available)
                try:
                    # Check if we have access to the GUI's odds tracker
                    if hasattr(self, 'gui_odds_tracker') and self.gui_odds_tracker:
                        for market in bookmaker.get("markets", []):
                            market_key = market.get("key", "")
                            if market_key in ["h2h", "spreads", "totals"]:
                                self.gui_odds_tracker.record_odds_snapshot(
                                    game_info.game_id,
                                    bookmaker.get("key", ""),
                                    market_key,
                                    market.get("outcomes", [])
                                )
                except Exception as e:
                    # Silently fail if odds tracking isn't available
                    pass

    def _process_bookmaker_markets(
        self, bookmaker: dict, game_info: GameInfo, structured_odds: StructuredOdds
    ):
        """Processes individual bookmaker markets."""
        bookmaker_key = bookmaker.get("key", "")
        last_update = bookmaker.get("last_update", "")

        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")

            if market_key == "h2h":
                self._process_h2h_market(
                    market, game_info, bookmaker_key, last_update, structured_odds
                )
            elif market_key == "spreads":
                self._process_spread_market(
                    market, game_info, bookmaker_key, last_update, structured_odds
                )
            elif market_key == "totals":
                self._process_total_market(
                    market, game_info, bookmaker_key, last_update, structured_odds
                )

    def _process_h2h_market(
        self,
        market: dict,
        game_info: GameInfo,
        bookmaker: str,
        last_update: str,
        structured_odds: StructuredOdds,
    ):
        """Processes head-to-head (moneyline) market."""
        home_odds = None
        away_odds = None

        for outcome in market.get("outcomes", []):
            if outcome.get("name") == game_info.home_team:
                home_odds = outcome.get("price", 0)
            elif outcome.get("name") == game_info.away_team:
                away_odds = outcome.get("price", 0)

        if home_odds is not None and away_odds is not None:
            h2h_bet = H2HBet(
                game_id=game_info.game_id,
                home_team=game_info.home_team,
                away_team=game_info.away_team,
                home_odds=home_odds,
                away_odds=away_odds,
                bookmaker=bookmaker,
                last_update=last_update,
            )
            structured_odds.h2h_bets.append(h2h_bet)

    def _process_spread_market(
        self,
        market: dict,
        game_info: GameInfo,
        bookmaker: str,
        last_update: str,
        structured_odds: StructuredOdds,
    ):
        """Processes point spread market."""
        home_spread = None
        away_spread = None
        home_odds = None
        away_odds = None

        for outcome in market.get("outcomes", []):
            if outcome.get("name") == game_info.home_team:
                home_spread = outcome.get("point", 0)
                home_odds = outcome.get("price", 0)
            elif outcome.get("name") == game_info.away_team:
                away_spread = outcome.get("point", 0)
                away_odds = outcome.get("price", 0)

        if home_spread is not None and away_spread is not None:
            # Coerce odds to float values even if missing
            home_odds = float(home_odds) if home_odds is not None else 0.0
            away_odds = float(away_odds) if away_odds is not None else 0.0
            spread_bet = SpreadBet(
                game_id=game_info.game_id,
                home_team=game_info.home_team,
                away_team=game_info.away_team,
                home_spread=float(home_spread),
                away_spread=float(away_spread),
                home_odds=home_odds,
                away_odds=away_odds,
                bookmaker=bookmaker,
                last_update=last_update,
            )
            structured_odds.spread_bets.append(spread_bet)

    def _process_total_market(
        self,
        market: dict,
        game_info: GameInfo,
        bookmaker: str,
        last_update: str,
        structured_odds: StructuredOdds,
    ):
        """Processes over/under total market."""
        total_points = None
        over_odds = None
        under_odds = None

        for outcome in market.get("outcomes", []):
            if outcome.get("name") == "Over":
                total_points = outcome.get("point", 0)
                over_odds = outcome.get("price", 0)
            elif outcome.get("name") == "Under":
                under_odds = outcome.get("price", 0)

        if total_points is not None and over_odds is not None and under_odds is not None:
            total_bet = TotalBet(
                game_id=game_info.game_id,
                home_team=game_info.home_team,
                away_team=game_info.away_team,
                total_points=total_points,
                over_odds=over_odds,
                under_odds=under_odds,
                bookmaker=bookmaker,
                last_update=last_update,
            )
            structured_odds.total_bets.append(total_bet)

    async def _fetch_player_props(self, structured_odds: StructuredOdds):
        """Fetches player props data."""
        # Note: Player props API endpoint may vary by provider
        # This is a placeholder for the actual implementation
        logger.info("📊 Player props fetching not yet implemented")
        pass


async def main():
    """Test the football odds fetcher."""
    from api_config import get_api_keys

    api_keys = get_api_keys()
    if not api_keys.get("odds_api"):
        logger.error("No odds API key found")
        return

    async with FootballOddsFetcher(
        api_key=api_keys["odds_api"], sport_key="americanfootball_nfl"
    ) as fetcher:
        odds = await fetcher.get_all_odds_with_props()
        logger.info(f"Fetched: {odds.get_summary()}")


if __name__ == "__main__":
    asyncio.run(main())
