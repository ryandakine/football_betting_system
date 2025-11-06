#!/usr/bin/env python3
"""
Football Game Data Fetcher - Real NFL & NCAAF game data
Fetches live game schedules, scores, and team statistics.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import aiohttp
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TeamStats:
    """Team statistics."""
    team_name: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    yards_offense: float = 0.0
    yards_defense: float = 0.0
    turnover_differential: int = 0


@dataclass
class LiveGameData:
    """Live game information and stats."""
    game_id: str
    home_team: str
    away_team: str
    home_score: int = 0
    away_score: int = 0
    quarter: str = "Not Started"
    time_remaining: str = ""
    possession: str = ""
    yard_line: str = ""
    down: int = 0
    yards_to_go: int = 0
    game_status: str = "scheduled"  # scheduled, in_progress, completed, postponed
    start_time: str = ""
    venue: str = ""
    weather: str = ""
    attendance: int = 0


@dataclass
class FootballGameData:
    """Complete football game data structure."""
    games: List[LiveGameData] = field(default_factory=list)
    team_stats: Dict[str, TeamStats] = field(default_factory=dict)
    fetch_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_summary(self) -> str:
        return f"Games: {len(self.games)}, Teams: {len(self.team_stats)}"


class ESPNGameDataClient:
    """ESPN API client for NFL/NCAFF game data."""

    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Football-Betting-System/1.0',
            'Accept': 'application/json'
        })

    def get_nfl_games(self, season: int = None) -> Dict[str, Any]:
        """Get NFL games for current or specified season."""
        if season is None:
            season = datetime.now().year

        url = f"{self.base_url}/nfl/scoreboard"
        params = {'season': season}

        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"ESPN NFL API error: {e}")
            return {}

    def get_ncaaf_games(self, season: int = None) -> Dict[str, Any]:
        """Get NCAAF games for current or specified season."""
        if season is None:
            season = datetime.now().year

        url = f"{self.base_url}/college-football/scoreboard"
        params = {'season': season}

        try:
            response = self.session.get(url, params=params, timeout=15)
            return response.json()
        except requests.RequestException as e:
            logger.error(f"ESPN NCAAF API error: {e}")
            return {}

    def get_team_stats(self, team_id: str, sport: str = "nfl") -> Dict[str, Any]:
        """Get detailed team statistics."""
        url = f"{self.base_url}/{sport}/teams/{team_id}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"ESPN team stats error: {e}")
            return {}


class NFLGameDataClient:
    """NFL official API client."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.nfl.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Football-Betting-System/1.0',
            'Accept': 'application/json'
        })
        if api_key:
            self.session.headers['Authorization'] = f'Bearer {api_key}'

    def get_current_week_games(self) -> Dict[str, Any]:
        """Get current week NFL games."""
        url = f"{self.base_url}/games"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"NFL API error: {e}")
            return {}

    def get_live_game_data(self, game_id: str) -> Dict[str, Any]:
        """Get live game data for specific game."""
        url = f"{self.base_url}/games/{game_id}/live"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"NFL live game error: {e}")
            return {}


class FootballGameDataFetcher:
    """Fetches comprehensive football game data from multiple sources."""

    def __init__(self, espn_client: ESPNGameDataClient = None, nfl_client: NFLGameDataClient = None):
        self.espn_client = espn_client or ESPNGameDataClient()
        self.nfl_client = nfl_client or NFLGameDataClient()
        logger.info("🏈 Football Game Data Fetcher initialized")

    def get_all_game_data(self, sport: str = "nfl") -> FootballGameData:
        """Fetch complete game data for specified sport."""
        game_data = FootballGameData()

        try:
            if sport.lower() == "nfl":
                self._fetch_nfl_data(game_data)
            elif sport.lower() == "ncaaf":
                self._fetch_ncaaf_data(game_data)
            else:
                logger.error(f"Unsupported sport: {sport}")
                return game_data

            # Fetch team statistics
            self._fetch_team_stats(game_data, sport)

            logger.info(f"✅ {sport.upper()} game data fetch complete: {game_data.get_summary()}")
            return game_data

        except Exception as e:
            logger.error(f"❌ Game data fetch failed: {e}")
            return game_data

    def _fetch_nfl_data(self, game_data: FootballGameData):
        """Fetch NFL game data."""
        # Try NFL official API first
        nfl_games = self.nfl_client.get_current_week_games()
        if nfl_games:
            self._process_nfl_games(nfl_games, game_data)
        else:
            # Fallback to ESPN
            logger.info("NFL API unavailable, using ESPN fallback")
            espn_data = self.espn_client.get_nfl_games()
            self._process_espn_games(espn_data, game_data, "nfl")

    def _fetch_ncaaf_data(self, game_data: FootballGameData):
        """Fetch NCAAF game data."""
        espn_data = self.espn_client.get_ncaaf_games()
        self._process_espn_games(espn_data, game_data, "college-football")

    def _process_nfl_games(self, nfl_data: Dict[str, Any], game_data: FootballGameData):
        """Process NFL API game data."""
        games = nfl_data.get('games', [])

        for game in games:
            live_game = LiveGameData(
                game_id=str(game.get('id', '')),
                home_team=game.get('home_team', {}).get('name', ''),
                away_team=game.get('away_team', {}).get('name', ''),
                home_score=game.get('home_score', 0),
                away_score=game.get('away_score', 0),
                quarter=game.get('quarter', 'Not Started'),
                time_remaining=game.get('time_remaining', ''),
                game_status=game.get('status', 'scheduled'),
                start_time=game.get('start_time', ''),
                venue=game.get('venue', ''),
                weather=game.get('weather', '')
            )
            game_data.games.append(live_game)

    def _process_espn_games(self, espn_data: Dict[str, Any], game_data: FootballGameData, sport: str):
        """Process ESPN API game data."""
        events = espn_data.get('events', [])

        for event in events:
            competitors = event.get('competitions', [{}])[0].get('competitors', [])

            if len(competitors) >= 2:
                home_team = competitors[0].get('team', {}).get('displayName', '')
                away_team = competitors[1].get('team', {}).get('displayName', '')

                # Get scores
                home_score = 0
                away_score = 0
                if 'score' in competitors[0]:
                    home_score = int(competitors[0]['score'])
                if 'score' in competitors[1]:
                    away_score = int(competitors[1]['score'])

                live_game = LiveGameData(
                    game_id=event.get('id', ''),
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    quarter=event.get('status', {}).get('period', 0),
                    time_remaining=event.get('status', {}).get('displayClock', ''),
                    game_status=event.get('status', {}).get('type', {}).get('state', 'scheduled'),
                    start_time=event.get('date', ''),
                    venue=event.get('competitions', [{}])[0].get('venue', {}).get('fullName', '')
                )
                game_data.games.append(live_game)

    def _fetch_team_stats(self, game_data: FootballGameData, sport: str):
        """Fetch team statistics."""
        # Collect unique teams from games
        teams = set()
        for game in game_data.games:
            teams.add(game.home_team)
            teams.add(game.away_team)

        # Fetch stats for each team (limit to avoid rate limits)
        for team_name in list(teams)[:20]:  # Limit to 20 teams to avoid API limits
            try:
                # For now, create placeholder stats - in production would fetch real data
                team_stats = TeamStats(team_name=team_name)
                game_data.team_stats[team_name] = team_stats
            except Exception as e:
                logger.warning(f"Failed to fetch stats for {team_name}: {e}")


async def main():
    """Test the football game data fetcher."""
    fetcher = FootballGameDataFetcher()

    print("Fetching NFL game data...")
    nfl_data = fetcher.get_all_game_data("nfl")
    print(f"NFL: {nfl_data.get_summary()}")

    print("Fetching NCAAF game data...")
    ncaaf_data = fetcher.get_all_game_data("ncaaf")
    print(f"NCAAF: {ncaaf_data.get_summary()}")


if __name__ == "__main__":
    asyncio.run(main())
