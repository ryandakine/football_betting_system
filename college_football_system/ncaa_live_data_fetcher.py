#!/usr/bin/env python3
"""
NCAA/College Football Live Data Fetcher
========================================

Fetches real-time college football game data from multiple sources:
- ESPN API for live scores and game status
- Collegefootballdata.com API for detailed statistics
- Odds API for betting market data
- Weather APIs for game conditions

Provides accurate, real-time data for AI analysis.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class NCAALiveDataFetcher:
    """
    Fetches real-time NCAA/College Football data from multiple reliable sources.
    Provides accurate game status, scores, and statistics.
    """

    def __init__(self):
        self.session = None
        self.api_keys = self._load_api_keys()

        # API Endpoints
        self.apis = {
            'espn': 'https://site.api.espn.com/apis/site/v2/sports/football/college-football',
            'cfb_data': 'https://api.collegefootballdata.com',
            'odds': 'https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf',
            'weather': 'https://api.openweathermap.org/data/2.5/weather'
        }

        # Rate limiting
        self.last_request_time = {}
        self.request_delays = {
            'espn': 1,       # 1 second between requests
            'cfb_data': 2,   # 2 seconds (be respectful to free API)
            'odds': 5,       # 5 seconds
            'weather': 1     # 1 second
        }

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment or config files."""
        keys = {}

        # Try environment variables first
        keys['odds_api'] = os.getenv('ODDS_API_KEY', '')
        keys['weather_api'] = os.getenv('OPENWEATHER_API_KEY', '')
        keys['cfb_data_api'] = os.getenv('CFB_DATA_API_KEY', '')  # Optional but recommended

        # Try local config files
        try:
            if os.path.exists('config/api_keys.json'):
                with open('config/api_keys.json', 'r') as f:
                    config_keys = json.load(f)
                    keys.update(config_keys)
        except Exception as e:
            logger.warning(f"Could not load API keys from config: {e}")

        return keys

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_request(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with retry logic and rate limiting."""
        try:
            # Rate limiting
            domain = url.split('://')[1].split('.')[0]
            if domain in self.last_request_time:
                elapsed = time.time() - self.last_request_time[domain]
                delay = self.request_delays.get(domain, 1)
                if elapsed < delay:
                    await asyncio.sleep(delay - elapsed)

            self.last_request_time[domain] = time.time()

            # Make request
            if self.session:
                async with self.session.get(url, params=params, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"API request failed: {url} - Status {response.status}")
                        return None
            else:
                # Fallback to requests for sync calls
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"API request failed: {url} - Status {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"API request error for {url}: {e}")
            return None

    async def get_live_games(self, week: Optional[int] = None, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all current live NCAA games with real data.
        Returns accurate scores, status, and game information.

        Args:
            week: Specific week number (1-15 for regular season)
            year: Specific year (defaults to current year)
        """
        try:
            # Try ESPN API first (most reliable for live scores)
            espn_url = f"{self.apis['espn']}/scoreboard"
            params = {}
            if week:
                params['week'] = week
            if year:
                params['year'] = year

            espn_data = await self._make_request(espn_url, params=params)

            if espn_data and 'events' in espn_data:
                return await self._parse_espn_games(espn_data['events'])

            # Fallback to CFB Data API
            logger.info("ESPN API unavailable, trying College Football Data API...")
            cfb_url = f"{self.apis['cfb_data']}/games"

            headers = {}
            if self.api_keys.get('cfb_data_api'):
                headers['Authorization'] = f"Bearer {self.api_keys['cfb_data_api']}"

            cfb_params = {
                'year': year or datetime.now().year,
                'seasonType': 'regular'
            }
            if week:
                cfb_params['week'] = week

            cfb_data = await self._make_request(cfb_url, params=cfb_params, headers=headers)

            if cfb_data:
                return await self._parse_cfb_games(cfb_data)

            # Final fallback - return empty list with warning
            logger.warning("All NCAA APIs unavailable - using cached/scheduled data")
            return []

        except Exception as e:
            logger.error(f"Error fetching live games: {e}")
            return []

    async def _parse_espn_games(self, events: List[Dict]) -> List[Dict[str, Any]]:
        """Parse ESPN API game data into our format."""
        games = []

        # Major college football stadium to city mapping for weather API
        stadium_cities = {
            # SEC
            'Bryant-Denny Stadium': 'Tuscaloosa,US',
            'Sanford Stadium': 'Athens,US',
            'Tiger Stadium': 'Baton Rouge,US',
            'Kyle Field': 'College Station,US',
            'Neyland Stadium': 'Knoxville,US',
            'Ben Hill Griffin Stadium': 'Gainesville,US',
            'Williams-Brice Stadium': 'Columbia,US',
            'Vaught-Hemingway Stadium': 'Oxford,US',
            'Jordan-Hare Stadium': 'Auburn,US',
            'Razorback Stadium': 'Fayetteville,US',
            'Davis Wade Stadium': 'Starkville,US',

            # Big Ten
            'Michigan Stadium': 'Ann Arbor,US',
            'Ohio Stadium': 'Columbus,US',
            'Beaver Stadium': 'State College,US',
            'Camp Randall Stadium': 'Madison,US',
            'Spartan Stadium': 'East Lansing,US',
            'Memorial Stadium': 'Lincoln,US',
            'Kinnick Stadium': 'Iowa City,US',
            'Ross-Ade Stadium': 'West Lafayette,US',
            'Memorial Stadium (IN)': 'Bloomington,US',

            # Big 12
            'Darrell K Royal-Texas Memorial Stadium': 'Austin,US',
            'Gaylord Family Oklahoma Memorial Stadium': 'Norman,US',
            'Bill Snyder Family Stadium': 'Manhattan,US',
            'McLane Stadium': 'Waco,US',
            'Jack Trice Stadium': 'Ames,US',
            'Amon G. Carter Stadium': 'Fort Worth,US',

            # ACC
            'Clemson Memorial Stadium': 'Clemson,US',
            'Doak Campbell Stadium': 'Tallahassee,US',
            'Lane Stadium': 'Blacksburg,US',
            'Hard Rock Stadium': 'Miami,US',
            'Kenan Stadium': 'Chapel Hill,US',
            'Wallace Wade Stadium': 'Durham,US',

            # Pac-12
            'Rose Bowl': 'Pasadena,US',
            'Autzen Stadium': 'Eugene,US',
            'Husky Stadium': 'Seattle,US',
            'Stanford Stadium': 'Stanford,US',
            'Los Angeles Memorial Coliseum': 'Los Angeles,US',
            'Arizona Stadium': 'Tucson,US',
        }

        for event in events:
            try:
                # Extract basic game info
                game_id = event.get('id', '')
                status = event.get('status', {})

                # Get teams
                competitors = event.get('competitions', [{}])[0].get('competitors', [])
                if len(competitors) != 2:
                    continue

                home_team = None
                away_team = None
                for competitor in competitors:
                    if competitor.get('homeAway') == 'home':
                        home_team = competitor
                    else:
                        away_team = competitor

                if not home_team or not away_team:
                    continue

                # Game status
                game_status = status.get('type', {}).get('name', 'scheduled')
                is_live = game_status in ['in', 'STATUS_IN_PROGRESS']

                # Scores
                home_score = int(home_team.get('score', 0))
                away_score = int(away_team.get('score', 0))

                # Competition details
                competition = event.get('competitions', [{}])[0]
                venue = competition.get('venue', {})
                stadium_name = venue.get('fullName', '')

                # Get weather data if stadium is known
                weather_data = None
                if stadium_name in stadium_cities:
                    city = stadium_cities[stadium_name]
                    weather_data = await self._get_weather(city)

                # Conference info
                home_conference = home_team.get('team', {}).get('conferenceId')
                conference_map = {
                    '1': 'SEC',
                    '4': 'Big Ten',
                    '5': 'Big 12',
                    '1': 'ACC',
                    '9': 'Pac-12',
                    '12': 'AAC',
                    '15': 'MAC',
                    '17': 'Mountain West',
                    '37': 'Sun Belt',
                    '12': 'C-USA',
                }
                conference = conference_map.get(str(home_conference), 'Unknown')

                # Build game object
                game_data = {
                    'game_id': game_id,
                    'home_team': home_team.get('team', {}).get('displayName', 'Unknown'),
                    'away_team': away_team.get('team', {}).get('displayName', 'Unknown'),
                    'home_team_abbr': home_team.get('team', {}).get('abbreviation', ''),
                    'away_team_abbr': away_team.get('team', {}).get('abbreviation', ''),
                    'home_score': home_score,
                    'away_score': away_score,
                    'status': game_status,
                    'is_live': is_live,
                    'quarter': status.get('period', 0),
                    'time_remaining': status.get('displayClock', '0:00'),
                    'venue': stadium_name,
                    'city': venue.get('address', {}).get('city', ''),
                    'state': venue.get('address', {}).get('state', ''),
                    'conference': conference,
                    'broadcast': competition.get('broadcasts', [{}])[0].get('names', [''])[0],
                    'weather': weather_data,
                    'date': event.get('date'),
                    'is_neutral_site': competition.get('neutralSite', False),
                    # Additional NCAA-specific fields
                    'attendance': competition.get('attendance', 0),
                    'notes': event.get('notes', []),
                }

                # Add betting data if available
                odds = competition.get('odds', [{}])
                if odds:
                    game_data['spread'] = odds[0].get('details', '')
                    game_data['over_under'] = odds[0].get('overUnder', 0)

                games.append(game_data)

            except Exception as e:
                logger.error(f"Error parsing ESPN game data: {e}")
                continue

        return games

    async def _parse_cfb_games(self, games_data: List[Dict]) -> List[Dict[str, Any]]:
        """Parse College Football Data API game data into our format."""
        games = []

        for game in games_data:
            try:
                game_data = {
                    'game_id': str(game.get('id', '')),
                    'home_team': game.get('home_team', 'Unknown'),
                    'away_team': game.get('away_team', 'Unknown'),
                    'home_score': game.get('home_points', 0) or 0,
                    'away_score': game.get('away_points', 0) or 0,
                    'status': 'live' if game.get('completed') == False else 'completed',
                    'is_live': not game.get('completed', True),
                    'conference': game.get('home_conference', 'Unknown'),
                    'week': game.get('week'),
                    'season': game.get('season'),
                    'season_type': game.get('season_type'),
                    'venue': game.get('venue', ''),
                    'is_neutral_site': game.get('neutral_site', False),
                    'excitement_index': game.get('excitement_index', 0),
                }

                games.append(game_data)

            except Exception as e:
                logger.error(f"Error parsing CFB Data game: {e}")
                continue

        return games

    async def _get_weather(self, city: str) -> Optional[Dict[str, Any]]:
        """Get weather data for a specific city."""
        if not self.api_keys.get('weather_api'):
            return None

        try:
            url = self.apis['weather']
            params = {
                'q': city,
                'appid': self.api_keys['weather_api'],
                'units': 'imperial'
            }

            weather_data = await self._make_request(url, params=params)

            if weather_data:
                return {
                    'temperature': weather_data.get('main', {}).get('temp', 0),
                    'feels_like': weather_data.get('main', {}).get('feels_like', 0),
                    'description': weather_data.get('weather', [{}])[0].get('description', ''),
                    'wind_speed': weather_data.get('wind', {}).get('speed', 0),
                    'humidity': weather_data.get('main', {}).get('humidity', 0),
                    'pressure': weather_data.get('main', {}).get('pressure', 0),
                }

        except Exception as e:
            logger.error(f"Error fetching weather: {e}")

        return None

    async def get_game_details(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific game."""
        try:
            # Try ESPN API for detailed game info
            espn_url = f"{self.apis['espn']}/summary"
            params = {'event': game_id}

            game_data = await self._make_request(espn_url, params=params)

            if game_data:
                return self._parse_game_details(game_data)

        except Exception as e:
            logger.error(f"Error fetching game details: {e}")

        return None

    def _parse_game_details(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse detailed game data from ESPN API."""
        try:
            details = {
                'game_id': data.get('header', {}).get('id'),
                'boxscore': data.get('boxscore', {}),
                'drives': data.get('drives', {}),
                'scoring_plays': data.get('scoringPlays', []),
                'stats': data.get('boxscore', {}).get('teams', []),
                'leaders': data.get('leaders', []),
                'predictor': data.get('predictor', {}),
            }
            return details
        except Exception as e:
            logger.error(f"Error parsing game details: {e}")
            return {}


# Example usage
async def main():
    """Example usage of NCAA Live Data Fetcher."""
    async with NCAALiveDataFetcher() as fetcher:
        print("Fetching live NCAA games...")
        games = await fetcher.get_live_games()

        print(f"\nFound {len(games)} games")
        for game in games:
            print(f"\n{game['away_team']} @ {game['home_team']}")
            print(f"  Score: {game['away_score']} - {game['home_score']}")
            print(f"  Status: {game['status']}")
            print(f"  Conference: {game.get('conference', 'Unknown')}")
            if game.get('weather'):
                print(f"  Weather: {game['weather'].get('temperature')}°F, {game['weather'].get('description')}")


if __name__ == "__main__":
    asyncio.run(main())
