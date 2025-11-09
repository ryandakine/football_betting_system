#!/usr/bin/env python3
"""
Data Collector Agent - Fetches NCAA game data and SP+ ratings
"""

import asyncio
import requests
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DataCollectorAgent:
    """Agent responsible for collecting NCAA football data"""

    def __init__(self, config):
        self.config = config
        self.base_url = "https://api.collegefootballdata.com"

    async def collect_current_season(self):
        """Collect data for current season"""
        season = datetime.now().year if datetime.now().month >= 8 else datetime.now().year - 1

        logger.info(f"Collecting {season} season data...")

        try:
            # Collect games
            games = await self._fetch_games(season)

            # Collect SP+ ratings
            sp_ratings = await self._fetch_sp_ratings(season)

            # Save data
            self._save_games(season, games)
            self._save_sp_ratings(season, sp_ratings)

            return {
                'success': True,
                'season': season,
                'games_count': len(games),
                'sp_teams': len(sp_ratings)
            }

        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _fetch_games(self, year, max_week=15):
        """Fetch all games for a season"""
        headers = {"Authorization": f"Bearer {self.config.cfb_api_key}"}
        all_games = []

        for week in range(1, max_week + 1):
            try:
                url = f"{self.base_url}/games"
                params = {
                    "year": year,
                    "seasonType": "regular",
                    "week": week
                }

                response = requests.get(url, headers=headers, params=params, timeout=15)

                if response.status_code == 200:
                    games = response.json()
                    all_games.extend(games)
                    logger.debug(f"Week {week}: {len(games)} games")

                await asyncio.sleep(0.5)  # Rate limiting

            except Exception as e:
                logger.warning(f"Failed to fetch week {week}: {e}")

        return all_games

    async def _fetch_sp_ratings(self, year):
        """Fetch SP+ ratings"""
        headers = {"Authorization": f"Bearer {self.config.cfb_api_key}"}

        try:
            url = f"{self.base_url}/ratings/sp"
            params = {"year": year}

            response = requests.get(url, headers=headers, params=params, timeout=15)

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"SP+ fetch failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"SP+ fetch error: {e}")
            return []

    def _save_games(self, year, games):
        """Save games to file"""
        output_file = self.config.data_dir / f"ncaaf_{year}_games.json"
        with open(output_file, 'w') as f:
            json.dump(games, f, indent=2)
        logger.info(f"Saved {len(games)} games to {output_file}")

    def _save_sp_ratings(self, year, ratings):
        """Save SP+ ratings to file"""
        if ratings:
            output_file = self.config.data_dir / f"ncaaf_{year}_sp_ratings.json"
            with open(output_file, 'w') as f:
                json.dump(ratings, f, indent=2)
            logger.info(f"Saved {len(ratings)} SP+ ratings to {output_file}")

    async def get_upcoming_games(self, days_ahead=7):
        """Get upcoming games in next N days"""
        headers = {"Authorization": f"Bearer {self.config.cfb_api_key}"}

        try:
            # Get current week
            url = f"{self.base_url}/calendar"
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                calendar = response.json()
                # Find current week
                current_date = datetime.now()

                # Fetch games for current and next week
                all_upcoming = []
                season = current_date.year if current_date.month >= 8 else current_date.year - 1

                for week in range(1, 16):  # Check all weeks
                    games = await self._fetch_games_for_week(season, week)
                    # Filter for upcoming games
                    for game in games:
                        game_date = datetime.fromisoformat(game.get('start_date', '').replace('Z', '+00:00'))
                        days_until = (game_date - current_date).days
                        if 0 <= days_until <= days_ahead:
                            all_upcoming.append(game)

                return all_upcoming

        except Exception as e:
            logger.error(f"Failed to get upcoming games: {e}")
            return []

    async def _fetch_games_for_week(self, year, week):
        """Fetch games for a specific week"""
        headers = {"Authorization": f"Bearer {self.config.cfb_api_key}"}

        try:
            url = f"{self.base_url}/games"
            params = {
                "year": year,
                "seasonType": "regular",
                "week": week
            }

            response = requests.get(url, headers=headers, params=params, timeout=15)

            if response.status_code == 200:
                return response.json()

        except Exception as e:
            logger.warning(f"Failed to fetch week {week}: {e}")

        return []
