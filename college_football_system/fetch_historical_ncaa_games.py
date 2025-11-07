#!/usr/bin/env python3
"""
Fetch complete NCAA/College Football game data for the past 10 years (2015-2025).
Stores games in JSON/CSV for analysis and prediction.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ESPN API endpoints for college football
ESPN_CFB_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
ESPN_SEASON_URL = f"{ESPN_CFB_BASE}/seasons/{{season}}/types/2/weeks/{{week}}/events"

# Alternative: College Football Data API
CFB_DATA_API = "https://api.collegefootballdata.com/games"

DATA_DIR = Path("data/football/historical/ncaaf")


class HistoricalNCAAGamesFetcher:
    """Fetch NCAA/College Football games from multiple sources."""

    def __init__(self, cfb_api_key: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self.cfb_api_key = cfb_api_key
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_season_games_espn(self, season: int) -> List[Dict[str, Any]]:
        """Fetch all games for a season from ESPN API."""
        all_games = []

        # NCAA regular season typically has 15+ weeks
        for week in range(1, 16):
            try:
                url = ESPN_SEASON_URL.format(season=season, week=week)
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                data = response.json()
                events = data.get("events", [])

                for event in events:
                    game = self._parse_espn_event(event)
                    if game:
                        all_games.append(game)

                logger.info(f"Season {season} Week {week}: {len(events)} games")
                time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.error(f"Failed to fetch season {season} week {week}: {e}")
                continue

        return all_games

    def fetch_season_games_cfb_data(self, season: int) -> List[Dict[str, Any]]:
        """Fetch all games for a season from College Football Data API."""
        all_games = []

        try:
            headers = {}
            if self.cfb_api_key:
                headers['Authorization'] = f'Bearer {self.cfb_api_key}'

            params = {
                'year': season,
                'seasonType': 'regular',
            }

            response = self.session.get(
                CFB_DATA_API,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            for game in data:
                parsed_game = self._parse_cfb_data_event(game)
                if parsed_game:
                    all_games.append(parsed_game)

            logger.info(f"Season {season}: {len(all_games)} games from CFB Data API")
            time.sleep(2)  # Be respectful with rate limiting

        except Exception as e:
            logger.error(f"Failed to fetch season {season} from CFB Data API: {e}")

        return all_games

    def _parse_espn_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse ESPN event into game record."""
        try:
            competitors = event.get("competitions", [{}])[0].get("competitors", [])
            if len(competitors) < 2:
                return None

            # Find home and away teams
            home = None
            away = None
            for comp in competitors:
                if comp.get("homeAway") == "home":
                    home = comp
                else:
                    away = comp

            if not home or not away:
                return None

            game_date = event.get("date", "")
            game_id = event.get("id", "")
            status = event.get("status", {})

            # Extract scores
            home_score = int(home.get("score", 0) or 0)
            away_score = int(away.get("score", 0) or 0)

            # Skip games without final scores
            if status.get("type", {}).get("name") != "STATUS_FINAL":
                return None

            # Conference information
            home_team_data = home.get("team", {})
            away_team_data = away.get("team", {})

            # Get conference from team data
            home_conference = home_team_data.get("conference", {}).get("name", "Unknown")
            away_conference = away_team_data.get("conference", {}).get("name", "Unknown")

            # Venue information
            venue = event.get("competitions", [{}])[0].get("venue", {})

            # Broadcast info
            broadcasts = event.get("competitions", [{}])[0].get("broadcasts", [])
            broadcast = broadcasts[0].get("names", [""])[0] if broadcasts else "Unknown"

            return {
                "game_id": game_id,
                "date": game_date,
                "season": event.get("season", {}).get("year", 0),
                "week": event.get("week", {}).get("number", 0),
                "home_team": home_team_data.get("displayName", ""),
                "away_team": away_team_data.get("displayName", ""),
                "home_team_abbr": home_team_data.get("abbreviation", ""),
                "away_team_abbr": away_team_data.get("abbreviation", ""),
                "home_score": home_score,
                "away_score": away_score,
                "total_score": home_score + away_score,
                "home_win": 1 if home_score > away_score else 0,
                "actual_result": 1 if home_score > away_score else 0,  # For backtester
                "spread_result": home_score - away_score,
                "winner": home_team_data.get("abbreviation", "") if home_score > away_score else away_team_data.get("abbreviation", ""),
                "home_conference": home_conference,
                "away_conference": away_conference,
                "is_conference_game": home_conference == away_conference and home_conference != "Unknown",
                "venue": venue.get("fullName", ""),
                "city": venue.get("address", {}).get("city", ""),
                "state": venue.get("address", {}).get("state", ""),
                "neutral_site": event.get("competitions", [{}])[0].get("neutralSite", False),
                "attendance": event.get("competitions", [{}])[0].get("attendance", 0),
                "broadcast": broadcast,
                "completed": True,
                # Placeholder values for backtester (these would need real betting data)
                "edge_value": 0.05,  # Placeholder
                "confidence": 0.65,  # Placeholder
                "odds": -110,  # Placeholder
            }
        except Exception as e:
            logger.error(f"Failed to parse ESPN event: {e}")
            return None

    def _parse_cfb_data_event(self, game: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse College Football Data API event into game record."""
        try:
            home_points = game.get("home_points")
            away_points = game.get("away_points")

            # Skip games without scores
            if home_points is None or away_points is None:
                return None

            return {
                "game_id": str(game.get("id", "")),
                "date": game.get("start_date", ""),
                "season": game.get("season", 0),
                "week": game.get("week", 0),
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
                "home_score": int(home_points),
                "away_score": int(away_points),
                "total_score": int(home_points + away_points),
                "home_win": 1 if home_points > away_points else 0,
                "actual_result": 1 if home_points > away_points else 0,  # For backtester
                "spread_result": home_points - away_points,
                "winner": game.get("home_team", "") if home_points > away_points else game.get("away_team", ""),
                "home_conference": game.get("home_conference", "Unknown"),
                "away_conference": game.get("away_conference", "Unknown"),
                "is_conference_game": game.get("home_conference") == game.get("away_conference"),
                "venue": game.get("venue", ""),
                "neutral_site": game.get("neutral_site", False),
                "season_type": game.get("season_type", "regular"),
                "completed": game.get("completed", True),
                "excitement_index": game.get("excitement_index", 0.0),
                # Placeholder values for backtester
                "edge_value": 0.05,  # Placeholder
                "confidence": 0.65,  # Placeholder
                "odds": -110,  # Placeholder
            }
        except Exception as e:
            logger.error(f"Failed to parse CFB Data event: {e}")
            return None

    def fetch_all_seasons(
        self,
        start_year: int = 2015,
        end_year: int = 2025,
        use_cfb_data: bool = True
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Fetch games for all seasons."""
        all_seasons = {}

        for season in range(start_year, end_year + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fetching {season} season...")
            logger.info(f"{'='*60}")

            # Try CFB Data API first if available, then fall back to ESPN
            games = []
            if use_cfb_data and self.cfb_api_key:
                logger.info("Using College Football Data API...")
                games = self.fetch_season_games_cfb_data(season)

            if not games:
                logger.info("Using ESPN API...")
                games = self.fetch_season_games_espn(season)

            all_seasons[season] = games

            # Save season data immediately in both JSON and CSV formats
            self._save_season_data(season, games)

        return all_seasons

    def _save_season_data(self, season: int, games: List[Dict[str, Any]]):
        """Save season data in multiple formats."""
        # Save as JSON
        json_file = DATA_DIR / f"ncaaf_{season}.json"
        with open(json_file, "w") as f:
            json.dump(games, f, indent=2)
        logger.info(f"Saved {len(games)} games to {json_file}")

        # Save as CSV (for backtester compatibility)
        if games:
            df = pd.DataFrame(games)
            csv_file = DATA_DIR / f"ncaaf_{season}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved {len(games)} games to {csv_file}")

    def merge_all_seasons(
        self,
        start_year: int = 2015,
        end_year: int = 2025
    ) -> Dict[str, Any]:
        """Merge all season files into one comprehensive file."""
        all_games = []
        stats = {}

        for season in range(start_year, end_year + 1):
            json_file = DATA_DIR / f"ncaaf_{season}.json"
            if json_file.exists():
                with open(json_file) as f:
                    games = json.load(f)
                    all_games.extend(games)
                    stats[season] = len(games)

        # Save merged JSON file
        merged_json = DATA_DIR / f"ncaaf_{start_year}_{end_year}.json"
        with open(merged_json, "w") as f:
            json.dump(all_games, f, indent=2)

        # Save merged CSV file
        if all_games:
            df = pd.DataFrame(all_games)
            merged_csv = DATA_DIR / f"ncaaf_{start_year}_{end_year}.csv"
            df.to_csv(merged_csv, index=False)

        logger.info(f"\n{'='*60}")
        logger.info("MERGED RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total games: {len(all_games)}")
        for season, count in sorted(stats.items()):
            logger.info(f"  {season}: {count} games")
        logger.info(f"Merged JSON: {merged_json}")
        logger.info(f"Merged CSV: {merged_csv}")

        return {"total_games": len(all_games), "by_season": stats}


def main():
    """Main execution function."""
    import os

    # Check for CFB Data API key
    cfb_api_key = os.getenv("CFB_DATA_API_KEY")
    if cfb_api_key:
        logger.info("✓ CFB Data API key found")
    else:
        logger.warning("⚠ CFB Data API key not found, will use ESPN API only")

    fetcher = HistoricalNCAAGamesFetcher(cfb_api_key=cfb_api_key)

    logger.info("Fetching NCAA/College Football games 2015-2025...")
    seasons = fetcher.fetch_all_seasons(start_year=2015, end_year=2025)

    logger.info("\nMerging season data...")
    result = fetcher.merge_all_seasons(start_year=2015, end_year=2025)

    logger.info(f"\n✅ Complete! Total {result['total_games']} games fetched.")
    logger.info("\nData saved to:")
    logger.info(f"  - Individual seasons: {DATA_DIR}/ncaaf_YYYY.json/csv")
    logger.info(f"  - Merged data: {DATA_DIR}/ncaaf_2015_2025.json/csv")


if __name__ == "__main__":
    main()
