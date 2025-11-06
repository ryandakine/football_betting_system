#!/usr/bin/env python3
"""
Fetch complete NFL game data for the past 10 years (2015-2025).
Stores games in JSON for analysis and prediction.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ESPN API endpoints
ESPN_GAMES_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/events"
ESPN_SEASON_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/seasons/{season}/types/2/weeks/{week}/events"

DATA_DIR = Path("data/historical_games")


class HistoricalNFLGamesFetcher:
    """Fetch NFL games from ESPN API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_season_games(self, season: int) -> List[Dict[str, Any]]:
        """Fetch all games for a season from ESPN."""
        all_games = []
        
        # NFL seasons typically have 18 weeks
        for week in range(1, 19):
            try:
                url = ESPN_SEASON_URL.format(season=season, week=week)
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                events = data.get("events", [])
                
                for event in events:
                    game = self._parse_event(event)
                    if game:
                        all_games.append(game)
                
                logger.info(f"Season {season} Week {week}: {len(events)} games")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Failed to fetch season {season} week {week}: {e}")
                continue
        
        return all_games

    def _parse_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse ESPN event into game record."""
        try:
            competitors = event.get("competitions", [{}])[0].get("competitors", [])
            if len(competitors) < 2:
                return None
            
            home = competitors[0]
            away = competitors[1]
            
            game_date = event.get("date", "")
            game_id = event.get("id", "")
            
            # Extract scores
            home_score = int(home.get("score", 0) or 0)
            away_score = int(away.get("score", 0) or 0)
            
            # Skip games without final scores
            if home_score == 0 and away_score == 0:
                return None
            
            return {
                "game_id": game_id,
                "date": game_date,
                "home_team": home.get("team", {}).get("abbreviation", ""),
                "away_team": away.get("team", {}).get("abbreviation", ""),
                "home_score": home_score,
                "away_score": away_score,
                "total_score": home_score + away_score,
                "home_win": 1 if home_score > away_score else 0,
                "spread_result": home_score - away_score,
                "winner": home.get("team", {}).get("abbreviation", "") if home_score > away_score else away.get("team", {}).get("abbreviation", ""),
                "referee": event.get("competitions", [{}])[0].get("officials", [{}])[0].get("displayName", "Unknown"),
                "completed": event.get("status", {}).get("type", "") == "STATUS_FINAL",
            }
        except Exception as e:
            logger.error(f"Failed to parse event: {e}")
            return None

    def fetch_all_seasons(self, start_year: int = 2015, end_year: int = 2025) -> Dict[int, List[Dict[str, Any]]]:
        """Fetch games for all seasons."""
        all_seasons = {}
        
        for season in range(start_year, end_year + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fetching {season} season...")
            logger.info(f"{'='*60}")
            
            games = self.fetch_season_games(season)
            all_seasons[season] = games
            
            # Save season data immediately
            season_file = DATA_DIR / f"nfl_games_{season}.json"
            with open(season_file, "w") as f:
                json.dump(games, f, indent=2)
            logger.info(f"Saved {len(games)} games to {season_file}")
        
        return all_seasons

    def merge_all_seasons(self, start_year: int = 2015, end_year: int = 2025) -> Dict[str, Any]:
        """Merge all season files into one comprehensive file."""
        all_games = []
        stats = {}
        
        for season in range(start_year, end_year + 1):
            season_file = DATA_DIR / f"nfl_games_{season}.json"
            if season_file.exists():
                with open(season_file) as f:
                    games = json.load(f)
                    all_games.extend(games)
                    stats[season] = len(games)
        
        # Save merged file
        merged_file = DATA_DIR / "nfl_games_2015_2025.json"
        with open(merged_file, "w") as f:
            json.dump(all_games, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info("MERGED RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total games: {len(all_games)}")
        for season, count in sorted(stats.items()):
            logger.info(f"  {season}: {count} games")
        logger.info(f"Merged file: {merged_file}")
        
        return {"total_games": len(all_games), "by_season": stats}


if __name__ == "__main__":
    fetcher = HistoricalNFLGamesFetcher()
    
    logger.info("Fetching NFL games 2015-2025...")
    seasons = fetcher.fetch_all_seasons(start_year=2015, end_year=2025)
    
    logger.info("\nMerging season data...")
    result = fetcher.merge_all_seasons(start_year=2015, end_year=2025)
    
    logger.info(f"\n✅ Complete! Total {result['total_games']} games fetched.")
