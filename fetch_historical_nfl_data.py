#!/usr/bin/env python3
"""
Fetch Real Historical NFL Game Data with Betting Lines
=========================================================

Retrieves historical NFL game results, scores, and betting lines from ESPN API
and other public sources for seasons 2015-2024.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLHistoricalDataFetcher:
    """Fetch real historical NFL data from public APIs."""

    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        self.output_dir = Path("data/football/historical/nfl")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_season_games(self, season: int) -> List[Dict[str, Any]]:
        """Fetch all games for a season from ESPN API."""
        games = []
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/seasons/{season}/types/2/events"
                logger.info(f"Fetching season {season} from {url}")
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        events = data.get("events", [])
                        logger.info(f"Found {len(events)} games for season {season}")
                        
                        for event in events:
                            game = self._parse_espn_game(event, season)
                            if game:
                                games.append(game)
                    else:
                        logger.error(f"Failed to fetch season {season}: {resp.status}")
        except Exception as e:
            logger.error(f"Error fetching season {season}: {e}")
        
        return games

    def _parse_espn_game(self, event: Dict[str, Any], season: int) -> Optional[Dict[str, Any]]:
        """Parse ESPN game event into backtest format."""
        try:
            # Extract basic game info
            game_id = event.get("id", "")
            name = event.get("name", "")
            date_str = event.get("date", "")
            status = event.get("status", {}).get("type", "")
            
            # Skip if not played
            if status not in ("STATUS_FINAL", "STATUS_IN_PROGRESS"):
                return None
            
            # Parse teams
            competitions = event.get("competitions", [])
            if not competitions:
                return None
            
            comp = competitions[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                return None
            
            home_team = competitors[0].get("team", {}).get("abbreviation", "")
            away_team = competitors[1].get("team", {}).get("abbreviation", "")
            
            # Get scores
            home_score = int(competitors[0].get("score", 0))
            away_score = int(competitors[1].get("score", 0))
            
            # Get betting lines (odds)
            odds_data = comp.get("odds", [])
            spread = 0.0
            total = 0.0
            spread_odds = -110
            
            if odds_data:
                latest_odds = odds_data[0]
                spreads = latest_odds.get("details", [])
                for spread_item in spreads:
                    if spread_item.get("name") == "spread":
                        spread = float(spread_item.get("displayValue", "0").split()[0])
                    elif spread_item.get("name") == "overunder":
                        total = float(spread_item.get("displayValue", "0").split()[0])
            
            # Determine result (1 = beat the spread, 0 = didn't)
            # Assuming spread is from home team perspective
            spread_differential = (home_score - away_score) - spread
            actual_result = 1 if spread_differential > 0 else 0
            
            return {
                "game_id": game_id,
                "season": season,
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "spread": spread,
                "total": total,
                "odds": spread_odds,
                "actual_result": actual_result,
                "edge_value": 0.02,  # Default conservative edge
                "confidence": 0.55,  # Default low confidence
            }
        except Exception as e:
            logger.debug(f"Error parsing game: {e}")
            return None

    async def fetch_multiple_seasons(self, start_season: int, end_season: int) -> Dict[int, List[Dict[str, Any]]]:
        """Fetch games for multiple seasons."""
        all_seasons = {}
        
        for season in range(start_season, end_season + 1):
            games = await self.fetch_season_games(season)
            if games:
                all_seasons[season] = games
                logger.info(f"✓ Season {season}: {len(games)} games")
            else:
                logger.warning(f"✗ Season {season}: No games found")
        
        return all_seasons

    def save_season_data(self, season: int, games: List[Dict[str, Any]]) -> Path:
        """Save season data as CSV for backtesting."""
        if not games:
            logger.warning(f"No data to save for season {season}")
            return None
        
        df = pd.DataFrame(games)
        output_file = self.output_dir / f"nfl_{season}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(games)} games to {output_file}")
        return output_file

    async def run(self, start_season: int = 2015, end_season: int = 2024) -> None:
        """Fetch and save all historical data."""
        logger.info(f"Fetching NFL historical data for seasons {start_season}-{end_season}")
        
        seasons_data = await self.fetch_multiple_seasons(start_season, end_season)
        
        for season, games in seasons_data.items():
            self.save_season_data(season, games)
        
        logger.info(f"✓ Complete! Data saved to {self.output_dir}")


async def main():
    fetcher = NFLHistoricalDataFetcher()
    await fetcher.run(start_season=2015, end_season=2024)


if __name__ == "__main__":
    asyncio.run(main())
