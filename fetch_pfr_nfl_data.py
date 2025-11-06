#!/usr/bin/env python3
"""
Fetch Real Historical NFL Game Data from Pro-Football-Reference
==============================================================

Retrieves historical NFL game results and scores from pro-football-reference.com
for seasons 2015-2024.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLHistoricalDataFetcher:
    """Fetch real historical NFL data from pro-football-reference.com."""

    def __init__(self):
        self.base_url = "https://www.pro-football-reference.com"
        self.output_dir = Path("data/football/historical/nfl")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_season_games(self, season: int) -> List[Dict[str, Any]]:
        """Fetch all games for a season from pro-football-reference."""
        games = []
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/years/{season}/games.htm"
                logger.info(f"Fetching season {season} from {url}")
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30), 
                                     headers={"User-Agent": "Mozilla/5.0"}) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        games = self._parse_pfr_games(html, season)
                        logger.info(f"Found {len(games)} games for season {season}")
                    else:
                        logger.error(f"Failed to fetch season {season}: {resp.status}")
        except Exception as e:
            logger.error(f"Error fetching season {season}: {e}")
        
        return games

    def _parse_pfr_games(self, html: str, season: int) -> List[Dict[str, Any]]:
        """Parse pro-football-reference HTML into game records."""
        games = []
        try:
            soup = BeautifulSoup(html, "html.parser")
            table = soup.find("table", {"id": "games"})
            if not table:
                return games
            
            for row in table.tbody.find_all("tr"):
                if row.get("class") and "thead" in row.get("class"):
                    continue
                
                game = self._parse_pfr_row(row, season)
                if game:
                    games.append(game)
        except Exception as e:
            logger.debug(f"Error parsing PFR HTML: {e}")
        
        return games

    def _parse_pfr_row(self, row: Any, season: int) -> Optional[Dict[str, Any]]:
        """Parse a single game row from pro-football-reference table."""
        try:
            cells = row.find_all("td")
            if len(cells) < 10:
                return None
            
            # PFR format: Day | Date | Time | Team1 | (@ or empty) | Team2 | boxscore | Score1 | Score2 | ...
            # Cells 0-2: day, date, time
            day = cells[0].text.strip()
            date_str = cells[1].text.strip()
            time_str = cells[2].text.strip()
            
            # Skip header/junk rows
            if not date_str or date_str == "Date":
                return None
            
            team1 = cells[3].text.strip()
            marker = cells[4].text.strip()  # Either @ or empty
            team2 = cells[5].text.strip()
            
            # If marker is @ then: team1 is away, team2 is home
            # If marker is empty then: team1 is home, team2 is away (or neutral)
            if marker == "@":
                away_team = team1
                home_team = team2
            else:
                # No @ means team1 is home
                home_team = team1
                away_team = team2
            
            # Skip if no valid teams
            if not home_team or not away_team or len(home_team) < 2 or len(away_team) < 2:
                return None
            
            # Scores are in cells 7 and 8
            score1_text = cells[7].text.strip() if len(cells) > 7 else ""
            score2_text = cells[8].text.strip() if len(cells) > 8 else ""
            
            # Parse scores
            try:
                # If @ marker, score1 is away, score2 is home
                # If no @, score1 is home, score2 is away
                if marker == "@":
                    away_score = int(score1_text) if score1_text.isdigit() else None
                    home_score = int(score2_text) if score2_text.isdigit() else None
                else:
                    home_score = int(score1_text) if score1_text.isdigit() else None
                    away_score = int(score2_text) if score2_text.isdigit() else None
                
                if away_score is None or home_score is None or away_score < 0 or home_score < 0:
                    return None
            except (ValueError, TypeError):
                return None
            
            game_id = f"{season}_{away_team.replace(' ', '_')}_{home_team.replace(' ', '_')}_{date_str.replace(' ', '_')}"
            
            # Determine result (1 = home team won)
            actual_result = 1 if home_score > away_score else 0
            
            return {
                "game_id": game_id,
                "season": season,
                "week": "",
                "date": date_str,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "spread": 0.0,
                "total": float(home_score + away_score),
                "odds": -110,
                "actual_result": actual_result,
                "edge_value": 0.02,
                "confidence": 0.55,
            }
        except Exception as e:
            logger.debug(f"Error parsing PFR row: {e}")
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
            
            # Be respectful to the server
            await asyncio.sleep(2)
        
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

    async def run(self, start_season: int = 2020, end_season: int = 2024) -> None:
        """Fetch and save all historical data."""
        logger.info(f"Fetching NFL historical data for seasons {start_season}-{end_season}")
        
        seasons_data = await self.fetch_multiple_seasons(start_season, end_season)
        
        for season, games in seasons_data.items():
            self.save_season_data(season, games)
        
        logger.info(f"✓ Complete! Data saved to {self.output_dir}")


async def main():
    fetcher = NFLHistoricalDataFetcher()
    await fetcher.run(start_season=2014, end_season=2024)


if __name__ == "__main__":
    asyncio.run(main())