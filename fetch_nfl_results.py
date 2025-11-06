#!/usr/bin/env python3
"""
Fetch NFL Game Results
=======================
Fetches actual NFL game scores and stores them for backtesting.
Uses ESPN API or scrapes ESPN scores.
"""

import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/referee_conspiracy")


async def fetch_espn_scores(target_date: date) -> List[Dict[str, Any]]:
    """Fetch scores from ESPN for a given date"""
    
    # ESPN scoreboard URL
    date_str = target_date.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={date_str}"
    
    logger.info(f"Fetching ESPN scores for {target_date}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.error(f"ESPN API returned {resp.status}")
                    return []
                
                data = await resp.json()
                games = data.get("events", [])
                
                results = []
                
                for game in games:
                    try:
                        competition = game.get("competitions", [{}])[0]
                        competitors = competition.get("competitors", [])
                        
                        if len(competitors) != 2:
                            continue
                        
                        home_team = next((c for c in competitors if c.get("homeAway") == "home"), None)
                        away_team = next((c for c in competitors if c.get("homeAway") == "away"), None)
                        
                        if not home_team or not away_team:
                            continue
                        
                        home_abbr = home_team.get("team", {}).get("abbreviation", "")
                        away_abbr = away_team.get("team", {}).get("abbreviation", "")
                        home_score = int(home_team.get("score", 0))
                        away_score = int(away_team.get("score", 0))
                        
                        # Game status
                        status = competition.get("status", {}).get("type", {}).get("name", "")
                        completed = status == "STATUS_FINAL"
                        
                        if not completed:
                            logger.debug(f"Skipping incomplete game: {away_abbr} @ {home_abbr}")
                            continue
                        
                        total_score = home_score + away_score
                        spread_result = home_score - away_score
                        
                        game_id = f"2025_*_{away_abbr}_{home_abbr}"  # Week unknown
                        
                        results.append({
                            "game_id": game_id,
                            "date": str(target_date),
                            "home_team": home_abbr,
                            "away_team": away_abbr,
                            "home_score": home_score,
                            "away_score": away_score,
                            "total_score": total_score,
                            "spread_result": spread_result,
                            "winner": home_abbr if home_score > away_score else away_abbr,
                            "completed": completed,
                        })
                        
                        logger.info(f"  ✅ {away_abbr} {away_score} @ {home_abbr} {home_score} (Total: {total_score})")
                    
                    except Exception as e:
                        logger.warning(f"Failed to parse game: {e}")
                        continue
                
                return results
    
    except Exception as e:
        logger.error(f"Failed to fetch ESPN scores: {e}")
        return []


async def save_game_results(target_date: date, results: List[Dict]) -> Path:
    """Save game results to file"""
    
    output_file = DATA_DIR / f"game_results_{target_date}.json"
    
    # Convert to dict keyed by game_id pattern for easy lookup
    results_dict = {}
    for result in results:
        # Create multiple possible game_id patterns
        away = result["away_team"]
        home = result["home_team"]
        
        # Try different week numbers (we don't know exact week)
        for week in range(1, 19):
            game_id = f"2025_{week:02d}_{away}_{home}"
            results_dict[game_id] = result
    
    output_file.write_text(json.dumps(results_dict, indent=2))
    logger.info(f"💾 Saved {len(results)} game results to {output_file}")
    
    return output_file


async def fetch_week_results(start_date: date, end_date: date):
    """Fetch results for a week range"""
    
    all_results = []
    current_date = start_date
    
    while current_date <= end_date:
        results = await fetch_espn_scores(current_date)
        
        if results:
            await save_game_results(current_date, results)
            all_results.extend(results)
        
        current_date += timedelta(days=1)
        await asyncio.sleep(1)  # Rate limit
    
    print(f"\n{'='*80}")
    print(f"📊 RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Dates: {start_date} to {end_date}")
    print(f"Total Games: {len(all_results)}")
    print(f"{'='*80}\n")
    
    return all_results


async def main():
    """Main runner"""
    
    import sys
    
    if len(sys.argv) >= 3:
        start_date = date.fromisoformat(sys.argv[1])
        end_date = date.fromisoformat(sys.argv[2])
    else:
        # Default to last week
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
    
    print(f"\n🏈 Fetching NFL Results")
    print(f"📅 Date Range: {start_date} to {end_date}\n")
    
    results = await fetch_week_results(start_date, end_date)
    
    print(f"✅ Fetched {len(results)} completed games")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
