#!/usr/bin/env python3
"""
Analyze all college football games for today
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from narrative_conspiracy_engine import analyze_tonight_narratives

# The Odds API key
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
DATA_DIR = Path("data/referee_conspiracy")

async def fetch_todays_games():
    """Fetch today's college football games from The Odds API"""
    import aiohttp
    
    if not ODDS_API_KEY:
        raise ValueError("ODDS_API_KEY environment variable not set")
    
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Odds API returned {resp.status}: {error_text}")
            
            data = await resp.json()
            games = parse_odds_api_response(data)
            print(f"✅ Fetched {len(games)} games from The Odds API")
            return games


def parse_odds_api_response(data):
    """Parse The Odds API response into game objects"""
    games = []
    
    for event in data:
        if not event.get("bookmakers"):
            continue
        
        # Get first available bookmaker
        bookmaker = event["bookmakers"][0]
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        
        # Extract odds
        spread = None
        total = None
        
        for market in bookmaker.get("markets", []):
            if market["key"] == "spreads":
                spread = market["outcomes"][0].get("point", 0)
            elif market["key"] == "totals":
                total = market["outcomes"][0].get("point", 0)
        
        game_id = f"2025_CFB_{away_team.replace(' ', '_')}_{home_team.replace(' ', '_')}"
        
        games.append({
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "odds_data": {
                "spread": spread or 0,
                "total": total or 0,
            },
            "commence_time": event.get("commence_time", ""),
        })
    
    return games




async def analyze_all_games(games):
    """Analyze all games and track predictions"""
    results = []
    
    for i, game in enumerate(games, 1):
        print(f"\n{'=' * 80}")
        print(f"🏈 Game {i}/{len(games)}: {game['away_team']} @ {game['home_team']}")
        print(f"{'=' * 80}")
        
        try:
            result = await analyze_tonight_narratives(
                game_id=game["game_id"],
                home_team=game["home_team"],
                away_team=game["away_team"],
                odds_data=game["odds_data"],
            )
            
            results.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "recommendation": result["betting_recommendation"]["recommendation"],
                "confidence": result["betting_recommendation"]["confidence"],
                "conspiracy_prob": result["conspiracy_probability"],
                "signals": len(result["narrative_signals"]),
                "game_id": game["game_id"],
                "pick": result["betting_recommendation"].get("pick", "N/A"),
                "spread": game["odds_data"].get("spread", 0),
                "total": game["odds_data"].get("total", 0),
            })
            
            print(f"✅ {result['betting_recommendation']['recommendation']} (confidence: {result['betting_recommendation']['confidence']:.1%})")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "recommendation": "ERROR",
                "confidence": 0.0,
                "conspiracy_prob": 0.0,
                "signals": 0,
                "game_id": game["game_id"],
                "error": str(e),
            })
        
        # Rate limit between API calls
        await asyncio.sleep(2)
    
    return results


def print_summary(results):
    """Print summary of all predictions"""
    print("\n" + "=" * 80)
    print("📊 TODAY'S PREDICTIONS SUMMARY")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['game']}")
        print(f"   Line: {result.get('spread', 'N/A')} / O/U {result.get('total', 'N/A')}")
        print(f"   📌 PICK: {result.get('pick', 'N/A')}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Conspiracy Prob: {result['conspiracy_prob']:.1%}")
    
    # Save tracking file
    tracking_file = DATA_DIR / f"predictions_tracker_{datetime.now().strftime('%Y%m%d')}.json"
    tracking_file.write_text(json.dumps({
        "date": datetime.now().isoformat(),
        "total_games": len(results),
        "predictions": results,
    }, indent=2))
    
    print(f"\n💾 Saved predictions tracker to {tracking_file}")
    print("\n" + "=" * 80)
    print("🎯 High Confidence Picks:")
    print("=" * 80)
    
    high_conf = [r for r in results if r["confidence"] > 0.6 and r["recommendation"] != "NO_ACTION"]
    
    if high_conf:
        for pick in sorted(high_conf, key=lambda x: x["confidence"], reverse=True):
            print(f"✅ {pick['game']}")
            print(f"   📌 {pick.get('pick', 'N/A')}")
            print(f"   Confidence: {pick['confidence']:.1%}")
    else:
        print("   No high-confidence picks found")
    
    print("=" * 80)


def filter_primetime_games(games: list) -> list:
    """Filter for primetime/ranked team games"""
    # Top 25 teams and major programs
    ranked_teams = [
        "Georgia", "Ohio State", "Texas", "Penn State", "Notre Dame",
        "Oregon", "Miami", "Alabama", "Ole Miss", "Tennessee",
        "Indiana", "BYU", "Clemson", "Iowa State", "SMU",
        "Army", "Boise State", "LSU", "South Carolina", "Kansas State",
        "Colorado", "Washington State", "Louisville", "Missouri", "Memphis",
        "Michigan", "USC", "Florida", "Texas A&M", "Oklahoma"
    ]
    
    primetime_games = []
    for game in games:
        home = game["home_team"]
        away = game["away_team"]
        
        # Check if either team is ranked/major program
        if any(team in home or team in away for team in ranked_teams):
            primetime_games.append(game)
    
    return primetime_games


async def main():
    print("🔍 Fetching today's college football games...")
    games = await fetch_todays_games()
    
    if not games:
        print("❌ No games found")
        return
    
    # Filter to primetime/ranked games
    primetime = filter_primetime_games(games)
    
    if primetime:
        print(f"\n🌟 Found {len(primetime)} primetime/ranked games (out of {len(games)} total)")
        games = primetime
    else:
        print(f"⚠️  No ranked teams found - analyzing first 10 games")
        games = games[:10]
    
    print(f"\n📋 Analyzing {len(games)} games...\n")
    results = await analyze_all_games(games)
    
    # Filter results to only show high conspiracy probability
    high_conspiracy = [r for r in results if r.get("conspiracy_prob", 0) > 0.70]
    
    if high_conspiracy:
        print(f"\n⚠️  Showing only {len(high_conspiracy)} games with conspiracy probability >70%")
        print_summary(high_conspiracy)
    else:
        print(f"\n✅ No games with high conspiracy probability (>70%)")
        print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
