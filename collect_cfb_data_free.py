#!/usr/bin/env python3
"""
Collect NCAA Football data using public/free endpoints
Works without API key for basic data
"""

import requests
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_URL = "https://api.collegefootballdata.com"
OUTPUT_DIR = Path("data/football/historical/ncaaf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_request(endpoint, params=None, headers=None):
    """Make API request with error handling"""
    try:
        url = f"{BASE_URL}{endpoint}"
        response = requests.get(url, params=params, headers=headers, timeout=15)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"  ⚠️  {endpoint}: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"  ❌ {endpoint}: {str(e)[:50]}")
        return None

def collect_games(year, api_key=None):
    """Collect game results for a season"""
    print(f"\n📅 Collecting {year} season games...")

    headers = {}
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}

    all_games = []

    # Try to get games for each week
    for week in range(1, 16):
        params = {
            "year": year,
            "seasonType": "regular",
            "week": week
        }

        games = safe_request("/games", params=params, headers=headers)

        if games:
            print(f"  ✅ Week {week:2d}: {len(games)} games")
            all_games.extend(games)
            time.sleep(1)  # Rate limiting
        else:
            print(f"  ⚠️  Week {week:2d}: No data")
            time.sleep(1)

    if all_games:
        # Save raw JSON
        output_file = OUTPUT_DIR / f"ncaaf_{year}_raw.json"
        with open(output_file, 'w') as f:
            json.dump(all_games, f, indent=2)
        print(f"\n  💾 Saved {len(all_games)} games to {output_file}")

        # Convert to backtester format
        convert_to_backtester_format(all_games, year)

        return all_games
    else:
        print(f"\n  ❌ No games collected for {year}")
        return []

def convert_to_backtester_format(games, year):
    """Convert raw game data to backtester CSV format"""
    records = []

    for game in games:
        # Only include completed games with scores
        if game.get('completed') and game.get('home_points') is not None:
            home_score = game.get('home_points', 0)
            away_score = game.get('away_points', 0)

            record = {
                'game_id': game.get('id', ''),
                'home_team': game.get('home_team', ''),
                'away_team': game.get('away_team', ''),
                'home_score': home_score,
                'away_score': away_score,
                'edge_value': 0.05,  # Placeholder - will be calculated by model
                'confidence': 0.65,  # Placeholder - will be calculated by model
                'odds': -110,  # Placeholder - would need betting lines API
                'actual_result': 1 if home_score > away_score else 0,
                'conference': game.get('home_conference', 'Unknown'),
                'season': year,
                'week': game.get('week', 0)
            }
            records.append(record)

    if records:
        df = pd.DataFrame(records)
        csv_file = OUTPUT_DIR / f"ncaaf_{year}.csv"
        df.to_csv(csv_file, index=False)
        print(f"  💾 Converted to backtester format: {csv_file}")
        print(f"     {len(records)} completed games ready for backtesting")

def collect_team_stats(year, api_key=None):
    """Collect team statistics"""
    print(f"\n📊 Collecting {year} team stats...")

    headers = {}
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}

    params = {"year": year}
    stats = safe_request("/stats/season", params=params, headers=headers)

    if stats:
        output_file = OUTPUT_DIR / f"ncaaf_{year}_team_stats.json"
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✅ Saved {len(stats)} team stat records")
        return stats
    else:
        print(f"  ❌ No team stats available")
        return []

def collect_rankings(year, api_key=None):
    """Collect weekly rankings"""
    print(f"\n🏆 Collecting {year} rankings...")

    headers = {}
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}

    all_rankings = []

    for week in range(1, 16):
        params = {"year": year, "week": week, "seasonType": "regular"}
        rankings = safe_request("/rankings", params=params, headers=headers)

        if rankings:
            all_rankings.append({
                "year": year,
                "week": week,
                "rankings": rankings
            })
            print(f"  ✅ Week {week:2d}: Rankings collected")
            time.sleep(0.5)

    if all_rankings:
        output_file = OUTPUT_DIR / f"ncaaf_{year}_rankings.json"
        with open(output_file, 'w') as f:
            json.dump(all_rankings, f, indent=2)
        print(f"  ✅ Saved rankings data")

def main():
    """Main collection routine"""
    print("="*70)
    print("NCAA FOOTBALL DATA COLLECTOR")
    print("="*70)
    print("\nAttempting to collect data (trying with and without API key)...")

    # Load API key if available
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("CFB_DATA_API_KEY")

    if api_key:
        print(f"✅ API key found (will try authenticated requests)")
    else:
        print("⚠️  No API key found (will try public endpoints)")

    # Collect data for recent seasons (most likely to work)
    start_year = 2023
    end_year = 2024

    for year in range(start_year, end_year + 1):
        print(f"\n{'='*70}")
        print(f"SEASON {year}")
        print('='*70)

        # Collect games (most important)
        games = collect_games(year, api_key)

        # If games worked, try other data
        if games:
            collect_team_stats(year, api_key)
            collect_rankings(year, api_key)

        time.sleep(2)  # Be respectful with rate limiting

    print("\n" + "="*70)
    print("COLLECTION COMPLETE")
    print("="*70)
    print(f"\nData saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Check the CSV files in data/football/historical/ncaaf/")
    print("2. Run: python3 demo_ncaa_backtest.py")
    print("3. If you got data, your system is ready!")
    print("\nIf no data was collected:")
    print("- Sign up at https://collegefootballdata.com")
    print("- Generate a new API key from your account dashboard")
    print("- Add it to .env file: CFB_DATA_API_KEY=your_key_here")

if __name__ == "__main__":
    main()
