#!/usr/bin/env python3
"""
NCAA Data Collector for Warp AI Environment
Run this script in Warp AI or any environment with proper network access
"""

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("CFB_DATA_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"
OUTPUT_DIR = Path("data/football/historical/ncaaf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text)
    print("="*70)

def test_api_key():
    """Test if API key is valid"""
    print_header("🔑 TESTING API KEY")

    if not API_KEY:
        print("❌ ERROR: No API key found in .env file!")
        print("\nPlease add to .env:")
        print("CFB_DATA_API_KEY=your_api_key_here")
        return False

    print(f"API Key: {API_KEY[:15]}...{API_KEY[-10:]}")
    print("Testing authentication...")

    # Test with simple endpoint
    url = f"{BASE_URL}/teams/fbs"
    params = {"year": 2024}
    headers = {"Authorization": f"Bearer {API_KEY}"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)

        if response.status_code == 200:
            teams = response.json()
            print(f"✅ SUCCESS! API key is valid")
            print(f"   Retrieved {len(teams)} FBS teams")
            return True
        elif response.status_code == 401:
            print(f"❌ UNAUTHORIZED: Invalid API key")
            print(f"   Please verify your key at https://collegefootballdata.com")
            return False
        elif response.status_code == 403:
            print(f"❌ FORBIDDEN: Access denied")
            print(f"   Your account may need activation or email verification")
            return False
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ CONNECTION ERROR: {str(e)}")
        return False

def collect_season_games(year, week_start=1, week_end=15):
    """Collect game data for a season"""
    print_header(f"📊 COLLECTING {year} SEASON DATA")

    url = f"{BASE_URL}/games"
    headers = {"Authorization": f"Bearer {API_KEY}"}

    all_games = []

    for week in range(week_start, week_end + 1):
        try:
            params = {
                "year": year,
                "seasonType": "regular",
                "week": week
            }

            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 200:
                games = response.json()
                if games:
                    print(f"  ✅ Week {week:2d}: {len(games):3d} games")
                    all_games.extend(games)
                else:
                    print(f"  ⚠️  Week {week:2d}: No games")
            else:
                print(f"  ❌ Week {week:2d}: Error {response.status_code}")

        except Exception as e:
            print(f"  ❌ Week {week:2d}: {str(e)[:50]}")

    return all_games

def collect_team_stats(year):
    """Collect team statistics"""
    print_header(f"📈 COLLECTING {year} TEAM STATS")

    url = f"{BASE_URL}/stats/season"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"year": year}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)

        if response.status_code == 200:
            stats = response.json()
            print(f"  ✅ Retrieved stats for {len(stats)} teams")
            return stats
        else:
            print(f"  ❌ Error {response.status_code}")
            return []

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return []

def collect_sp_ratings(year):
    """Collect SP+ ratings (requires Silver/Gold tier)"""
    print_header(f"⭐ COLLECTING {year} SP+ RATINGS")

    url = f"{BASE_URL}/ratings/sp"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"year": year}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)

        if response.status_code == 200:
            ratings = response.json()
            print(f"  ✅ Retrieved SP+ ratings for {len(ratings)} teams")
            print(f"  🎉 You have SILVER/GOLD tier access!")
            return ratings
        elif response.status_code == 403:
            print(f"  ⚠️  SP+ data requires Silver/Gold tier ($25/month)")
            print(f"  💡 Your free tier data is still very valuable!")
            return []
        else:
            print(f"  ❌ Error {response.status_code}")
            return []

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return []

def save_data(year, games, stats, sp_ratings):
    """Save collected data to files"""
    print_header(f"💾 SAVING {year} DATA")

    # Save games
    games_file = OUTPUT_DIR / f"ncaaf_{year}_games.json"
    with open(games_file, 'w') as f:
        json.dump(games, f, indent=2)
    print(f"  ✅ Games saved: {games_file}")
    print(f"     {len(games)} games")

    # Save team stats
    if stats:
        stats_file = OUTPUT_DIR / f"ncaaf_{year}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✅ Stats saved: {stats_file}")
        print(f"     {len(stats)} teams")

    # Save SP+ ratings
    if sp_ratings:
        sp_file = OUTPUT_DIR / f"ncaaf_{year}_sp_ratings.json"
        with open(sp_file, 'w') as f:
            json.dump(sp_ratings, f, indent=2)
        print(f"  ✅ SP+ Ratings saved: {sp_file}")
        print(f"     {len(sp_ratings)} teams")

    # Create summary
    summary = {
        "year": year,
        "collected_at": datetime.now().isoformat(),
        "total_games": len(games),
        "total_team_stats": len(stats),
        "sp_ratings": len(sp_ratings),
        "has_advanced_metrics": len(sp_ratings) > 0
    }

    summary_file = OUTPUT_DIR / f"ncaaf_{year}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary saved: {summary_file}")

def main():
    """Main collection workflow"""
    print_header("🏈 NCAA FOOTBALL DATA COLLECTOR - WARP AI")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {OUTPUT_DIR.absolute()}")

    # Step 1: Test API key
    if not test_api_key():
        print("\n❌ API key validation failed. Exiting.")
        sys.exit(1)

    # Step 2: Ask which seasons to collect
    print("\n" + "="*70)
    print("Which seasons do you want to collect?")
    print("  Recommended: 2023, 2024 (most recent)")
    print("  Full history: 2015-2024 (takes longer)")
    print("="*70)

    seasons_input = input("\nEnter seasons (comma-separated, e.g., 2023,2024): ").strip()

    if not seasons_input:
        seasons = [2023, 2024]  # Default
        print(f"Using default: {seasons}")
    else:
        try:
            seasons = [int(s.strip()) for s in seasons_input.split(",")]
        except:
            print("Invalid input. Using default: 2023, 2024")
            seasons = [2023, 2024]

    # Step 3: Collect data for each season
    total_games = 0
    total_stats = 0
    total_sp = 0

    for year in seasons:
        # Collect games
        games = collect_season_games(year)
        total_games += len(games)

        # Collect stats
        stats = collect_team_stats(year)
        total_stats += len(stats)

        # Collect SP+ (if available)
        sp_ratings = collect_sp_ratings(year)
        total_sp += len(sp_ratings)

        # Save everything
        save_data(year, games, stats, sp_ratings)

    # Final summary
    print_header("✅ COLLECTION COMPLETE!")
    print(f"Total Seasons: {len(seasons)}")
    print(f"Total Games: {total_games}")
    print(f"Total Team Stats: {total_stats}")
    print(f"Total SP+ Ratings: {total_sp}")
    print(f"\nData saved to: {OUTPUT_DIR.absolute()}")

    if total_games > 0:
        print("\n🎉 SUCCESS! Next steps:")
        print("1. Run backtests: python demo_ncaa_backtest.py")
        print("2. Analyze games: python analyze_today_games.py")
        print("3. Run live system: python run_ncaaf_today.py")
    else:
        print("\n⚠️  No games collected. Please check:")
        print("- Your API key is valid")
        print("- You have network access")
        print("- The season/week parameters are correct")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
