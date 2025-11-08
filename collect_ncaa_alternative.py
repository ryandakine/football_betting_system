#!/usr/bin/env python3
"""
Alternative NCAA data collection using ESPN and SR-CFB APIs
Works when College Football Data API is unavailable
"""

import requests
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("data/football/historical/ncaaf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def collect_from_espn(year):
    """
    Collect NCAA data from ESPN's public API
    More reliable and doesn't require authentication
    """
    print(f"\n📺 Collecting {year} season from ESPN API...")

    base_url = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"

    all_games = []

    for week in range(1, 16):
        try:
            # ESPN's scoreboard endpoint (public, no auth)
            url = f"{base_url}/scoreboard"
            params = {
                "dates": year,
                "week": week,
                "seasontype": 2,  # Regular season
                "limit": 500
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                events = data.get("events", [])

                if events:
                    print(f"  ✅ Week {week:2d}: {len(events)} games")

                    for event in events:
                        game = parse_espn_game(event, year, week)
                        if game:
                            all_games.append(game)
                else:
                    print(f"  ⚠️  Week {week:2d}: No games")

                time.sleep(1)  # Rate limiting
            else:
                print(f"  ❌ Week {week:2d}: API error {response.status_code}")

        except Exception as e:
            print(f"  ❌ Week {week:2d}: {str(e)[:50]}")

        time.sleep(0.5)

    if all_games:
        # Save to CSV for backtester
        df = pd.DataFrame(all_games)
        csv_file = OUTPUT_DIR / f"ncaaf_{year}.csv"
        df.to_csv(csv_file, index=False)

        # Save raw JSON too
        json_file = OUTPUT_DIR / f"ncaaf_{year}_espn.json"
        with open(json_file, 'w') as f:
            json.dump(all_games, f, indent=2)

        print(f"\n  ✅ SUCCESS! Collected {len(all_games)} games")
        print(f"  💾 Saved to: {csv_file}")

        return all_games
    else:
        print(f"\n  ❌ No games collected for {year}")
        return []

def parse_espn_game(event, year, week):
    """Parse ESPN event JSON into our format"""
    try:
        # Get competition data
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])

        if len(competitors) != 2:
            return None

        # Find home and away teams
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)

        if not home or not away:
            return None

        # Check if game is completed
        status = event.get("status", {}).get("type", {})
        if status.get("name") != "STATUS_FINAL":
            return None  # Only completed games

        # Extract scores
        home_score = int(home.get("score", 0) or 0)
        away_score = int(away.get("score", 0) or 0)

        # Get team info
        home_team = home.get("team", {}).get("displayName", "Unknown")
        away_team = away.get("team", {}).get("displayName", "Unknown")

        # Get conference
        home_conference = home.get("team", {}).get("conferenceId", "")
        conference_map = {
            "1": "ACC",
            "4": "Big Ten",
            "5": "Big 12",
            "8": "SEC",
            "9": "Pac-12",
            "12": "American",
            "15": "MAC",
            "17": "Mountain West",
            "37": "Sun Belt",
            "18": "C-USA",
        }
        conference = conference_map.get(str(home_conference), "Other")

        # Create record for backtester
        return {
            "game_id": event.get("id", ""),
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "edge_value": 0.05,  # Placeholder - model will calculate
            "confidence": 0.65,  # Placeholder - model will calculate
            "odds": -110,  # Placeholder - need betting lines API
            "actual_result": 1 if home_score > away_score else 0,
            "conference": conference,
            "season": year,
            "week": week,
            "date": event.get("date", ""),
            "venue": competition.get("venue", {}).get("fullName", ""),
        }

    except Exception as e:
        print(f"  ⚠️  Error parsing game: {str(e)[:50]}")
        return None

def main():
    """Main collection routine"""
    print("="*70)
    print("NCAA FOOTBALL DATA COLLECTOR - ESPN API")
    print("="*70)
    print("\n✅ Using ESPN's public API (no authentication required!)")
    print("⚠️  Note: This gets game results only (no advanced stats)")
    print("   For SP+/EPA, you'll need College Football Data API\n")

    # Collect most recent seasons
    seasons = [2023, 2024]

    total_games = 0

    for year in seasons:
        print(f"\n{'='*70}")
        print(f"SEASON {year}")
        print('='*70)

        games = collect_from_espn(year)
        total_games += len(games)

        time.sleep(2)  # Be respectful

    print("\n" + "="*70)
    print("COLLECTION COMPLETE!")
    print("="*70)
    print(f"\n📊 Total games collected: {total_games}")
    print(f"💾 Data saved to: {OUTPUT_DIR}\n")

    if total_games > 0:
        print("✅ SUCCESS! You can now run backtests:")
        print("   python3 demo_ncaa_backtest.py\n")
    else:
        print("❌ No data collected. Try:")
        print("   1. Check your internet connection")
        print("   2. Try again in a few minutes")
        print("   3. Get College Football Data API key from collegefootballdata.com\n")

if __name__ == "__main__":
    main()
