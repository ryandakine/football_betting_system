#!/usr/bin/env python3
"""
NFL Live Tomorrow Plus - Fetch This Week's NFL Games
Fetches live NFL games from The Odds API and prepares for prediction
"""
import os
import sys
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path

def fetch_nfl_games_this_week():
    """Fetch NFL games from The Odds API for this week"""
    api_key = os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY')

    if not api_key:
        print("❌ No Odds API key found")
        print("   Set ODDS_API_KEY environment variable")
        return []

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'american'
    }

    try:
        print(f"🏈 Fetching NFL games from The Odds API...")
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 403:
            print("❌ API Key Error (403 Forbidden)")
            print("   Get new key from: https://the-odds-api.com/")
            return []

        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return []

        games = response.json()
        print(f"✅ Found {len(games)} NFL games this week")

        return games

    except requests.exceptions.Timeout:
        print("❌ API request timed out")
        return []
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        return []

def parse_game_data(game):
    """Parse game data into prediction format"""
    try:
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))

        # Get odds from first bookmaker
        bookmaker = game['bookmakers'][0] if game.get('bookmakers') else None

        spread_line = None
        spread_home_odds = None
        spread_away_odds = None
        total_line = None
        total_over_odds = None
        total_under_odds = None
        home_ml_odds = None
        away_ml_odds = None

        if bookmaker:
            for market in bookmaker.get('markets', []):
                if market['key'] == 'spreads':
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            spread_line = outcome['point']
                            spread_home_odds = outcome['price']
                        elif outcome['name'] == away_team:
                            spread_away_odds = outcome['price']

                elif market['key'] == 'totals':
                    total_line = market['outcomes'][0]['point']
                    for outcome in market['outcomes']:
                        if outcome['name'] == 'Over':
                            total_over_odds = outcome['price']
                        elif outcome['name'] == 'Under':
                            total_under_odds = outcome['price']

                elif market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            home_ml_odds = outcome['price']
                        elif outcome['name'] == away_team:
                            away_ml_odds = outcome['price']

        return {
            'game_id': f"{away_team}_at_{home_team}_{commence_time.strftime('%Y%m%d')}",
            'home_team': home_team,
            'away_team': away_team,
            'commence_time': game['commence_time'],
            'kickoff_time': commence_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'spread': spread_line,
            'spread_home_odds': spread_home_odds,
            'spread_away_odds': spread_away_odds,
            'total': total_line,
            'total_over_odds': total_over_odds,
            'total_under_odds': total_under_odds,
            'home_ml_odds': home_ml_odds,
            'away_ml_odds': away_ml_odds,
            'bookmaker': bookmaker['title'] if bookmaker else None
        }

    except Exception as e:
        print(f"⚠️  Error parsing game: {e}")
        return None

def save_games(games, output_file='data/nfl_live_games.json'):
    """Save games to JSON file"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(games, f, indent=2)

    print(f"💾 Saved {len(games)} games to {output_file}")

def display_games(games):
    """Display games in readable format"""
    if not games:
        print("\n⚠️  No games found")
        return

    print("\n" + "="*100)
    print(f"🏈 NFL GAMES THIS WEEK ({len(games)} games)")
    print("="*100)

    # Group by day
    games_by_day = {}
    for game in games:
        kickoff = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
        day_str = kickoff.strftime('%A, %B %d')

        if day_str not in games_by_day:
            games_by_day[day_str] = []
        games_by_day[day_str].append(game)

    for day, day_games in sorted(games_by_day.items()):
        print(f"\n📅 {day} ({len(day_games)} games)")
        print("-" * 100)

        for game in day_games:
            kickoff = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
            time_str = kickoff.strftime('%I:%M %p')

            spread_str = f"{game['spread']:+.1f}" if game['spread'] else "N/A"
            total_str = f"{game['total']:.1f}" if game['total'] else "N/A"

            print(f"  {time_str:<10} {game['away_team']:<30} @ {game['home_team']:<30}")
            print(f"             Spread: {spread_str:<8} Total: {total_str:<8} Book: {game['bookmaker']}")

def main():
    print("🏈 NFL LIVE GAMES FETCHER")
    print("="*100)

    # Fetch games
    raw_games = fetch_nfl_games_this_week()

    if not raw_games:
        print("\n❌ No games fetched. Check API key and try again.")
        sys.exit(1)

    # Parse games
    parsed_games = []
    for game in raw_games:
        parsed = parse_game_data(game)
        if parsed:
            parsed_games.append(parsed)

    # Display games
    display_games(parsed_games)

    # Save games
    output_file = 'data/nfl_live_games.json'
    save_games(parsed_games, output_file)

    print("\n✅ NFL games ready for prediction!")
    print(f"   Run: python run_nfl_12model_deepseek.py")
    print("="*100)

if __name__ == '__main__':
    main()
