#!/usr/bin/env python3
"""
Daily Odds Capture System
Runs daily to capture pre-game odds and save to historical database
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import existing odds fetcher (adapt to whichever exists)
try:
    from ultimate_odds_fetcher import OddsAPIFetcher
    USE_ULTIMATE = True
except ImportError:
    USE_ULTIMATE = False

# Fallback to basic requests
import requests


class DailyOddsCapture:
    """Captures daily odds and saves to historical database"""

    def __init__(self):
        self.api_key = os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY')
        self.db_file = Path('data/historical_odds.json')
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database if doesn't exist
        if not self.db_file.exists():
            self.db_file.write_text('{}')

    def load_database(self):
        """Load historical odds database"""
        with open(self.db_file) as f:
            return json.load(f)

    def save_database(self, db):
        """Save historical odds database"""
        with open(self.db_file, 'w') as f:
            json.dump(db, f, indent=2)

    def fetch_current_nfl_odds(self):
        """Fetch current NFL odds from The Odds API"""
        if not self.api_key:
            raise ValueError("No API key found. Set ODDS_API_KEY environment variable")

        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american'
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")

        return response.json()

    def parse_game_odds(self, game):
        """Parse odds from API response"""
        try:
            home_team = game['home_team']
            away_team = game['away_team']
            commence_time = game['commence_time'][:10]  # Date only

            # Create game key
            game_key = f"{commence_time}_{away_team}_at_{home_team}"

            # Get first bookmaker
            if not game.get('bookmakers'):
                return None

            bookmaker = game['bookmakers'][0]
            odds_data = {
                'spread': None,
                'total': None,
                'home_ml': None,
                'away_ml': None,
                'bookmaker': bookmaker['title'],
                'captured_at': datetime.now().isoformat(),
                'commence_time': game['commence_time']
            }

            # Parse markets
            for market in bookmaker.get('markets', []):
                if market['key'] == 'spreads':
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            odds_data['spread'] = outcome['point']

                elif market['key'] == 'totals':
                    odds_data['total'] = market['outcomes'][0]['point']

                elif market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            odds_data['home_ml'] = outcome['price']
                        elif outcome['name'] == away_team:
                            odds_data['away_ml'] = outcome['price']

            return game_key, odds_data

        except Exception as e:
            print(f"Error parsing game: {e}")
            return None

    def capture_daily_odds(self):
        """Main capture workflow"""
        print(f"📊 Daily Odds Capture - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        # Fetch current odds
        print("Fetching current NFL odds...")
        try:
            games = self.fetch_current_nfl_odds()
            print(f"✅ Found {len(games)} games with odds")
        except Exception as e:
            print(f"❌ Error fetching odds: {e}")
            return

        # Load database
        db = self.load_database()

        # Process each game
        added = 0
        updated = 0

        for game in games:
            result = self.parse_game_odds(game)
            if not result:
                continue

            game_key, odds_data = result

            if game_key in db:
                # Update existing entry (newer odds)
                db[game_key] = odds_data
                updated += 1
            else:
                # Add new entry
                db[game_key] = odds_data
                added += 1

        # Save database
        self.save_database(db)

        # Summary
        print("\n" + "="*80)
        print(f"✅ Capture Complete!")
        print(f"   New games: {added}")
        print(f"   Updated games: {updated}")
        print(f"   Total in database: {len(db)}")
        print(f"   Database: {self.db_file}")
        print("="*80)


def main():
    """CLI entry point"""
    try:
        capture = DailyOddsCapture()
        capture.capture_daily_odds()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
