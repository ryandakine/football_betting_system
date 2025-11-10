#!/usr/bin/env python3
"""
The Odds API Integration for NCAA Football
Fetches real-time spreads and odds for NCAA games
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

class NCAAOddsAPI:
    """Fetches NCAA football odds from The Odds API"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "americanfootball_ncaaf"

    def get_live_odds(self, bookmaker: str = "fanduel") -> List[Dict]:
        """
        Fetch current NCAA odds from The Odds API

        Args:
            bookmaker: Sportsbook to get odds from (fanduel, draftkings, etc.)

        Returns:
            List of games with odds
        """
        url = f"{self.base_url}/sports/{self.sport}/odds/"

        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'spreads,h2h',  # Get spreads and moneylines
            'oddsFormat': 'american',
            'bookmakers': bookmaker,
        }

        try:
            print(f"📡 Fetching NCAA odds from The Odds API...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            games = response.json()
            print(f"✅ Fetched {len(games)} games with live odds")

            # Process games
            processed = []
            for game in games:
                processed_game = self._process_game(game)
                if processed_game:
                    processed.append(processed_game)

            return processed

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching odds: {e}")
            return []

    def _process_game(self, game: Dict) -> Optional[Dict]:
        """Process a single game from API response"""
        try:
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            commence_time = game.get('commence_time')

            if not home_team or not away_team:
                return None

            # Extract spreads and odds
            bookmakers = game.get('bookmakers', [])
            if not bookmakers:
                return None

            bookmaker = bookmakers[0]  # Use first bookmaker
            markets = bookmaker.get('markets', [])

            spread_data = None
            h2h_data = None

            for market in markets:
                if market['key'] == 'spreads':
                    spread_data = market
                elif market['key'] == 'h2h':
                    h2h_data = market

            if not spread_data:
                return None

            # Parse spread outcomes
            outcomes = spread_data.get('outcomes', [])
            home_spread = None
            away_spread = None
            home_spread_odds = None
            away_spread_odds = None

            for outcome in outcomes:
                team = outcome.get('name')
                if team == home_team:
                    home_spread = outcome.get('point')
                    home_spread_odds = outcome.get('price')
                elif team == away_team:
                    away_spread = outcome.get('point')
                    away_spread_odds = outcome.get('price')

            # Parse moneyline if available
            home_ml = None
            away_ml = None
            if h2h_data:
                for outcome in h2h_data.get('outcomes', []):
                    team = outcome.get('name')
                    if team == home_team:
                        home_ml = outcome.get('price')
                    elif team == away_team:
                        away_ml = outcome.get('price')

            return {
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time,
                'home_spread': home_spread,
                'away_spread': away_spread,
                'home_spread_odds': home_spread_odds,
                'away_spread_odds': away_spread_odds,
                'home_moneyline': home_ml,
                'away_moneyline': away_ml,
                'bookmaker': bookmaker.get('title'),
            }

        except Exception as e:
            print(f"⚠️  Error processing game: {e}")
            return None

    def get_usage(self) -> Dict:
        """Check API usage"""
        url = f"{self.base_url}/sports/{self.sport}/odds/"

        params = {
            'apiKey': self.api_key,
            'regions': 'us',
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            # Get remaining requests from headers
            remaining = response.headers.get('x-requests-remaining')
            used = response.headers.get('x-requests-used')

            return {
                'remaining': remaining,
                'used': used,
                'status': 'success'
            }

        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

    def save_odds(self, odds: List[Dict], filepath: str = "data/ncaa_live_odds.json"):
        """Save odds to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            'timestamp': datetime.now().isoformat(),
            'games': odds
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved odds to {filepath}")


def test_odds_api(api_key: str):
    """Test the odds API integration"""
    print("🧪 Testing NCAA Odds API Integration")
    print("="*80)

    api = NCAAOddsAPI(api_key)

    # Check usage
    print("\n📊 Checking API usage...")
    usage = api.get_usage()
    if usage.get('status') == 'success':
        print(f"   Requests remaining: {usage.get('remaining')}")
        print(f"   Requests used: {usage.get('used')}")
    else:
        print(f"   ⚠️  Could not check usage: {usage.get('error')}")

    # Get live odds
    print("\n📡 Fetching live odds...")
    odds = api.get_live_odds()

    if odds:
        print(f"\n✅ Successfully fetched {len(odds)} games")
        print("\n📋 Sample games:")
        for i, game in enumerate(odds[:5], 1):
            print(f"\n{i}. {game['away_team']} @ {game['home_team']}")
            print(f"   Spread: {game['home_team']} {game['home_spread']:+.1f} ({game['home_spread_odds']:+d})")
            print(f"   Moneyline: {game['home_team']} {game['home_moneyline']:+d}, {game['away_team']} {game['away_moneyline']:+d}")

        # Save
        api.save_odds(odds)
    else:
        print("❌ No odds data retrieved")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Test with provided API key
    API_KEY = "e84d496405014d166f5dce95094ea024"
    test_odds_api(API_KEY)
