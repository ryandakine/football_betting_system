#!/usr/bin/env python3
"""
Line Shopper - Compares odds across multiple sportsbooks

WHY THIS EXISTS:
Getting GB -1 vs GB -1.5 or -105 vs -110 can add 3-5% to ROI.
This script fetches odds from all major sportsbooks and shows you
the best available line for each bet.

DESIGN PHILOSOPHY: Investment → System
- Agent doesn't know current odds
- This script fetches real-time odds from multiple books
- Shows you exactly where to place each bet for maximum value

OPERATIONAL COST:
- Initial: 2 hours to build
- Per bet: 0 seconds (automated)
- ROI impact: +3-5% from better lines

USAGE:
    python line_shopper.py --game "PHI @ GB" --bet-type spread
    python line_shopper.py --game "PHI @ GB" --bet-type total
    python line_shopper.py --game "PHI @ GB" --all
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("❌ Missing requests. Install with: pip install requests")
    sys.exit(1)


class LineShopper:
    """Compares odds across multiple sportsbooks"""

    # Major US sportsbooks (The Odds API keys)
    SPORTSBOOKS = [
        "draftkings",
        "fanduel",
        "betmgm",
        "caesars",
        "pointsbet",
        "betrivers",
        "wynnbet",
        "unibet_us",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            print("⚠️  No ODDS_API_KEY found. Set with:")
            print("   export ODDS_API_KEY='your_key_here'")
            print("   Or pass --api-key flag")

        self.base_url = "https://api.the-odds-api.com/v4"
        self.cache_dir = Path(__file__).parent / "data"
        self.cache_dir.mkdir(exist_ok=True)

    def fetch_odds(self, sport: str = "americanfootball_nfl") -> List[Dict]:
        """
        Fetch odds from The Odds API.

        Args:
            sport: Sport key (default: americanfootball_nfl)

        Returns:
            List of games with odds from all sportsbooks
        """
        if not self.api_key:
            return []

        try:
            url = f"{self.base_url}/sports/{sport}/odds/"
            params = {
                'apiKey': self.api_key,
                'regions': 'us',
                'markets': 'spreads,totals',
                'oddsFormat': 'american',
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            # Cache the response
            cache_file = self.cache_dir / f"odds_cache_{sport}.json"
            with open(cache_file, 'w') as f:
                json.dump(response.json(), f, indent=2)

            return response.json()

        except Exception as e:
            print(f"⚠️  Error fetching odds: {e}")
            return []

    def find_game(self, games: List[Dict], game_str: str) -> Optional[Dict]:
        """
        Find specific game from odds data.

        Args:
            games: List of games from API
            game_str: Game string like "PHI @ GB"

        Returns:
            Game dict or None
        """
        # Parse game string
        parts = game_str.upper().replace('@', ' ').split()
        if len(parts) < 2:
            return None

        away_abbr = parts[0]
        home_abbr = parts[1]

        # Search for matching game
        for game in games:
            away_team = game.get('away_team', '').upper()
            home_team = game.get('home_team', '').upper()

            # Check if abbreviations match
            if (away_abbr in away_team or away_team in away_abbr) and \
               (home_abbr in home_team or home_team in home_abbr):
                return game

        return None

    def compare_spreads(self, game: Dict, team: str) -> List[Dict]:
        """
        Compare spread odds across all sportsbooks.

        Args:
            game: Game dict from API
            team: Team to get spread for (e.g., "GB")

        Returns:
            List of {book, spread, odds} dicts sorted by best value
        """
        results = []

        for bookmaker in game.get('bookmakers', []):
            book_name = bookmaker.get('title', bookmaker.get('key'))

            for market in bookmaker.get('markets', []):
                if market.get('key') != 'spreads':
                    continue

                for outcome in market.get('outcomes', []):
                    outcome_team = outcome.get('name', '').upper()

                    # Check if this is the team we want
                    if team.upper() in outcome_team:
                        results.append({
                            'book': book_name,
                            'spread': outcome.get('point'),
                            'odds': outcome.get('price'),
                            'team': outcome.get('name')
                        })

        # Sort by best spread (most favorable for team)
        # For favorites (negative spread), more negative is worse
        # For underdogs (positive spread), more positive is better
        results.sort(key=lambda x: (-x['spread'], -x['odds']))

        return results

    def compare_totals(self, game: Dict, side: str) -> List[Dict]:
        """
        Compare total (over/under) odds across all sportsbooks.

        Args:
            game: Game dict from API
            side: "over" or "under"

        Returns:
            List of {book, total, odds} dicts sorted by best value
        """
        results = []

        for bookmaker in game.get('bookmakers', []):
            book_name = bookmaker.get('title', bookmaker.get('key'))

            for market in bookmaker.get('markets', []):
                if market.get('key') != 'totals':
                    continue

                for outcome in market.get('outcomes', []):
                    if outcome.get('name', '').lower() == side.lower():
                        results.append({
                            'book': book_name,
                            'total': outcome.get('point'),
                            'odds': outcome.get('price'),
                            'side': outcome.get('name')
                        })

        # Sort by best total + odds
        # For OVER: lower total is better (easier to hit)
        # For UNDER: higher total is better (easier to stay under)
        if side.lower() == 'over':
            results.sort(key=lambda x: (x['total'], -x['odds']))
        else:
            results.sort(key=lambda x: (-x['total'], -x['odds']))

        return results

    def calculate_ev_difference(self, odds1: int, odds2: int, prob: float) -> float:
        """
        Calculate EV difference between two odds at given win probability.

        Args:
            odds1: First odds (american format)
            odds2: Second odds (american format)
            prob: Win probability (0-1)

        Returns:
            EV difference in dollars per $100 wagered
        """
        def american_to_decimal(odds):
            if odds > 0:
                return (odds / 100) + 1
            else:
                return (100 / abs(odds)) + 1

        decimal1 = american_to_decimal(odds1)
        decimal2 = american_to_decimal(odds2)

        ev1 = (prob * (decimal1 - 1) * 100) - ((1 - prob) * 100)
        ev2 = (prob * (decimal2 - 1) * 100) - ((1 - prob) * 100)

        return ev2 - ev1

    def format_odds(self, odds: int) -> str:
        """Format odds with + for positive."""
        return f"+{odds}" if odds > 0 else str(odds)


def main():
    parser = argparse.ArgumentParser(
        description="Compare odds across sportsbooks"
    )
    parser.add_argument(
        "--game",
        required=True,
        help="Game to check (e.g. 'PHI @ GB')"
    )
    parser.add_argument(
        "--bet-type",
        choices=['spread', 'total', 'all'],
        default='all',
        help="Type of bet to compare"
    )
    parser.add_argument(
        "--team",
        help="Team for spread (e.g. 'GB')"
    )
    parser.add_argument(
        "--side",
        choices=['over', 'under'],
        help="Side for total"
    )
    parser.add_argument(
        "--api-key",
        help="The Odds API key (or set ODDS_API_KEY env var)"
    )

    args = parser.parse_args()

    shopper = LineShopper(api_key=args.api_key)

    # Fetch odds
    print("🔍 Fetching odds from sportsbooks...")
    games = shopper.fetch_odds()

    if not games:
        print("❌ Could not fetch odds. Check your API key.")
        sys.exit(1)

    # Find the game
    game = shopper.find_game(games, args.game)

    if not game:
        print(f"❌ Could not find game: {args.game}")
        print(f"   Available games:")
        for g in games[:5]:
            print(f"   - {g.get('away_team')} @ {g.get('home_team')}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"🏈 {game['away_team']} @ {game['home_team']}")
    print(f"{'=' * 70}\n")

    # Compare spreads
    if args.bet_type in ['spread', 'all']:
        if args.team:
            spreads = shopper.compare_spreads(game, args.team)

            if spreads:
                print(f"📊 SPREAD ODDS FOR {args.team.upper()}")
                print(f"{'-' * 70}")
                print(f"{'SPORTSBOOK':<20} {'SPREAD':<10} {'ODDS':<10} {'VALUE'}")
                print(f"{'-' * 70}")

                best_odds = spreads[0]
                for i, line in enumerate(spreads):
                    marker = "🏆 BEST" if i == 0 else ""
                    spread_str = f"{line['spread']:+.1f}".replace('+', ' +')
                    odds_str = shopper.format_odds(line['odds'])

                    # Calculate value vs worst line
                    if i > 0:
                        ev_diff = shopper.calculate_ev_difference(
                            spreads[-1]['odds'],
                            line['odds'],
                            0.70  # Assume 70% win prob
                        )
                        value = f"+${ev_diff:.2f}/100" if ev_diff > 0 else ""
                    else:
                        value = marker

                    print(f"{line['book']:<20} {spread_str:<10} {odds_str:<10} {value}")

                print(f"{'-' * 70}\n")

    # Compare totals
    if args.bet_type in ['total', 'all']:
        if args.side:
            totals = shopper.compare_totals(game, args.side)

            if totals:
                print(f"📊 TOTAL ODDS FOR {args.side.upper()}")
                print(f"{'-' * 70}")
                print(f"{'SPORTSBOOK':<20} {'TOTAL':<10} {'ODDS':<10} {'VALUE'}")
                print(f"{'-' * 70}")

                for i, line in enumerate(totals):
                    marker = "🏆 BEST" if i == 0 else ""
                    total_str = f"{line['total']:.1f}"
                    odds_str = shopper.format_odds(line['odds'])

                    # Calculate value vs worst line
                    if i > 0:
                        ev_diff = shopper.calculate_ev_difference(
                            totals[-1]['odds'],
                            line['odds'],
                            0.70  # Assume 70% win prob
                        )
                        value = f"+${ev_diff:.2f}/100" if ev_diff > 0 else ""
                    else:
                        value = marker

                    print(f"{line['book']:<20} {total_str:<10} {odds_str:<10} {value}")

                print(f"{'-' * 70}\n")

    # Show recommendation
    print("💡 RECOMMENDATION:")
    print("   1. Use the 🏆 BEST sportsbook for maximum value")
    print("   2. Better lines = higher ROI over time")
    print("   3. Even small odds differences compound to big profits\n")


if __name__ == "__main__":
    main()
