#!/usr/bin/env python3
"""
Action Network Scraper - Fetches Handle Percentages

WHY THIS EXISTS:
To detect trap games, we need HANDLE DATA (where the money is going).
Action Network provides public betting % AND money % (handle).

KEY DISTINCTION:
- Public % = Number of bets (how many people)
- Money % = Dollar volume (HOW MUCH money)
- We want MONEY % (handle) to detect sharp vs public

USAGE:
    python action_network_scraper.py --week 11
    python action_network_scraper.py --game "BAL @ PIT"
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install requests beautifulsoup4")
    sys.exit(1)


class ActionNetworkScraper:
    """Scrapes handle data from Action Network"""

    def __init__(self):
        self.base_url = "https://www.actionnetwork.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def fetch_nfl_games(self, week: Optional[int] = None) -> List[Dict]:
        """
        Fetch NFL games with betting percentages.

        Args:
            week: NFL week number (None = current week)

        Returns:
            List of games with handle data
        """
        print(f"🔍 Fetching NFL games from Action Network...")

        # PRODUCTION MODE: NO MOCK DATA
        # This scraper must fetch REAL data or ERROR

        print("❌ ERROR: action_network_scraper.py fetch_nfl_games() not fully implemented")
        print()
        print("This scraper requires:")
        print("  1. Selenium implementation (see action_network_selenium_scraper.py)")
        print("  2. Or use paid Action Network API")
        print("  3. Or manual CSV input from Action Network Pro")
        print()
        print("🚨 CRITICAL: NO MOCK DATA in production!")
        print("   Do NOT use _get_sample_data() for real betting")
        print()
        raise NotImplementedError(
            "action_network_scraper.py must fetch REAL data. "
            "Use action_network_selenium_scraper.py or implement actual scraping. "
            "NO MOCK DATA ALLOWED in production betting."
        )

    # DELETED: _get_sample_data() - NO MOCK DATA IN PRODUCTION
    # If you need sample data for testing, use test files only

    def analyze_handle_divergence(self, game_data: Dict) -> Dict:
        """
        Analyze handle divergence for trap detection.

        Args:
            game_data: Game data with public % and money %

        Returns:
            Dict with trap analysis
        """
        spread_data = game_data.get('spread', {})
        public_pct = spread_data.get('public_percent', 50)
        money_pct = spread_data.get('money_percent', 50)

        # Calculate divergence
        divergence = money_pct - public_pct

        # Classify trap strength
        if abs(divergence) > 15:
            trap_strength = 5  # Extreme
        elif abs(divergence) > 10:
            trap_strength = 4  # Strong
        elif abs(divergence) > 7:
            trap_strength = 3  # Moderate
        elif abs(divergence) > 4:
            trap_strength = 2  # Slight
        else:
            trap_strength = 1  # None

        # Determine trap side
        if divergence > 0:
            # More money than bets = sharps on favorite
            trap_side = "FADE_UNDERDOG"
            message = f"Sharps loading favorite (money {money_pct}% vs bets {public_pct}%)"
        elif divergence < 0:
            # More bets than money = sharps on underdog
            trap_side = "FADE_FAVORITE"
            message = f"Sharps loading underdog (bets {public_pct}% vs money {money_pct}%)"
        else:
            trap_side = "NONE"
            message = "No significant divergence"

        return {
            'divergence': divergence,
            'trap_strength': trap_strength,
            'trap_side': trap_side,
            'message': message,
            'reverse_line_movement': game_data.get('line_movement', {}).get('reverse_line_movement', False)
        }

    def save_handle_data(self, games: List[Dict], week: int):
        """Save handle data to JSON file."""
        output_file = self.data_dir / f"action_network_week_{week}.json"

        output = {
            'week': week,
            'fetched_at': datetime.now().isoformat(),
            'games': games,
            'notes': [
                'public_percent = % of bets (number of tickets)',
                'money_percent = % of handle (dollar volume)',
                'Gap between them = sharp vs public divergence'
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"💾 Saved to {output_file}")
        return output_file

    def get_trap_games(self, games: List[Dict], min_strength: int = 3) -> List[Dict]:
        """
        Filter games with strong trap signals.

        Args:
            games: List of games with handle data
            min_strength: Minimum trap strength (1-5)

        Returns:
            List of trap games
        """
        trap_games = []

        for game in games:
            trap_analysis = self.analyze_handle_divergence(game)

            if trap_analysis['trap_strength'] >= min_strength:
                game['trap_analysis'] = trap_analysis
                trap_games.append(game)

        return trap_games


def main():
    parser = argparse.ArgumentParser(
        description="Scrape handle data from Action Network"
    )
    parser.add_argument(
        "--week",
        type=int,
        help="NFL week number (default: current week)"
    )
    parser.add_argument(
        "--game",
        help="Specific game to analyze (e.g., 'BAL @ PIT')"
    )
    parser.add_argument(
        "--trap-only",
        action="store_true",
        help="Show only games with strong trap signals (≥3)"
    )

    args = parser.parse_args()

    scraper = ActionNetworkScraper()

    # Fetch games
    week = args.week or 11  # Default to current week
    games = scraper.fetch_nfl_games(week)

    if args.game:
        # Filter to specific game
        games = [g for g in games if args.game.upper() in g['game'].upper()]
        if not games:
            print(f"❌ Game not found: {args.game}")
            return

    if args.trap_only:
        # Show only trap games
        games = scraper.get_trap_games(games, min_strength=3)
        print(f"\n🚨 Found {len(games)} trap games (strength ≥3)")

    # Display results
    print("\n" + "=" * 70)
    print("📊 HANDLE DATA (Sharp vs Public Money)")
    print("=" * 70)
    print()

    for game in games:
        print(f"🏈 {game['game']}")
        print(f"   Line: {game['spread']['favorite']} {game['spread']['line']}")
        print()

        # Spread analysis
        spread = game['spread']
        print(f"   SPREAD BETTING:")
        print(f"      Public: {spread['public_percent']}% of bets")
        print(f"      Money:  {spread['money_percent']}% of handle")
        print(f"      Gap:    {spread.get('bets_vs_money_gap', 0):+d}%")

        # Analyze divergence
        trap = scraper.analyze_handle_divergence(game)
        print(f"      Trap:   {'⭐' * trap['trap_strength']} ({trap['trap_strength']}/5)")
        print(f"      Signal: {trap['message']}")

        # Line movement
        movement = game.get('line_movement', {})
        if movement.get('reverse_line_movement'):
            print(f"      🚨 REVERSE LINE MOVEMENT DETECTED!")

        print()

    # Save data
    scraper.save_handle_data(games, week)

    print()
    print("💡 NEXT STEPS:")
    print("   1. Use this data with trap_detector.py")
    print("   2. Feed trap scores into DeepSeek analysis")
    print("   3. Fade public when trap strength ≥3")


if __name__ == "__main__":
    main()
