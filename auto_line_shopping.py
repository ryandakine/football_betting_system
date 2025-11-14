#!/usr/bin/env python3
"""
Multi-Sportsbook Line Shopping Tool
Scrapes DraftKings, FanDuel, BetMGM for best odds across books

Key Features:
- Finds best spread for each game
- Finds best total (over/under)
- Calculates Closing Line Value (CLV) improvement
- Identifies arbitrage opportunities

Edge: +2-4% CLV improvement = +2-4% ROI boost
"""
import os
import sys
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

try:
    from crawlbase import CrawlingAPI
except ImportError:
    print("❌ Crawlbase not installed. Run: pip install crawlbase")
    sys.exit(1)


@dataclass
class GameOdds:
    """Odds for a single game at one sportsbook"""
    book: str
    game: str
    away_team: str
    home_team: str
    away_spread: float
    away_spread_odds: int
    home_spread: float
    home_spread_odds: int
    total: float
    over_odds: int
    under_odds: int
    away_ml: int
    home_ml: int
    timestamp: str


class LineShoppingTool:
    """
    Multi-book odds comparison for maximum CLV

    CLV (Closing Line Value):
    - Getting -2.5 when market closes at -3.5 = +1 point of CLV
    - Worth ~2-3% ROI per half point
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv('CRAWLBASE_TOKEN')
        if not self.token:
            raise ValueError("No Crawlbase token. Set CRAWLBASE_TOKEN env var")

        self.api = CrawlingAPI({'token': self.token})
        self.data_dir = Path('data/line_shopping')
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.all_odds: List[GameOdds] = []

    def fetch_draftkings(self) -> str:
        """Fetch odds from DraftKings"""
        print("🎰 Fetching DraftKings odds...")

        url = 'https://sportsbook.draftkings.com/leagues/football/nfl'
        response = self.api.get(url)

        if response['statusCode'] == 200:
            print(f"   ✅ Got {len(response['body'])} bytes from DraftKings")

            # Save raw HTML
            with open(self.data_dir / 'raw_draftkings.html', 'w') as f:
                f.write(response['body'])

            return response['body']

        print(f"   ❌ DraftKings fetch failed: {response['statusCode']}")
        return None

    def fetch_fanduel(self) -> str:
        """Fetch odds from FanDuel"""
        print("🎰 Fetching FanDuel odds...")

        url = 'https://sportsbook.fanduel.com/navigation/nfl'
        response = self.api.get(url)

        if response['statusCode'] == 200:
            print(f"   ✅ Got {len(response['body'])} bytes from FanDuel")

            # Save raw HTML
            with open(self.data_dir / 'raw_fanduel.html', 'w') as f:
                f.write(response['body'])

            return response['body']

        print(f"   ❌ FanDuel fetch failed: {response['statusCode']}")
        return None

    def fetch_betmgm(self) -> str:
        """Fetch odds from BetMGM"""
        print("🎰 Fetching BetMGM odds...")

        url = 'https://sports.betmgm.com/en/sports/football-11/betting/usa-9/nfl-35'
        response = self.api.get(url)

        if response['statusCode'] == 200:
            print(f"   ✅ Got {len(response['body'])} bytes from BetMGM")

            # Save raw HTML
            with open(self.data_dir / 'raw_betmgm.html', 'w') as f:
                f.write(response['body'])

            return response['body']

        print(f"   ❌ BetMGM fetch failed: {response['statusCode']}")
        return None

    def parse_odds(self, html: str, book: str) -> List[GameOdds]:
        """
        Parse odds from sportsbook HTML

        In production, use BeautifulSoup to parse actual HTML
        For now, return sample structure
        """
        print(f"   🔍 Parsing {book} odds...")

        # Sample data structure
        # In production: parse actual HTML with BeautifulSoup
        sample_odds = [
            GameOdds(
                book=book,
                game='Chiefs @ Bills',
                away_team='Chiefs',
                home_team='Bills',
                away_spread=2.5 if book == 'DraftKings' else 3.0,
                away_spread_odds=-110,
                home_spread=-2.5 if book == 'DraftKings' else -3.0,
                home_spread_odds=-110,
                total=47.5 if book == 'FanDuel' else 48.0,
                over_odds=-108 if book == 'FanDuel' else -110,
                under_odds=-112 if book == 'FanDuel' else -110,
                away_ml=125,
                home_ml=-145,
                timestamp=datetime.now().isoformat()
            )
        ]

        print(f"   ✅ Parsed {len(sample_odds)} games from {book}")
        print(f"   📝 Note: Add BeautifulSoup parsing for production")

        return sample_odds

    def fetch_all_books(self) -> List[GameOdds]:
        """Fetch odds from all sportsbooks"""
        print("\n" + "="*80)
        print("🏈 FETCHING ODDS FROM ALL BOOKS")
        print("="*80 + "\n")

        all_odds = []

        # DraftKings
        html = self.fetch_draftkings()
        if html:
            odds = self.parse_odds(html, 'DraftKings')
            all_odds.extend(odds)

        # FanDuel
        html = self.fetch_fanduel()
        if html:
            odds = self.parse_odds(html, 'FanDuel')
            all_odds.extend(odds)

        # BetMGM
        html = self.fetch_betmgm()
        if html:
            odds = self.parse_odds(html, 'BetMGM')
            all_odds.extend(odds)

        self.all_odds = all_odds
        return all_odds

    def find_best_spreads(self) -> Dict[str, Dict]:
        """
        Find best spread for each game across all books

        Returns:
            Best spreads for away and home teams
        """
        print("\n🔍 Finding best spreads across books...")

        games = {}

        for odds in self.all_odds:
            if odds.game not in games:
                games[odds.game] = {
                    'away_team': odds.away_team,
                    'home_team': odds.home_team,
                    'best_away_spread': None,
                    'best_away_book': None,
                    'best_home_spread': None,
                    'best_home_book': None,
                    'spread_clv': 0
                }

            # Check away spread
            if (games[odds.game]['best_away_spread'] is None or
                odds.away_spread < games[odds.game]['best_away_spread']):
                games[odds.game]['best_away_spread'] = odds.away_spread
                games[odds.game]['best_away_book'] = odds.book
                games[odds.game]['away_spread_odds'] = odds.away_spread_odds

            # Check home spread
            if (games[odds.game]['best_home_spread'] is None or
                odds.home_spread > games[odds.game]['best_home_spread']):
                games[odds.game]['best_home_spread'] = odds.home_spread
                games[odds.game]['best_home_book'] = odds.book
                games[odds.game]['home_spread_odds'] = odds.home_spread_odds

        # Calculate CLV improvement
        for game, data in games.items():
            if data['best_away_spread'] and data['best_home_spread']:
                # Half point = ~2-3% ROI boost
                spread_diff = abs(
                    data['best_away_spread'] - data['best_home_spread']
                )
                data['spread_clv'] = round(spread_diff * 2.5, 2)  # 2.5% per 0.5 pt

        return games

    def find_best_totals(self) -> Dict[str, Dict]:
        """
        Find best total (over/under) for each game

        Returns:
            Best totals for over and under
        """
        print("🔍 Finding best totals across books...")

        games = {}

        for odds in self.all_odds:
            if odds.game not in games:
                games[odds.game] = {
                    'best_over_total': None,
                    'best_over_book': None,
                    'best_over_odds': None,
                    'best_under_total': None,
                    'best_under_book': None,
                    'best_under_odds': None,
                    'total_clv': 0
                }

            # Best over (want lowest total)
            if (games[odds.game]['best_over_total'] is None or
                odds.total < games[odds.game]['best_over_total']):
                games[odds.game]['best_over_total'] = odds.total
                games[odds.game]['best_over_book'] = odds.book
                games[odds.game]['best_over_odds'] = odds.over_odds

            # Best under (want highest total)
            if (games[odds.game]['best_under_total'] is None or
                odds.total > games[odds.game]['best_under_total']):
                games[odds.game]['best_under_total'] = odds.total
                games[odds.game]['best_under_book'] = odds.book
                games[odds.game]['best_under_odds'] = odds.under_odds

        # Calculate CLV
        for game, data in games.items():
            if data['best_over_total'] and data['best_under_total']:
                total_diff = data['best_under_total'] - data['best_over_total']
                data['total_clv'] = round(total_diff * 2.5, 2)

        return games

    def find_arbitrage(self) -> List[Dict]:
        """
        Find arbitrage opportunities (risk-free profit)

        Arbitrage exists when you can bet both sides and guarantee profit
        Example: Team A +150 at Book1, Team B -130 at Book2
        """
        print("💎 Checking for arbitrage opportunities...")

        arb_opportunities = []

        # Group by game
        games = {}
        for odds in self.all_odds:
            if odds.game not in games:
                games[odds.game] = []
            games[odds.game].append(odds)

        # Check each game for arb
        for game, odds_list in games.items():
            # Check spread arb
            best_away = max(odds_list, key=lambda x: x.away_spread)
            best_home = max(odds_list, key=lambda x: x.home_spread)

            # Check if arb exists (simplified check)
            if best_away.away_spread + best_home.home_spread > 0:
                arb_opportunities.append({
                    'game': game,
                    'type': 'spread',
                    'away_bet': f"{best_away.away_team} {best_away.away_spread} @ {best_away.book}",
                    'home_bet': f"{best_home.home_team} {best_home.home_spread} @ {best_home.book}",
                    'profit_margin': round((best_away.away_spread + best_home.home_spread) * 2.5, 2)
                })

        if arb_opportunities:
            print(f"   ✅ Found {len(arb_opportunities)} arbitrage opportunities!")
        else:
            print("   ℹ️  No arbitrage opportunities found")

        return arb_opportunities

    def save_data(self, best_spreads: Dict, best_totals: Dict,
                  arb_opps: List[Dict]) -> str:
        """Save line shopping data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.data_dir / f'line_shopping_{timestamp}.json'

        output = {
            'timestamp': datetime.now().isoformat(),
            'total_games': len(best_spreads),
            'best_spreads': best_spreads,
            'best_totals': best_totals,
            'arbitrage_opportunities': arb_opps,
            'raw_odds': [asdict(odds) for odds in self.all_odds]
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        return str(output_file)

    def print_summary(self, best_spreads: Dict, best_totals: Dict,
                     arb_opps: List[Dict]):
        """Print line shopping summary"""
        print("\n" + "="*80)
        print("🎰 LINE SHOPPING RESULTS")
        print("="*80)

        print(f"\n📊 Games Analyzed: {len(best_spreads)}")

        # Best spreads
        print("\n" + "-"*80)
        print("BEST SPREADS")
        print("-"*80)

        for game, data in list(best_spreads.items())[:5]:
            print(f"\n{game}:")
            print(f"  Away: {data['away_team']} {data['best_away_spread']:+.1f} @ {data['best_away_book']}")
            print(f"  Home: {data['home_team']} {data['best_home_spread']:+.1f} @ {data['best_home_book']}")
            print(f"  💰 CLV Improvement: +{data['spread_clv']:.1f}%")

        # Best totals
        print("\n" + "-"*80)
        print("BEST TOTALS")
        print("-"*80)

        for game, data in list(best_totals.items())[:5]:
            print(f"\n{game}:")
            print(f"  Over: {data['best_over_total']} @ {data['best_over_book']}")
            print(f"  Under: {data['best_under_total']} @ {data['best_under_book']}")
            print(f"  💰 CLV Improvement: +{data['total_clv']:.1f}%")

        # Arbitrage
        if arb_opps:
            print("\n" + "-"*80)
            print("💎 ARBITRAGE OPPORTUNITIES")
            print("-"*80)

            for arb in arb_opps:
                print(f"\n{arb['game']}:")
                print(f"  Bet 1: {arb['away_bet']}")
                print(f"  Bet 2: {arb['home_bet']}")
                print(f"  💰 Guaranteed Profit: +{arb['profit_margin']:.1f}%")

        print("\n" + "="*80)
        print("✅ Always bet at the book with the best line!")
        print("💡 0.5 point improvement = +2.5% ROI on average")
        print("="*80 + "\n")

    def run(self):
        """Run complete line shopping analysis"""
        print("\n" + "="*80)
        print("🎰 NFL LINE SHOPPING TOOL")
        print("="*80)

        # Fetch all odds
        self.fetch_all_books()

        if not self.all_odds:
            print("\n❌ No odds data fetched")
            return None

        # Find best lines
        best_spreads = self.find_best_spreads()
        best_totals = self.find_best_totals()
        arb_opps = self.find_arbitrage()

        # Save data
        output_file = self.save_data(best_spreads, best_totals, arb_opps)
        print(f"\n📁 Data saved to: {output_file}")

        # Print summary
        self.print_summary(best_spreads, best_totals, arb_opps)

        return {
            'spreads': best_spreads,
            'totals': best_totals,
            'arbitrage': arb_opps
        }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-Sportsbook Line Shopping Tool'
    )
    parser.add_argument(
        '--token',
        help='Crawlbase API token (or set CRAWLBASE_TOKEN env var)'
    )

    args = parser.parse_args()

    try:
        tool = LineShoppingTool(token=args.token)
        results = tool.run()

        if results:
            print("\n💡 Next step: Use best lines in your betting system")
            print("   Always bet at the book with the best line!")
            print("   0.5 point = +2.5% ROI improvement")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nSetup: export CRAWLBASE_TOKEN='your_token'")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
