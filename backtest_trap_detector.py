#!/usr/bin/env python3
"""
Trap Detector Backtest - Measure Historical Performance

WHY THIS EXISTS:
To validate that trap scores actually predict winners.
Measures ROI of betting based on trap signals.

USAGE:
    python backtest_trap_detector.py --season 2024
    python backtest_trap_detector.py --all-seasons
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from trap_detector import TrapDetector


class TrapBacktester:
    """Backtests trap detection strategy on historical games"""

    def __init__(self):
        self.detector = TrapDetector()
        self.data_dir = Path(__file__).parent / "data"
        self.results_dir = Path(__file__).parent / "backtest_results"
        self.results_dir.mkdir(exist_ok=True)

    def load_historical_games(self, season: int) -> List[Dict]:
        """
        Load historical game data with handle percentages.

        In production, this would load from:
        - Action Network historical data
        - Your own collected data
        - Third-party data providers

        For now, shows expected data structure.
        """
        # Sample historical games with trap data
        return [
            {
                'game': 'BAL @ PIT',
                'week': 10,
                'season': 2024,
                'home_ml': -150,
                'away_ml': +130,
                'home_handle': 0.85,  # 85% of money on PIT
                'away_handle': 0.15,
                'opening_home_ml': -130,
                'opening_away_ml': +110,
                'actual_result': 'away_win',  # BAL won
                'home_score': 17,
                'away_score': 20,
                'spread_line': -3.5,
                'spread_result': 'away_cover'
            },
            {
                'game': 'KC @ BUF',
                'week': 11,
                'season': 2024,
                'home_ml': -130,
                'away_ml': +110,
                'home_handle': 0.75,  # 75% on BUF but...
                'away_handle': 0.25,
                'opening_home_ml': -150,
                'opening_away_ml': +130,
                'actual_result': 'away_win',  # KC won (RLM!)
                'home_score': 20,
                'away_score': 27,
                'spread_line': -2.5,
                'spread_result': 'away_cover'
            },
            # More games would be loaded here
        ]

    def backtest_trap_strategy(
        self,
        games: List[Dict],
        min_trap_score: int = 60
    ) -> Dict:
        """
        Backtest the trap betting strategy.

        Strategy:
        - When trap score ≤ -60: Fade public (bet opposite)
        - When trap score ≥ +60: Ride sharps (bet with them)

        Args:
            games: List of historical games with handle data
            min_trap_score: Minimum absolute trap score to bet (default: 60)

        Returns:
            Dict with backtest results
        """
        results = {
            'total_games': len(games),
            'bets_placed': 0,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
            'total_wagered': 0,
            'total_profit': 0,
            'roi': 0,
            'win_rate': 0,
            'trap_fade_bets': [],  # Trap ≤ -60
            'sharp_consensus_bets': [],  # Trap ≥ +60
            'by_trap_score': {}
        }

        for game in games:
            # Calculate trap scores
            trap_analysis = self.detector.detect_trap_patterns(
                game['game'],
                game['home_ml'],
                game['away_ml'],
                game['home_handle'],
                game['away_handle'],
                game.get('opening_home_ml'),
                game.get('opening_away_ml')
            )

            home_score = trap_analysis['home_analysis']['trap_score']
            away_score = trap_analysis['away_analysis']['trap_score']

            # Determine if we should bet
            bet_placed = False
            bet_side = None
            bet_type = None

            # FADE PUBLIC (trap ≤ -60)
            if home_score <= -min_trap_score:
                bet_placed = True
                bet_side = 'away'  # Fade home, bet away
                bet_type = 'trap_fade'
            elif away_score <= -min_trap_score:
                bet_placed = True
                bet_side = 'home'  # Fade away, bet home
                bet_type = 'trap_fade'

            # RIDE SHARPS (trap ≥ +60)
            elif home_score >= min_trap_score:
                bet_placed = True
                bet_side = 'home'  # Sharps on home
                bet_type = 'sharp_consensus'
            elif away_score >= min_trap_score:
                bet_placed = True
                bet_side = 'away'  # Sharps on away
                bet_type = 'sharp_consensus'

            if not bet_placed:
                continue  # Skip games without strong signals

            # Determine bet outcome
            actual_result = game['actual_result']

            if bet_side == 'home':
                won = (actual_result == 'home_win')
            else:  # bet_side == 'away'
                won = (actual_result == 'away_win')

            # Calculate profit/loss (assume -110 odds, $100 bet)
            bet_amount = 100
            if won:
                profit = bet_amount * (100 / 110)  # Win $90.91 on $100 bet
            else:
                profit = -bet_amount  # Lose $100

            # Record results
            results['bets_placed'] += 1
            results['total_wagered'] += bet_amount

            if won:
                results['wins'] += 1
                results['total_profit'] += profit
            else:
                results['losses'] += 1
                results['total_profit'] += profit

            # Categorize by bet type
            bet_record = {
                'game': game['game'],
                'week': game['week'],
                'trap_score': home_score if bet_side == 'home' else away_score,
                'bet_side': bet_side,
                'result': 'WIN' if won else 'LOSS',
                'profit': profit
            }

            if bet_type == 'trap_fade':
                results['trap_fade_bets'].append(bet_record)
            else:
                results['sharp_consensus_bets'].append(bet_record)

        # Calculate summary stats
        if results['bets_placed'] > 0:
            results['win_rate'] = (results['wins'] / results['bets_placed']) * 100
            results['roi'] = (results['total_profit'] / results['total_wagered']) * 100

        return results

    def print_results(self, results: Dict):
        """Print backtest results in formatted table."""
        print("\n" + "=" * 70)
        print("🎯 TRAP DETECTOR BACKTEST RESULTS")
        print("=" * 70)
        print()

        print("OVERALL PERFORMANCE:")
        print(f"   Total Games Analyzed: {results['total_games']}")
        print(f"   Bets Placed: {results['bets_placed']}")
        print(f"   Wins: {results['wins']}")
        print(f"   Losses: {results['losses']}")
        print(f"   Win Rate: {results['win_rate']:.1f}%")
        print(f"   Total Wagered: ${results['total_wagered']:,.0f}")
        print(f"   Total Profit: ${results['total_profit']:,.2f}")
        print(f"   ROI: {results['roi']:+.2f}%")
        print()

        # Trap Fade Performance
        trap_fade = results['trap_fade_bets']
        if trap_fade:
            wins = sum(1 for b in trap_fade if b['result'] == 'WIN')
            total = len(trap_fade)
            profit = sum(b['profit'] for b in trap_fade)
            win_rate = (wins / total) * 100

            print("TRAP FADE PERFORMANCE (Betting Against Public):")
            print(f"   Bets: {total}")
            print(f"   Wins: {wins}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Profit: ${profit:,.2f}")
            print(f"   ROI: {(profit / (total * 100)) * 100:+.2f}%")
            print()

        # Sharp Consensus Performance
        sharp = results['sharp_consensus_bets']
        if sharp:
            wins = sum(1 for b in sharp if b['result'] == 'WIN')
            total = len(sharp)
            profit = sum(b['profit'] for b in sharp)
            win_rate = (wins / total) * 100

            print("SHARP CONSENSUS PERFORMANCE (Riding Sharp Money):")
            print(f"   Bets: {total}")
            print(f"   Wins: {wins}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Profit: ${profit:,.2f}")
            print(f"   ROI: {(profit / (total * 100)) * 100:+.2f}%")
            print()

        print("=" * 70)

    def save_results(self, results: Dict, filename: str):
        """Save backtest results to JSON file."""
        output_file = self.results_dir / filename

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest trap detector on historical games"
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Season to backtest (e.g., 2024)"
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="Backtest all available seasons"
    )
    parser.add_argument(
        "--min-trap-score",
        type=int,
        default=60,
        help="Minimum trap score to bet (default: 60)"
    )

    args = parser.parse_args()

    backtester = TrapBacktester()

    if args.all_seasons:
        seasons = [2020, 2021, 2022, 2023, 2024]
    elif args.season:
        seasons = [args.season]
    else:
        print("❌ Please specify --season or --all-seasons")
        parser.print_help()
        return

    for season in seasons:
        print(f"\n{'=' * 70}")
        print(f"📊 BACKTESTING SEASON {season}")
        print(f"{'=' * 70}")

        # Load historical games
        games = backtester.load_historical_games(season)

        if not games:
            print(f"⚠️  No historical data for {season}")
            print("   NOTE: This is sample data - load your own historical games")
            continue

        # Run backtest
        results = backtester.backtest_trap_strategy(
            games,
            min_trap_score=args.min_trap_score
        )

        # Print results
        backtester.print_results(results)

        # Save results
        filename = f"trap_backtest_{season}.json"
        backtester.save_results(results, filename)

    print("\n💡 NEXT STEPS:")
    print("   1. Add real historical game data with handle percentages")
    print("   2. Run backtest on multiple seasons")
    print("   3. Compare trap detector ROI vs baseline strategy")
    print("   4. Optimize min_trap_score threshold")


if __name__ == "__main__":
    main()
