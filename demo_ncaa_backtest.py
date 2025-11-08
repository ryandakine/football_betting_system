#!/usr/bin/env python3
"""
Standalone NCAA Backtester Demo
Demonstrates the backtesting engine without complex dependencies
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


class SimpleNCAABacktester:
    """Simplified NCAA backtester for demo purposes"""

    def __init__(self, bankroll=10000, unit_size=100, min_edge=0.03, min_confidence=0.60):
        self.bankroll_start = bankroll
        self.unit_size = unit_size
        self.min_edge = min_edge
        self.min_confidence = min_confidence

    async def run_backtest(self, season='2023'):
        """Run backtest on historical data"""
        print(f"\n🏈 NCAA BACKTEST - {season} Season")
        print("=" * 70)

        # Load historical data
        data_file = Path(f"data/football/historical/ncaaf/ncaaf_{season}.csv")
        if not data_file.exists():
            print(f"❌ No data found for {season} season")
            return None

        df = pd.read_csv(data_file)
        print(f"✅ Loaded {len(df)} games from {season} season\n")

        # Initialize tracking
        bankroll = self.bankroll_start
        bets_placed = 0
        wins = 0
        total_profit = 0.0
        bet_history = []

        print("Processing games...")
        print("-" * 70)

        # Simulate betting on each game
        for idx, game in df.iterrows():
            edge = game['edge_value']
            confidence = game['confidence']
            odds = game['odds']
            actual_result = game['actual_result']

            # Skip if below thresholds
            if confidence < self.min_confidence or edge < self.min_edge:
                continue

            # Calculate stake using Kelly criterion
            unit_mult = 1.0 + (edge * 5.0)
            unit_mult = np.clip(unit_mult, 0.5, 3.0)
            stake = min(bankroll * 0.12, self.unit_size * unit_mult)
            stake = min(stake, bankroll)

            if stake < 1:
                continue

            # Calculate payout
            payout_mult = odds / 100.0 if odds >= 0 else 100.0 / abs(odds)
            won = bool(actual_result)
            profit = stake * payout_mult if won else -stake

            # Update bankroll
            bankroll += profit
            total_profit += profit
            bets_placed += 1
            if won:
                wins += 1

            # Track bet
            bet_history.append({
                'game': f"{game['away_team']} @ {game['home_team']}",
                'conference': game.get('conference', 'Unknown'),
                'stake': stake,
                'profit': profit,
                'won': won,
                'edge': edge,
                'confidence': confidence,
                'bankroll': bankroll
            })

            # Print each bet
            result_emoji = "✅" if won else "❌"
            print(f"{result_emoji} Week {game.get('week', idx+1):2d}: {game['away_team']:15s} @ {game['home_team']:15s} | "
                  f"Stake: ${stake:6.2f} | Profit: ${profit:7.2f} | Edge: {edge:.1%} | Conf: {confidence:.0%}")

        # Calculate metrics
        win_rate = wins / bets_placed if bets_placed > 0 else 0
        roi = (total_profit / self.bankroll_start) * 100 if self.bankroll_start > 0 else 0

        # Calculate advanced metrics
        profits = [bet['profit'] for bet in bet_history]
        if profits:
            std_dev = np.std(profits, ddof=1)
            mean_profit = np.mean(profits)
            sharpe = (mean_profit / std_dev) * np.sqrt(len(profits)) if std_dev > 0 else 0

            # Max drawdown
            bankroll_curve = [bet['bankroll'] for bet in bet_history]
            peak = bankroll_curve[0]
            max_dd = 0
            for value in bankroll_curve:
                peak = max(peak, value)
                if peak > 0:
                    dd = (peak - value) / peak
                    max_dd = max(max_dd, dd)
        else:
            sharpe = 0
            max_dd = 0

        # Display results
        print("\n")
        print("=" * 70)
        print("📊 BACKTEST RESULTS")
        print("=" * 70)
        print(f"Season: {season}")
        print(f"Games Analyzed: {len(df)}")
        print(f"Bets Placed: {bets_placed}")
        print(f"Winning Bets: {wins}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Starting Bankroll: ${self.bankroll_start:,.2f}")
        print(f"Final Bankroll: ${bankroll:,.2f}")
        print(f"Total Profit: ${total_profit:,.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"\nSharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.1%}")

        # Performance grading
        if roi > 18:
            grade = "EXCELLENT"
        elif roi > 10:
            grade = "GOOD"
        elif roi > 3:
            grade = "AVERAGE"
        else:
            grade = "POOR"

        print(f"\n📈 Performance Grade: {grade}")

        # Recommendations
        print("\n💡 Recommendations:")
        if win_rate < 0.54:
            print("   • Win rate under 54%. Consider tightening confidence thresholds.")
        if roi < 6:
            print("   • ROI modest. Review edge calibration and staking strategy.")
        if bets_placed < len(df) * 0.18:
            print("   • Bet volume low. Consider reducing min edge threshold.")
        print("   • Leverage game prioritizer to tune conference weights.")
        print("   • Focus on Group of Five conferences for hidden value.")

        return {
            'season': season,
            'games_analyzed': len(df),
            'bets_placed': bets_placed,
            'wins': wins,
            'win_rate': win_rate,
            'roi': roi,
            'total_profit': total_profit,
            'final_bankroll': bankroll,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'grade': grade
        }


async def main():
    """Run demo"""
    print("\n" + "="*70)
    print("NCAA FOOTBALL BACKTESTING SYSTEM - DEMO")
    print("="*70)

    backtester = SimpleNCAABacktester(
        bankroll=10000,
        unit_size=100,
        min_edge=0.03,
        min_confidence=0.60
    )

    results = await backtester.run_backtest(season='2023')

    if results:
        print(f"\n✅ Backtest completed successfully!")
        print(f"\nKey Metrics:")
        print(f"  - Win Rate: {results['win_rate']:.1%}")
        print(f"  - ROI: {results['roi']:.2f}%")
        print(f"  - Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  - Grade: {results['grade']}")


if __name__ == "__main__":
    asyncio.run(main())
