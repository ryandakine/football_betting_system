#!/usr/bin/env python3
"""
Manual bet logging - For when you need to add historical bets

USAGE:
    python log_bet_manual.py --game "PHI @ GB" --pick "GB -1.5" --amount 4 --odds -110 --result LOSS
    python log_bet_manual.py --game "PHI @ GB" --pick "UNDER 45.5" --amount 6 --odds -110 --result WIN
"""

import argparse
import sys
from pathlib import Path

try:
    from bankroll_tracker import BankrollTracker
except ImportError:
    print("❌ Could not import bankroll_tracker")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Manually log a bet result")
    parser.add_argument("--game", required=True, help="Game (e.g. 'PHI @ GB')")
    parser.add_argument("--pick", required=True, help="Your pick (e.g. 'GB -1.5')")
    parser.add_argument("--amount", type=float, required=True, help="Bet amount")
    parser.add_argument("--odds", type=int, default=-110, help="American odds (default: -110)")
    parser.add_argument("--result", required=True, choices=['WIN', 'LOSS', 'PUSH'], help="Bet result")

    args = parser.parse_args()

    tracker = BankrollTracker()

    # Step 1: Record the bet (deducts from bankroll)
    print(f"📝 Logging bet: {args.pick} for ${args.amount}")
    tracker.record_bet(amount=args.amount, game=args.game, pick=args.pick)

    # Step 2: Record the result
    print(f"📊 Recording result: {args.result}")
    tracker.record_result(game=args.game, result=args.result, odds=args.odds)

    # Show updated stats
    print("\n" + "="*60)
    stats = tracker.get_stats()
    print(f"Updated Bankroll: ${stats['current_bankroll']:.2f}")
    print(f"Total Profit: ${stats['total_profit']:.2f}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()
