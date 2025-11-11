#!/usr/bin/env python3
"""
Automated Bankroll Tracker - SINGLE SOURCE OF TRUTH
====================================================

WHY THIS EXISTS:
----------------
Agent is stateless and forgets bankroll each session.
Manual tracking causes errors ($100 → $104? Or $103.64?)

INVESTMENT → SYSTEM:
--------------------
Instead of user manually updating bankroll:
- System tracks every bet automatically
- System grades results from API
- System calculates current bankroll
- Hook injects current bankroll into context

ERROR PREVENTION:
-----------------
Agent cannot:
- Guess bankroll (reads from .bankroll file)
- Bet more than available (hook validates against this file)
- Miss updating after win/loss (automatic grading)

OPERATIONAL COST:
-----------------
Initial: 1 hour (this file)
Ongoing: 0 (fully automatic)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

BANKROLL_FILE = Path(__file__).parent / ".bankroll"
BET_LOG_FILE = Path(__file__).parent / "data" / "bet_log.json"


class BankrollTracker:
    """
    Single source of truth for bankroll.

    WHY: Agent queries this instead of guessing.
    """

    def __init__(self):
        self.bankroll_file = BANKROLL_FILE
        self.bet_log_file = BET_LOG_FILE

    def get_current_bankroll(self) -> float:
        """
        Get current bankroll (SINGLE SOURCE OF TRUTH).

        Returns:
            Current bankroll in dollars

        WHY: Agent cannot guess or use stale value.
        """
        if not self.bankroll_file.exists():
            # Initialize with $100 default
            self._set_bankroll(100.0, "Initial bankroll")
            return 100.0

        with open(self.bankroll_file, 'r') as f:
            data = json.load(f)
            return data['current_bankroll']

    def _set_bankroll(self, amount: float, reason: str):
        """
        Set bankroll (private - only called by system).

        WHY: Prevents manual edits, ensures audit trail.
        """
        data = {
            'current_bankroll': amount,
            'last_updated': datetime.now().isoformat(),
            'reason': reason
        }

        with open(self.bankroll_file, 'w') as f:
            json.dump(data, f, indent=2)

    def record_bet(self, amount: float, game: str, pick: str):
        """
        Record bet (deduct from bankroll immediately).

        Args:
            amount: Dollar amount bet
            game: Game identifier
            pick: What was bet on

        WHY: Bankroll reflects pending bets immediately.
             Cannot double-bet same amount.
        """
        current = self.get_current_bankroll()

        if amount > current:
            raise ValueError(
                f"Insufficient bankroll: ${amount} bet > ${current} available\n"
                "DECISION: Reduce bet size or skip bet"
            )

        # Deduct immediately
        new_bankroll = current - amount
        self._set_bankroll(new_bankroll, f"Bet placed: {game} - {pick} (${amount})")

        # Log bet to bet_log.json
        if not self.bet_log_file.parent.exists():
            self.bet_log_file.parent.mkdir(parents=True, exist_ok=True)

        if self.bet_log_file.exists():
            with open(self.bet_log_file, 'r') as f:
                bets = json.load(f)
        else:
            bets = []

        bets.append({
            'game': game,
            'pick': pick,
            'amount': amount,
            'placed_at': datetime.now().isoformat(),
            'result': 'PENDING'
        })

        with open(self.bet_log_file, 'w') as f:
            json.dump(bets, f, indent=2)

        return new_bankroll

    def record_result(self, game: str, result: str, odds: int = -110):
        """
        Record bet result (update bankroll automatically).

        Args:
            game: Game identifier
            result: "WIN", "LOSS", or "PUSH"
            odds: Odds the bet was placed at (default -110)

        WHY: Eliminates manual calculation errors.
        """
        # Find bet in log
        if not self.bet_log_file.exists():
            raise ValueError(f"Bet log not found: {self.bet_log_file}")

        with open(self.bet_log_file, 'r') as f:
            bets = json.load(f)

        # Find this game's bet
        bet = None
        for b in bets:
            if b['game'] == game and b.get('result') == 'PENDING':
                bet = b
                break

        if not bet:
            raise ValueError(f"No pending bet found for game: {game}")

        bet_amount = bet['amount']
        current = self.get_current_bankroll()

        if result == "WIN":
            # Calculate winnings based on odds
            if odds < 0:
                # American odds (negative)
                profit = bet_amount / (abs(odds) / 100)
            else:
                # American odds (positive)
                profit = bet_amount * (odds / 100)

            # Return original bet + profit
            new_bankroll = current + bet_amount + profit
            reason = f"{game} WON: +${profit:.2f} profit"

        elif result == "LOSS":
            # Already deducted when bet was placed
            new_bankroll = current
            reason = f"{game} LOST: -${bet_amount} (already deducted)"

        elif result == "PUSH":
            # Return original bet
            new_bankroll = current + bet_amount
            reason = f"{game} PUSH: ${bet_amount} returned"

        else:
            raise ValueError(f"Invalid result: {result}. Must be WIN, LOSS, or PUSH")

        self._set_bankroll(new_bankroll, reason)

        # Update bet log
        bet['result'] = result
        bet['settled_at'] = datetime.now().isoformat()

        with open(self.bet_log_file, 'w') as f:
            json.dump(bets, f, indent=2)

        return new_bankroll

    def get_pending_bets(self):
        """
        Get all pending bets (not yet graded).

        Returns:
            List of pending bets

        WHY: Agent knows what results to check for.
        """
        if not self.bet_log_file.exists():
            return []

        with open(self.bet_log_file, 'r') as f:
            bets = json.load(f)

        return [b for b in bets if b.get('result') == 'PENDING']

    def get_stats(self):
        """
        Get bankroll statistics.

        Returns:
            Dict with ROI, win rate, total profit

        WHY: Always know performance vs expectations.
        """
        if not self.bet_log_file.exists():
            current = self.get_current_bankroll()
            initial = 100.0
            return {
                'current_bankroll': current,
                'initial_bankroll': initial,
                'total_bets': 0,
                'wins': 0,
                'losses': 0,
                'pushes': 0,
                'win_rate': 0.0,
                'roi': 0.0,
                'total_profit': current - initial,
                'pending_bets': 0
            }

        with open(self.bet_log_file, 'r') as f:
            bets = json.load(f)

        wins = sum(1 for b in bets if b.get('result') == 'WIN')
        losses = sum(1 for b in bets if b.get('result') == 'LOSS')
        pushes = sum(1 for b in bets if b.get('result') == 'PUSH')
        total_bet = sum(b['amount'] for b in bets if b.get('result') != 'PENDING')

        current = self.get_current_bankroll()
        initial = 100.0  # Starting bankroll
        total_profit = current - initial

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
        roi = (total_profit / total_bet * 100) if total_bet > 0 else 0.0

        return {
            'current_bankroll': current,
            'initial_bankroll': initial,
            'total_bets': len([b for b in bets if b.get('result') != 'PENDING']),
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'win_rate': win_rate,
            'roi': roi,
            'total_profit': total_profit,
            'pending_bets': len(self.get_pending_bets())
        }


def main():
    """CLI interface for bankroll operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Bankroll Tracker")
    parser.add_argument('--balance', action='store_true', help='Show current balance')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--pending', action='store_true', help='Show pending bets')
    parser.add_argument('--grade', nargs=2, metavar=('GAME', 'RESULT'),
                       help='Grade bet result (e.g., --grade "PHI @ GB" WIN)')

    args = parser.parse_args()

    tracker = BankrollTracker()

    if args.balance:
        balance = tracker.get_current_bankroll()
        print(f"${balance:.2f}")

    elif args.stats:
        stats = tracker.get_stats()
        print("\n" + "="*60)
        print("📊 BANKROLL STATISTICS")
        print("="*60)
        print(f"Current Bankroll:  ${stats['current_bankroll']:.2f}")
        print(f"Initial Bankroll:  ${stats['initial_bankroll']:.2f}")
        print(f"Total Profit:      ${stats['total_profit']:.2f}")
        print(f"ROI:               {stats['roi']:.2f}%")
        print(f"\nTotal Bets:        {stats['total_bets']}")
        print(f"Wins:              {stats['wins']}")
        print(f"Losses:            {stats['losses']}")
        print(f"Pushes:            {stats['pushes']}")
        print(f"Win Rate:          {stats['win_rate']:.2%}")
        print(f"Pending:           {stats['pending_bets']}")
        print("="*60)

    elif args.pending:
        pending = tracker.get_pending_bets()
        if not pending:
            print("No pending bets")
        else:
            print("\n" + "="*60)
            print("⏳ PENDING BETS")
            print("="*60)
            for bet in pending:
                print(f"{bet['game']}: {bet['pick']} (${bet['amount']})")
            print("="*60)

    elif args.grade:
        game, result = args.grade
        result = result.upper()

        if result not in ['WIN', 'LOSS', 'PUSH']:
            print(f"ERROR: Result must be WIN, LOSS, or PUSH (got: {result})")
            return

        try:
            new_balance = tracker.record_result(game, result)
            print(f"✅ {game} graded as {result}")
            print(f"New bankroll: ${new_balance:.2f}")
        except ValueError as e:
            print(f"ERROR: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
