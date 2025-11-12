#!/usr/bin/env python3
"""
NCAA Closing Line Value (CLV) Tracker
======================================

THE #1 PREDICTOR OF LONG-TERM PROFITABILITY

What is CLV?
- Closing Line Value = Your bet line vs Closing line
- If you consistently beat closing line → You WILL profit
- Sharps average +2 points CLV

Why it matters:
- CLV is better predictor than win rate
- You can lose bets but have +CLV (you're still sharp)
- You can win bets but have -CLV (you're getting lucky)

Example:
  Your bet: Toledo -3.0 (Thursday)
  Closing line: Toledo -5.0 (Saturday)
  CLV: +2.0 points (EXCELLENT!)

Goal: Average +1.0 to +2.0 points CLV

USAGE:
    from ncaa_clv_tracker import CLVTracker

    tracker = CLVTracker()
    tracker.record_bet(
        game="Toledo @ BG",
        your_line=-3.0,
        closing_line=-5.0,
        bet_amount=100,
        result='win'  # or 'loss'
    )

    tracker.get_clv_stats()
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class CLVBet:
    """Single bet with CLV tracking"""
    bet_id: str
    timestamp: str
    game: str
    your_line: float
    closing_line: float
    clv: float  # Your line - Closing line
    bet_amount: float
    result: Optional[str] = None  # 'win', 'loss', 'push', 'pending'
    profit: Optional[float] = None
    notes: Optional[str] = None


class CLVTracker:
    """
    Track Closing Line Value for all bets

    The #1 metric sharp bettors use to measure edge
    """

    def __init__(self, data_file: str = "data/clv_history.json"):
        self.data_file = Path(data_file)
        self.bets: List[CLVBet] = []
        self._load_history()

    def _load_history(self):
        """Load bet history from file"""
        if self.data_file.exists():
            with open(self.data_file) as f:
                data = json.load(f)
                self.bets = [CLVBet(**bet) for bet in data.get('bets', [])]

    def _save_history(self):
        """Save bet history to file"""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.data_file, 'w') as f:
            json.dump({
                'last_updated': datetime.now().isoformat(),
                'total_bets': len(self.bets),
                'bets': [asdict(bet) for bet in self.bets]
            }, f, indent=2)

    def record_bet(
        self,
        game: str,
        your_line: float,
        closing_line: float,
        bet_amount: float,
        result: Optional[str] = None,
        profit: Optional[float] = None,
        notes: Optional[str] = None
    ) -> CLVBet:
        """
        Record a bet with CLV

        Args:
            game: Game description (e.g., "Toledo @ BG")
            your_line: Line you bet at (e.g., -3.0)
            closing_line: Closing line (e.g., -5.0)
            bet_amount: Amount wagered
            result: 'win', 'loss', 'push', or 'pending'
            profit: Actual profit/loss
            notes: Optional notes

        Returns:
            CLVBet object
        """

        # Calculate CLV
        clv = your_line - closing_line

        # Generate bet ID
        bet_id = f"NCAA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create bet record
        bet = CLVBet(
            bet_id=bet_id,
            timestamp=datetime.now().isoformat(),
            game=game,
            your_line=your_line,
            closing_line=closing_line,
            clv=clv,
            bet_amount=bet_amount,
            result=result,
            profit=profit,
            notes=notes
        )

        self.bets.append(bet)
        self._save_history()

        # Print CLV analysis
        self._print_bet_clv(bet)

        return bet

    def update_result(
        self,
        bet_id: str,
        result: str,
        profit: float
    ):
        """Update bet result after game completes"""
        for bet in self.bets:
            if bet.bet_id == bet_id:
                bet.result = result
                bet.profit = profit
                self._save_history()
                print(f"✅ Updated bet {bet_id}: {result} (${profit:+.2f})")
                return

        print(f"⚠️  Bet {bet_id} not found")

    def _print_bet_clv(self, bet: CLVBet):
        """Print CLV analysis for single bet"""
        print(f"\n📊 CLV ANALYSIS")
        print(f"{'='*60}")
        print(f"Game: {bet.game}")
        print(f"Your line: {bet.your_line:+.1f}")
        print(f"Closing line: {bet.closing_line:+.1f}")
        print(f"CLV: {bet.clv:+.1f} points")
        print()

        # Interpret CLV
        if bet.clv >= 2.0:
            print(f"🔥 EXCELLENT CLV (+{bet.clv:.1f}) - Elite sharp level!")
        elif bet.clv >= 1.0:
            print(f"✅ GOOD CLV (+{bet.clv:.1f}) - Above sharp average")
        elif bet.clv >= 0.5:
            print(f"👍 POSITIVE CLV (+{bet.clv:.1f}) - Beating market")
        elif bet.clv >= 0:
            print(f"😐 NEUTRAL CLV ({bet.clv:+.1f}) - At market")
        elif bet.clv >= -1.0:
            print(f"⚠️  NEGATIVE CLV ({bet.clv:+.1f}) - Below market")
        else:
            print(f"❌ BAD CLV ({bet.clv:+.1f}) - Significantly below market")

        print(f"{'='*60}\n")

    def get_clv_stats(self) -> Dict:
        """Get overall CLV statistics"""

        if not self.bets:
            return {
                'total_bets': 0,
                'avg_clv': 0,
                'positive_clv_pct': 0,
                'message': 'No bets recorded yet'
            }

        # Calculate stats
        total_bets = len(self.bets)
        avg_clv = sum(bet.clv for bet in self.bets) / total_bets
        positive_clv = sum(1 for bet in self.bets if bet.clv > 0)
        positive_clv_pct = (positive_clv / total_bets) * 100

        # CLV by result (if results available)
        winning_bets = [bet for bet in self.bets if bet.result == 'win']
        losing_bets = [bet for bet in self.bets if bet.result == 'loss']

        avg_clv_winners = sum(b.clv for b in winning_bets) / len(winning_bets) if winning_bets else 0
        avg_clv_losers = sum(b.clv for b in losing_bets) / len(losing_bets) if losing_bets else 0

        # Total profit (if available)
        total_profit = sum(bet.profit for bet in self.bets if bet.profit is not None)

        stats = {
            'total_bets': total_bets,
            'avg_clv': avg_clv,
            'positive_clv_pct': positive_clv_pct,
            'avg_clv_winners': avg_clv_winners,
            'avg_clv_losers': avg_clv_losers,
            'total_profit': total_profit
        }

        return stats

    def print_stats(self):
        """Print CLV statistics report"""
        stats = self.get_clv_stats()

        if stats['total_bets'] == 0:
            print("No bets recorded yet.")
            return

        print(f"\n{'='*80}")
        print(f"📊 CLV TRACKER STATISTICS")
        print(f"{'='*80}\n")

        print(f"Total Bets: {stats['total_bets']}")
        print(f"Average CLV: {stats['avg_clv']:+.2f} points")
        print(f"Positive CLV: {stats['positive_clv_pct']:.1f}%")
        print()

        # Interpret average CLV
        avg_clv = stats['avg_clv']
        if avg_clv >= 2.0:
            print(f"🔥 ELITE SHARP BETTOR - You're crushing the market!")
        elif avg_clv >= 1.0:
            print(f"✅ SHARP BETTOR - Above average, profitable long-term")
        elif avg_clv >= 0.5:
            print(f"👍 GOOD - Beating market, likely profitable")
        elif avg_clv >= 0:
            print(f"😐 NEUTRAL - Breaking even with market")
        elif avg_clv >= -0.5:
            print(f"⚠️  BELOW MARKET - Need to improve line timing")
        else:
            print(f"❌ CHASING LINES - Betting too late, getting bad prices")

        print()

        # Win/loss CLV comparison
        if stats['avg_clv_winners'] or stats['avg_clv_losers']:
            print(f"CLV on Winners: {stats['avg_clv_winners']:+.2f}")
            print(f"CLV on Losers: {stats['avg_clv_losers']:+.2f}")
            print()

            if stats['avg_clv_losers'] > 0:
                print(f"💡 INSIGHT: Even your losses have +CLV!")
                print(f"   This means you're sharp but got unlucky.")
                print(f"   Keep betting - long-term profit expected.")
            elif stats['avg_clv_winners'] > 0 and stats['avg_clv_losers'] < 0:
                print(f"💡 INSIGHT: Winning bets have +CLV, losers have -CLV")
                print(f"   You're identifying good spots but need better timing.")

        if stats['total_profit'] != 0:
            print(f"\nTotal Profit: ${stats['total_profit']:+.2f}")

        print(f"\n{'='*80}\n")

    def get_recent_bets(self, n: int = 10) -> List[CLVBet]:
        """Get most recent N bets"""
        return sorted(self.bets, key=lambda b: b.timestamp, reverse=True)[:n]

    def print_recent_bets(self, n: int = 10):
        """Print recent bets with CLV"""
        recent = self.get_recent_bets(n)

        if not recent:
            print("No bets recorded yet.")
            return

        print(f"\n{'='*80}")
        print(f"📋 RECENT BETS (Last {min(n, len(recent))})")
        print(f"{'='*80}\n")

        for bet in recent:
            result_icon = {
                'win': '✅',
                'loss': '❌',
                'push': '➖',
                'pending': '⏳'
            }.get(bet.result, '❓')

            clv_icon = '🔥' if bet.clv >= 1.0 else '✅' if bet.clv > 0 else '⚠️'

            print(f"{result_icon} {bet.game}")
            print(f"   Line: {bet.your_line:+.1f} → Close: {bet.closing_line:+.1f}")
            print(f"   {clv_icon} CLV: {bet.clv:+.1f} points | Bet: ${bet.bet_amount:.0f}")
            if bet.profit is not None:
                print(f"   Profit: ${bet.profit:+.2f}")
            print()


def main():
    """Demo CLV tracker"""

    print("NCAA CLV Tracker Demo\n")

    tracker = CLVTracker()

    # Example 1: Great CLV
    print("Example 1: Great CLV (+2.5 points)")
    tracker.record_bet(
        game="Toledo @ Bowling Green",
        your_line=-3.0,
        closing_line=-5.5,
        bet_amount=100,
        result='win',
        profit=90.91,
        notes="Tuesday MACtion early bet"
    )

    # Example 2: Negative CLV
    print("\nExample 2: Bad CLV (-1.5 points)")
    tracker.record_bet(
        game="Alabama vs Auburn",
        your_line=-7.0,
        closing_line=-5.5,
        bet_amount=100,
        result='win',
        profit=90.91,
        notes="Bet too late, line moved against us"
    )

    # Example 3: Losing bet with positive CLV
    print("\nExample 3: Loss but GOOD CLV (+1.0)")
    tracker.record_bet(
        game="Ohio vs Kent State",
        your_line=+3.0,
        closing_line=+4.0,
        bet_amount=100,
        result='loss',
        profit=-110.00,
        notes="Lost but had good line - sharp bet"
    )

    # Print overall stats
    tracker.print_stats()

    # Print recent bets
    tracker.print_recent_bets()


if __name__ == "__main__":
    main()
