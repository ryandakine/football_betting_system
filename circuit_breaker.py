#!/usr/bin/env python3
"""
Emergency Circuit Breaker
==========================

WHY THIS EXISTS:
----------------
Even with 74.57% win rate, variance happens.
Bad streaks can wipe out bankroll without protection.

INVESTMENT → SYSTEM:
--------------------
System automatically:
- Detects losing streaks
- Reduces bet sizes during variance
- Stops betting if drawdown exceeds limits
- Resumes when bankroll recovers

ERROR PREVENTION:
-----------------
Agent cannot:
- Bet during emergency drawdown
- Chase losses with larger bets
- Ignore risk of ruin

OPERATIONAL COST:
-----------------
Initial: 30 min (this file)
Ongoing: 0 (automatic protection)

DESIGN PHILOSOPHY:
------------------
"Make mistakes structurally impossible"
→ Hook blocks bets during circuit breaker activation
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from bankroll_tracker import BankrollTracker

CIRCUIT_BREAKER_FILE = Path(__file__).parent / ".circuit_breaker"


class CircuitBreaker:
    """
    Protects bankroll during variance.

    WHY: Prevents ruin from bad streaks.
    """

    # Thresholds
    DRAWDOWN_WARNING = 0.10      # 10% drawdown → warning
    DRAWDOWN_REDUCE = 0.15       # 15% drawdown → reduce bet sizes
    DRAWDOWN_STOP = 0.25         # 25% drawdown → STOP betting
    LOSING_STREAK_REDUCE = 3     # 3 losses in row → reduce sizes
    LOSING_STREAK_STOP = 5       # 5 losses in row → STOP betting

    def __init__(self):
        self.tracker = BankrollTracker()
        self.breaker_file = CIRCUIT_BREAKER_FILE

    def check_status(self) -> Tuple[str, Dict]:
        """
        Check circuit breaker status.

        Returns:
            Tuple of (status, details)
            status: "NORMAL", "WARNING", "REDUCED", "STOPPED"

        WHY: Single source of truth for betting permissions.
        """
        stats = self.tracker.get_stats()

        initial = stats['initial_bankroll']
        current = stats['current_bankroll']
        drawdown = (initial - current) / initial if initial > 0 else 0

        # Check losing streak
        losing_streak = self._get_losing_streak()

        # Determine status
        if drawdown >= self.DRAWDOWN_STOP or losing_streak >= self.LOSING_STREAK_STOP:
            status = "STOPPED"
            reason = f"Drawdown {drawdown:.1%} or {losing_streak} losses in a row"

        elif drawdown >= self.DRAWDOWN_REDUCE or losing_streak >= self.LOSING_STREAK_REDUCE:
            status = "REDUCED"
            reason = f"Drawdown {drawdown:.1%} or {losing_streak} losses"

        elif drawdown >= self.DRAWDOWN_WARNING:
            status = "WARNING"
            reason = f"Drawdown {drawdown:.1%}"

        else:
            status = "NORMAL"
            reason = "All systems operational"

        details = {
            'status': status,
            'reason': reason,
            'current_bankroll': current,
            'initial_bankroll': initial,
            'drawdown_percent': drawdown * 100,
            'losing_streak': losing_streak,
            'win_rate': stats['win_rate'],
            'roi': stats['roi']
        }

        # Save status
        with open(self.breaker_file, 'w') as f:
            json.dump(details, f, indent=2)

        return status, details

    def _get_losing_streak(self) -> int:
        """
        Get current losing streak.

        Returns:
            Number of consecutive losses

        WHY: Detect variance early.
        """
        bet_log_file = Path(__file__).parent / "data" / "bet_log.json"

        if not bet_log_file.exists():
            return 0

        with open(bet_log_file, 'r') as f:
            bets = json.load(f)

        # Get recent results (reverse chronological)
        recent_results = []
        for bet in reversed(bets):
            result = bet.get('result')
            if result in ['WIN', 'LOSS']:
                recent_results.append(result)
            if len(recent_results) >= 10:  # Check last 10
                break

        # Count consecutive losses
        streak = 0
        for result in recent_results:
            if result == 'LOSS':
                streak += 1
            else:
                break

        return streak

    def get_adjusted_bet_size(self, confidence: int, normal_bet: int) -> int:
        """
        Get bet size adjusted for circuit breaker status.

        Args:
            confidence: Bet confidence (70-100)
            normal_bet: Normal bet size (2-6 units)

        Returns:
            Adjusted bet size

        WHY: Automatically reduces risk during variance.
        """
        status, details = self.check_status()

        if status == "STOPPED":
            return 0  # No betting allowed

        elif status == "REDUCED":
            # Half normal bet size
            return max(1, normal_bet // 2)

        elif status == "WARNING":
            # Slightly reduced (75% of normal)
            return max(1, int(normal_bet * 0.75))

        else:
            # Normal betting
            return normal_bet

    def can_bet(self) -> Tuple[bool, str]:
        """
        Check if betting is allowed.

        Returns:
            Tuple of (allowed, reason)

        WHY: Single decision point for bet permission.
        """
        status, details = self.check_status()

        if status == "STOPPED":
            return False, f"Circuit breaker ACTIVE: {details['reason']}"

        elif status == "REDUCED":
            return True, f"Reduced betting: {details['reason']}"

        elif status == "WARNING":
            return True, f"Proceed with caution: {details['reason']}"

        else:
            return True, "Normal operation"


def main():
    """CLI interface for circuit breaker."""
    import argparse

    parser = argparse.ArgumentParser(description="Circuit Breaker Status")
    parser.add_argument('--status', action='store_true',
                       help='Show current status')
    parser.add_argument('--check-bet', type=int, metavar='CONFIDENCE',
                       help='Check if bet allowed at confidence level')

    args = parser.parse_args()

    breaker = CircuitBreaker()

    if args.status:
        status, details = breaker.check_status()

        emoji = {
            "NORMAL": "✅",
            "WARNING": "⚠️ ",
            "REDUCED": "🔶",
            "STOPPED": "🛑"
        }.get(status, "❓")

        print(f"\n{'='*60}")
        print(f"{emoji} CIRCUIT BREAKER STATUS: {status}")
        print(f"{'='*60}")
        print(f"Bankroll:       ${details['current_bankroll']:.2f} / ${details['initial_bankroll']:.2f}")
        print(f"Drawdown:       {details['drawdown_percent']:.1f}%")
        print(f"Losing Streak:  {details['losing_streak']}")
        print(f"Win Rate:       {details['win_rate']:.2%}")
        print(f"ROI:            {details['roi']:.2f}%")
        print(f"\nReason: {details['reason']}")

        if status == "STOPPED":
            print(f"\n🛑 BETTING DISABLED")
            print(f"   Wait for bankroll to recover above ${details['initial_bankroll'] * 0.85:.2f}")

        elif status == "REDUCED":
            print(f"\n🔶 BET SIZES REDUCED TO 50%")
            print(f"   Normal 6 units → 3 units")
            print(f"   Normal 4 units → 2 units")
            print(f"   Normal 2 units → 1 unit")

        elif status == "WARNING":
            print(f"\n⚠️  PROCEED WITH CAUTION")
            print(f"   Bet sizes reduced to 75%")

        print(f"{'='*60}\n")

    elif args.check_bet:
        confidence = args.check_bet

        # Determine normal bet size
        if confidence >= 80:
            normal_bet = 6
        elif confidence >= 75:
            normal_bet = 4
        elif confidence >= 70:
            normal_bet = 2
        else:
            print(f"❌ Confidence {confidence}% below 70% threshold")
            return

        # Check circuit breaker
        allowed, reason = breaker.can_bet()

        if not allowed:
            print(f"🛑 BET BLOCKED: {reason}")
            return

        # Get adjusted bet size
        adjusted_bet = breaker.get_adjusted_bet_size(confidence, normal_bet)

        print(f"\n✅ Bet allowed at {confidence}% confidence")
        print(f"   Normal bet size: {normal_bet} units (${normal_bet * 2})")
        print(f"   Adjusted bet size: {adjusted_bet} units (${adjusted_bet * 2})")
        print(f"   Reason: {reason}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
