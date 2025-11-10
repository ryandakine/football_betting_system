#!/usr/bin/env python3
"""
Parlay Hedge Calculator
=======================
Calculates optimal hedge amounts for live parlays.

Scenarios:
1. All legs hit except last one (hedge final leg)
2. Mid-parlay hedging (some legs hit, some pending)
3. Live game hedging (game in progress)
4. Arbitrage hedging (lock in guaranteed profit)

Usage:
    python parlay_hedge_calculator.py --parlay-odds 600 --bet 100 --hedge-odds -110
    python parlay_hedge_calculator.py --legs-hit 2 --legs-pending 1 --original-bet 100
"""

import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class HedgeCalculation:
    """Result of hedge calculation."""
    original_bet: float
    parlay_odds: int
    potential_parlay_payout: float
    hedge_bet_amount: float
    hedge_odds: int
    hedge_payout: float

    # Outcomes
    if_parlay_wins: float
    if_parlay_loses: float
    guaranteed_profit: Optional[float]
    break_even_hedge: float

    # Recommendations
    recommendation: str
    details: str


class ParlayHedgeCalculator:
    """Calculates optimal hedge strategies for parlays."""

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))

    def calculate_parlay_odds(self, leg_odds: List[int]) -> int:
        """Calculate combined parlay odds from individual legs."""
        combined_decimal = 1.0
        for odds in leg_odds:
            combined_decimal *= self.american_to_decimal(odds)

        return self.decimal_to_american(combined_decimal)

    def calculate_hedge(
        self,
        original_bet: float,
        parlay_odds: int,
        hedge_odds: int,
        hedge_goal: str = 'guaranteed_profit'
    ) -> HedgeCalculation:
        """
        Calculate optimal hedge amount.

        Args:
            original_bet: Amount wagered on parlay
            parlay_odds: Combined parlay odds (American)
            hedge_odds: Odds available for hedge bet (American)
            hedge_goal: 'guaranteed_profit', 'maximize_profit', or 'break_even'

        Returns:
            HedgeCalculation with all scenarios
        """
        # Calculate potential parlay payout
        parlay_decimal = self.american_to_decimal(parlay_odds)
        parlay_payout = original_bet * parlay_decimal

        # Calculate hedge odds in decimal
        hedge_decimal = self.american_to_decimal(hedge_odds)

        # Different hedge strategies
        if hedge_goal == 'guaranteed_profit':
            # Hedge amount that guarantees equal profit either way
            # If parlay wins: parlay_payout - original_bet - hedge_amount
            # If hedge wins: (hedge_amount * hedge_decimal) - original_bet - hedge_amount
            # Set equal: parlay_payout - original_bet - H = H * hedge_decimal - original_bet - H
            # parlay_payout = H * hedge_decimal
            hedge_amount = parlay_payout / hedge_decimal

        elif hedge_goal == 'maximize_profit':
            # Hedge just enough to get original bet back if parlay loses
            hedge_amount = original_bet / (hedge_decimal - 1)

        elif hedge_goal == 'break_even':
            # Hedge to break even if parlay loses
            hedge_amount = original_bet / (hedge_decimal - 1)

        else:
            hedge_amount = 0.0

        # Calculate outcomes
        if_parlay_wins = (parlay_payout - original_bet) - hedge_amount
        if_hedge_wins = (hedge_amount * hedge_decimal - hedge_amount) - original_bet

        # Determine if guaranteed profit exists
        if if_parlay_wins > 0 and if_hedge_wins > 0:
            guaranteed_profit = min(if_parlay_wins, if_hedge_wins)
        else:
            guaranteed_profit = None

        # Break-even hedge
        break_even_hedge = original_bet / (hedge_decimal - 1)

        # Generate recommendation
        if guaranteed_profit and guaranteed_profit > 0:
            recommendation = "HEDGE - Lock in guaranteed profit"
            details = f"Win ${guaranteed_profit:.2f} either way"
        elif if_parlay_wins > original_bet * 0.5:
            recommendation = "LET IT RIDE - High upside remains"
            details = f"Potential profit: ${if_parlay_wins:.2f} if parlay hits"
        else:
            recommendation = "CONSIDER HEDGING - Protect investment"
            details = f"Risk ${original_bet:.2f} for ${if_parlay_wins:.2f} gain"

        return HedgeCalculation(
            original_bet=original_bet,
            parlay_odds=parlay_odds,
            potential_parlay_payout=parlay_payout,
            hedge_bet_amount=hedge_amount,
            hedge_odds=hedge_odds,
            hedge_payout=hedge_amount * hedge_decimal,
            if_parlay_wins=if_parlay_wins,
            if_parlay_loses=if_hedge_wins,
            guaranteed_profit=guaranteed_profit,
            break_even_hedge=break_even_hedge,
            recommendation=recommendation,
            details=details
        )

    def calculate_partial_hedge(
        self,
        original_bet: float,
        legs_hit: int,
        legs_pending: int,
        leg_odds: List[int],
        final_hedge_odds: int
    ) -> Dict:
        """
        Calculate hedge for partially completed parlay.

        Args:
            original_bet: Original parlay wager
            legs_hit: Number of legs that already won
            legs_pending: Number of legs still pending
            leg_odds: Odds for remaining legs
            final_hedge_odds: Odds to hedge final leg

        Returns:
            Dictionary with hedge scenarios
        """
        # Calculate current parlay value (legs that hit)
        current_odds = self.calculate_parlay_odds(leg_odds[:legs_hit])

        # Calculate remaining parlay odds (legs pending)
        remaining_odds = self.calculate_parlay_odds(leg_odds[legs_hit:])

        # Full parlay odds
        full_parlay_odds = self.calculate_parlay_odds(leg_odds)

        return {
            'original_bet': original_bet,
            'legs_hit': legs_hit,
            'legs_pending': legs_pending,
            'current_position_odds': current_odds,
            'remaining_odds': remaining_odds,
            'full_parlay_odds': full_parlay_odds,
            'hedge_calculation': self.calculate_hedge(
                original_bet, remaining_odds, final_hedge_odds
            )
        }

    def calculate_middle_opportunity(
        self,
        bet1_amount: float,
        bet1_line: float,
        bet1_odds: int,
        bet2_line: float,
        bet2_odds: int
    ) -> Dict:
        """
        Calculate middle opportunity (win both bets).

        Example: Bet Over 45.5, then Under 47.5
        If final total is 46 or 47, both bets win!

        Args:
            bet1_amount: Amount on first bet
            bet1_line: Line on first bet (e.g., 45.5)
            bet1_odds: Odds on first bet
            bet2_line: Line on second bet (e.g., 47.5)
            bet2_odds: Odds on second bet

        Returns:
            Middle opportunity analysis
        """
        # Calculate optimal bet2 amount
        bet1_decimal = self.american_to_decimal(bet1_odds)
        bet2_decimal = self.american_to_decimal(bet2_odds)

        # For balanced risk, match the payouts
        bet2_amount = (bet1_amount * bet1_decimal) / bet2_decimal

        # Calculate outcomes
        middle_range = abs(bet2_line - bet1_line)

        # If middle hits (both win)
        if_middle_hits = (bet1_amount * bet1_decimal - bet1_amount) + \
                        (bet2_amount * bet2_decimal - bet2_amount)

        # If bet1 wins, bet2 loses
        if_bet1_only = (bet1_amount * bet1_decimal - bet1_amount) - bet2_amount

        # If bet2 wins, bet1 loses
        if_bet2_only = (bet2_amount * bet2_decimal - bet2_amount) - bet1_amount

        # If both lose (outside both ranges)
        if_both_lose = -bet1_amount - bet2_amount

        return {
            'bet1_amount': bet1_amount,
            'bet1_line': bet1_line,
            'bet2_amount': bet2_amount,
            'bet2_line': bet2_line,
            'middle_range': middle_range,
            'middle_profit': if_middle_hits,
            'bet1_only_result': if_bet1_only,
            'bet2_only_result': if_bet2_only,
            'both_lose_result': if_both_lose,
            'recommendation': 'MIDDLE OPPORTUNITY' if middle_range > 0 else 'NO MIDDLE'
        }

    def generate_report(self, hedge_calc: HedgeCalculation) -> str:
        """Generate hedge calculator report."""
        report = []
        report.append("=" * 80)
        report.append("🎯 PARLAY HEDGE CALCULATOR")
        report.append("=" * 80)
        report.append("")

        report.append("📊 PARLAY DETAILS:")
        report.append(f"  Original Bet: ${hedge_calc.original_bet:.2f}")
        report.append(f"  Parlay Odds: {hedge_calc.parlay_odds:+d}")
        report.append(f"  Potential Payout: ${hedge_calc.potential_parlay_payout:.2f}")
        report.append(f"  Potential Profit: ${hedge_calc.potential_parlay_payout - hedge_calc.original_bet:.2f}")
        report.append("")

        report.append("🛡️ HEDGE OPTION:")
        report.append(f"  Hedge Bet Amount: ${hedge_calc.hedge_bet_amount:.2f}")
        report.append(f"  Hedge Odds: {hedge_calc.hedge_odds:+d}")
        report.append(f"  Hedge Payout if wins: ${hedge_calc.hedge_payout:.2f}")
        report.append("")

        report.append("💰 PROFIT SCENARIOS:")
        report.append(f"  If parlay wins: ${hedge_calc.if_parlay_wins:+.2f}")
        report.append(f"  If hedge wins: ${hedge_calc.if_parlay_loses:+.2f}")

        if hedge_calc.guaranteed_profit:
            report.append(f"\n  ✅ GUARANTEED PROFIT: ${hedge_calc.guaranteed_profit:.2f}")
        report.append("")

        report.append("📈 ALTERNATIVE HEDGES:")
        report.append(f"  Break-even hedge: ${hedge_calc.break_even_hedge:.2f}")
        report.append(f"  50% hedge: ${hedge_calc.hedge_bet_amount * 0.5:.2f}")
        report.append(f"  75% hedge: ${hedge_calc.hedge_bet_amount * 0.75:.2f}")
        report.append("")

        report.append("=" * 80)
        report.append(f"💡 RECOMMENDATION: {hedge_calc.recommendation}")
        report.append(f"   {hedge_calc.details}")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate optimal parlay hedge strategies"
    )
    parser.add_argument(
        "--original-bet",
        type=float,
        required=True,
        help="Original parlay bet amount ($)"
    )
    parser.add_argument(
        "--parlay-odds",
        type=int,
        required=True,
        help="Current parlay odds (American format, e.g., +600)"
    )
    parser.add_argument(
        "--hedge-odds",
        type=int,
        required=True,
        help="Hedge bet odds (American format, e.g., -110)"
    )
    parser.add_argument(
        "--goal",
        choices=['guaranteed_profit', 'maximize_profit', 'break_even'],
        default='guaranteed_profit',
        help="Hedging goal (default: guaranteed_profit)"
    )

    args = parser.parse_args()

    # Create calculator
    calc = ParlayHedgeCalculator()

    # Calculate hedge
    result = calc.calculate_hedge(
        original_bet=args.original_bet,
        parlay_odds=args.parlay_odds,
        hedge_odds=args.hedge_odds,
        hedge_goal=args.goal
    )

    # Generate report
    report = calc.generate_report(result)
    print("\n" + report)

    # Show examples
    print("\n" + "=" * 80)
    print("📚 EXAMPLE SCENARIOS")
    print("=" * 80)
    print("")
    print("Scenario 1: Let It Ride")
    result1 = calc.calculate_hedge(args.original_bet, args.parlay_odds, args.hedge_odds, 'maximize_profit')
    print(f"  Hedge: ${result1.hedge_bet_amount:.2f}")
    print(f"  If parlay wins: ${result1.if_parlay_wins:+.2f}")
    print(f"  If parlay loses: ${result1.if_parlay_loses:+.2f}")
    print("")

    print("Scenario 2: Break Even")
    result2 = calc.calculate_hedge(args.original_bet, args.parlay_odds, args.hedge_odds, 'break_even')
    print(f"  Hedge: ${result2.hedge_bet_amount:.2f}")
    print(f"  If parlay wins: ${result2.if_parlay_wins:+.2f}")
    print(f"  If parlay loses: ${result2.if_parlay_loses:+.2f}")
    print("")


if __name__ == "__main__":
    main()
