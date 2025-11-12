#!/usr/bin/env python3
"""
NCAA Key Number Analyzer
=========================

Analyzes key numbers in football betting (3, 7, 10, etc.)

WHY KEY NUMBERS MATTER:
- 3: Most common margin (field goal) - 15% of games
- 7: Touchdown margin - 9% of games
- 10: TD + FG - 5% of games
- 6, 4, 14: Also significant

Getting right side of key number = Fewer bad beats

Example:
  Bet Toledo -2.5: ✅ Good (3 pushes, you don't lose)
  Bet Toledo -3.5: ❌ Bad (3 loses for you)

  Difference: ~3% win rate (worth $300/year on $10k bets)

USAGE:
    from ncaa_key_numbers import KeyNumberAnalyzer

    analyzer = KeyNumberAnalyzer()

    # Analyze a spread
    analysis = analyzer.analyze_spread(-3.0, 'Toledo')
    print(analysis['recommendation'])

    # Compare two lines
    comparison = analyzer.compare_lines(-2.5, -3.5)
    print(f"Better line: {comparison['better_line']}")
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KeyNumberAnalysis:
    """Analysis of spread relative to key numbers"""
    spread: float
    team: str
    is_favorite: bool
    nearest_key_number: int
    distance_to_key: float
    on_good_side: bool
    recommendation: str
    warning: Optional[str]
    ev_adjustment: float  # Estimated EV adjustment vs crossing key


class KeyNumberAnalyzer:
    """
    Analyze spreads relative to key numbers

    Helps avoid bad beats and find value
    """

    # Key numbers in football (most common margins)
    KEY_NUMBERS = {
        3: {
            'frequency': 0.15,  # 15% of games
            'description': 'Field goal',
            'importance': 'CRITICAL'
        },
        7: {
            'frequency': 0.09,  # 9% of games
            'description': 'Touchdown',
            'importance': 'CRITICAL'
        },
        10: {
            'frequency': 0.05,  # 5% of games
            'description': 'TD + FG',
            'importance': 'HIGH'
        },
        6: {
            'frequency': 0.04,  # 4% of games
            'description': 'Two field goals',
            'importance': 'MEDIUM'
        },
        4: {
            'frequency': 0.03,  # 3% of games
            'description': 'TD with missed PAT',
            'importance': 'MEDIUM'
        },
        14: {
            'frequency': 0.04,  # 4% of games
            'description': 'Two touchdowns',
            'importance': 'MEDIUM'
        }
    }

    def __init__(self):
        pass

    def analyze_spread(
        self,
        spread: float,
        team: str,
        consider_half_point: bool = True
    ) -> KeyNumberAnalysis:
        """
        Analyze spread relative to key numbers

        Args:
            spread: Spread value (negative = favorite)
            team: Team name
            consider_half_point: If True, prefer half-point lines

        Returns:
            KeyNumberAnalysis with recommendation
        """

        is_favorite = spread < 0
        abs_spread = abs(spread)

        # Find nearest key number
        nearest_key = min(
            self.KEY_NUMBERS.keys(),
            key=lambda k: abs(abs_spread - k)
        )

        distance = abs_spread - nearest_key

        # Determine if on good side of key number
        if is_favorite:
            # Favorite: Want to give fewer points
            # -2.5 is better than -3.5 (good side of 3)
            on_good_side = abs_spread < nearest_key
        else:
            # Underdog: Want to get more points
            # +3.5 is better than +2.5 (good side of 3)
            on_good_side = abs_spread > nearest_key

        # Generate recommendation
        recommendation = self._generate_recommendation(
            spread,
            nearest_key,
            distance,
            on_good_side,
            is_favorite
        )

        # Generate warning if applicable
        warning = self._generate_warning(
            spread,
            nearest_key,
            distance,
            on_good_side,
            is_favorite
        )

        # Estimate EV adjustment
        ev_adjustment = self._estimate_ev_adjustment(
            nearest_key,
            distance,
            on_good_side
        )

        return KeyNumberAnalysis(
            spread=spread,
            team=team,
            is_favorite=is_favorite,
            nearest_key_number=nearest_key,
            distance_to_key=distance,
            on_good_side=on_good_side,
            recommendation=recommendation,
            warning=warning,
            ev_adjustment=ev_adjustment
        )

    def _generate_recommendation(
        self,
        spread: float,
        nearest_key: int,
        distance: float,
        on_good_side: bool,
        is_favorite: bool
    ) -> str:
        """Generate betting recommendation"""

        abs_spread = abs(spread)
        key_info = self.KEY_NUMBERS[nearest_key]

        if abs(distance) < 0.1:
            # Exactly on key number
            return (
                f"⚠️  EXACTLY on key number {nearest_key} - AVOID if possible!\n"
                f"   {key_info['frequency']:.0%} of games land here ({key_info['description']})\n"
                f"   High push risk - look for half-point line"
            )

        elif abs(distance) <= 0.5:
            # Within half point of key number
            if on_good_side:
                return (
                    f"✅ GOOD SIDE of key number {nearest_key}!\n"
                    f"   Spread {spread:+.1f} avoids {key_info['frequency']:.0%} push risk\n"
                    f"   Worth ~{self._estimate_ev_adjustment(nearest_key, distance, True):.1f}% EV boost"
                )
            else:
                return (
                    f"❌ BAD SIDE of key number {nearest_key}!\n"
                    f"   Spread {spread:+.1f} crosses critical margin\n"
                    f"   Consider alternate line if available"
                )

        elif abs(distance) <= 1.0:
            # Within 1 point of key number
            if on_good_side:
                return (
                    f"👍 Safe distance from key number {nearest_key}\n"
                    f"   Spread {spread:+.1f} provides cushion"
                )
            else:
                return (
                    f"⚠️  Close to key number {nearest_key}\n"
                    f"   Spread {spread:+.1f} - watch for line movement"
                )

        else:
            # Far from key numbers
            return (
                f"😐 No key number concerns\n"
                f"   Spread {spread:+.1f} is {abs(distance):.1f} points from {nearest_key}"
            )

    def _generate_warning(
        self,
        spread: float,
        nearest_key: int,
        distance: float,
        on_good_side: bool,
        is_favorite: bool
    ) -> Optional[str]:
        """Generate warning if crossing key number"""

        if not on_good_side and abs(distance) <= 0.5:
            key_info = self.KEY_NUMBERS[nearest_key]
            if key_info['importance'] == 'CRITICAL':
                return (
                    f"🚨 CRITICAL: Crossing key number {nearest_key}!\n"
                    f"   This margin occurs in {key_info['frequency']:.0%} of games\n"
                    f"   Strong recommendation: Find alternate line"
                )

        return None

    def _estimate_ev_adjustment(
        self,
        nearest_key: int,
        distance: float,
        on_good_side: bool
    ) -> float:
        """Estimate EV adjustment from key number positioning"""

        key_info = self.KEY_NUMBERS[nearest_key]
        frequency = key_info['frequency']

        if abs(distance) > 1.0:
            # Far from key number - minimal impact
            return 0

        # Being on good side of key = getting value equal to ~frequency/2
        # Being on bad side = losing value
        if on_good_side:
            # Half-point better = ~half the frequency advantage
            return (frequency / 2) * 100
        else:
            return -(frequency / 2) * 100

    def compare_lines(
        self,
        line1: float,
        line2: float,
        team: str = "Team"
    ) -> Dict:
        """
        Compare two lines and determine which is better

        Args:
            line1: First line
            line2: Second line
            team: Team name

        Returns:
            Comparison with better line and reasoning
        """

        analysis1 = self.analyze_spread(line1, team)
        analysis2 = self.analyze_spread(line2, team)

        # Determine better line based on:
        # 1. Key number positioning
        # 2. EV adjustment
        # 3. Raw value

        if analysis1.on_good_side and not analysis2.on_good_side:
            better_line = line1
            reason = f"Line {line1:+.1f} on good side of key {analysis1.nearest_key_number}"
        elif analysis2.on_good_side and not analysis1.on_good_side:
            better_line = line2
            reason = f"Line {line2:+.1f} on good side of key {analysis2.nearest_key_number}"
        elif analysis1.ev_adjustment > analysis2.ev_adjustment:
            better_line = line1
            reason = f"Line {line1:+.1f} has {analysis1.ev_adjustment:.1f}% EV edge"
        elif analysis2.ev_adjustment > analysis1.ev_adjustment:
            better_line = line2
            reason = f"Line {line2:+.1f} has {analysis2.ev_adjustment:.1f}% EV edge"
        else:
            # Default to better raw value
            if abs(line1) < abs(line2):
                better_line = line1
                reason = f"Line {line1:+.1f} gives {abs(line2 - line1):.1f} points better value"
            else:
                better_line = line2
                reason = f"Line {line2:+.1f} gives {abs(line2 - line1):.1f} points better value"

        return {
            'better_line': better_line,
            'reason': reason,
            'line1_analysis': analysis1,
            'line2_analysis': analysis2,
            'ev_difference': abs(analysis1.ev_adjustment - analysis2.ev_adjustment)
        }

    def print_analysis(self, spread: float, team: str):
        """Print detailed analysis of spread"""

        analysis = self.analyze_spread(spread, team)

        print(f"\n{'='*80}")
        print(f"🔢 KEY NUMBER ANALYSIS")
        print(f"{'='*80}\n")

        print(f"Team: {team}")
        print(f"Spread: {spread:+.1f} ({'Favorite' if analysis.is_favorite else 'Underdog'})")
        print(f"Nearest Key: {analysis.nearest_key_number} ({self.KEY_NUMBERS[analysis.nearest_key_number]['description']})")
        print(f"Distance: {analysis.distance_to_key:+.1f} points")
        print()

        print(f"📊 RECOMMENDATION:")
        print(f"{analysis.recommendation}")
        print()

        if analysis.warning:
            print(f"⚠️  WARNING:")
            print(f"{analysis.warning}")
            print()

        if analysis.ev_adjustment != 0:
            print(f"💰 EV Impact: {analysis.ev_adjustment:+.1f}%")
            print()

        print(f"{'='*80}\n")


def main():
    """Demo key number analyzer"""

    print("NCAA Key Number Analyzer Demo\n")

    analyzer = KeyNumberAnalyzer()

    # Example 1: Good side of 3
    print("Example 1: Good side of key number 3")
    analyzer.print_analysis(-2.5, "Toledo")

    # Example 2: Bad side of 3
    print("\nExample 2: Bad side of key number 3")
    analyzer.print_analysis(-3.5, "Toledo")

    # Example 3: Exactly on 3
    print("\nExample 3: Exactly on key number 3")
    analyzer.print_analysis(-3.0, "Alabama")

    # Example 4: Good side of 7
    print("\nExample 4: Good side of key number 7")
    analyzer.print_analysis(+7.5, "Ohio")

    # Example 5: Compare two lines
    print("\nExample 5: Line Comparison")
    print("="*80)
    comparison = analyzer.compare_lines(-2.5, -3.5, "Toledo")
    print(f"Better line: {comparison['better_line']:+.1f}")
    print(f"Reason: {comparison['reason']}")
    print(f"EV Difference: {comparison['ev_difference']:.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()
