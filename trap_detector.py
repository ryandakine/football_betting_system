#!/usr/bin/env python3
"""
Trap Detector - Identifies Sharp Money vs Public Money Divergence

WHY THIS EXISTS:
Public betting % doesn't tell the full story. Each odds level has an EXPECTED
handle percentage. When actual handle diverges significantly = TRAP GAME.

THE EDGE:
- Models analyze the GAME (who's better, injuries, weather)
- This analyzes the MARKET (where is sharp money going?)
- When they diverge = opportunity

TRAP DETECTION:
Expected handle at -150 favorite: 60%
Actual handle: 85%
Divergence: +25% = 🚨 PUBLIC TRAP (Sharps hammering other side)

USAGE:
    python trap_detector.py --game "BAL @ PIT" --home-ml -150 --handle 0.85
    python trap_detector.py --week 11 --all
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TrapDetector:
    """Detects trap games using sharp vs public money divergence"""

    # Expected handle percentages by odds level
    # Source: Historical betting market data (2014-2024)
    EXPECTED_HANDLE = {
        -300: 0.75,  # Heavy favorite should get 75% of handle
        -250: 0.71,
        -200: 0.67,
        -175: 0.64,
        -150: 0.60,  # Moderate favorite gets 60%
        -130: 0.57,
        -110: 0.52,  # Small favorite gets 52%
        100: 0.50,   # Pick'em gets 50%
        110: 0.48,   # Small underdog gets 48%
        130: 0.43,
        150: 0.40,   # Moderate underdog gets 40%
        175: 0.36,
        200: 0.33,
        250: 0.29,
        300: 0.25,   # Heavy underdog gets 25%
    }

    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def calculate_trap_score(
        self,
        current_odds: int,
        handle_percentage: float,
        opening_odds: Optional[int] = None
    ) -> Tuple[int, Dict]:
        """
        Calculate trap score based on handle divergence.

        Args:
            current_odds: Current moneyline odds (e.g., -150)
            handle_percentage: Actual handle % on this side (0.0-1.0)
            opening_odds: Opening odds for line movement analysis

        Returns:
            Tuple of (trap_score, details_dict)
            trap_score range: -100 to +100
            - Negative = TRAP (fade the public)
            - Positive = Sharp consensus (ride with sharps)
            - Zero = Normal market
        """
        # Find expected handle for this odds level
        expected = self._get_expected_handle(current_odds)

        # Calculate divergence
        divergence = (handle_percentage - expected) * 100

        # Calculate base trap score
        trap_score = self._divergence_to_score(divergence)

        # Adjust for line movement (if available)
        if opening_odds is not None:
            movement_adjustment = self._analyze_line_movement(
                opening_odds,
                current_odds,
                handle_percentage
            )
            trap_score += movement_adjustment

        # Cap at -100 to +100
        trap_score = max(-100, min(100, trap_score))

        # Generate details
        details = {
            'current_odds': current_odds,
            'handle_percentage': handle_percentage,
            'expected_handle': expected,
            'divergence': divergence,
            'trap_score': trap_score,
            'severity': self._classify_severity(trap_score),
            'recommendation': self._get_recommendation(trap_score),
            'reasoning': self._explain_trap(divergence, trap_score)
        }

        if opening_odds is not None:
            details['opening_odds'] = opening_odds
            details['line_movement'] = current_odds - opening_odds

        return trap_score, details

    def _get_expected_handle(self, odds: int) -> float:
        """Get expected handle for given odds level."""
        # Find closest odds level in our lookup table
        closest_odds = min(
            self.EXPECTED_HANDLE.keys(),
            key=lambda x: abs(x - odds)
        )
        return self.EXPECTED_HANDLE[closest_odds]

    def _divergence_to_score(self, divergence: float) -> int:
        """Convert divergence percentage to trap score."""
        if divergence > 15:
            return -100  # STRONG TRAP - Fade the public
        elif divergence > 10:
            return -60   # MODERATE TRAP
        elif divergence > 5:
            return -30   # SLIGHT TRAP
        elif divergence < -15:
            return +80   # STRONG SHARP CONSENSUS
        elif divergence < -10:
            return +60   # SHARP CONSENSUS - Ride it
        elif divergence < -5:
            return +30   # SLIGHT SHARP LEAN
        else:
            return 0     # NORMAL MARKET

    def _analyze_line_movement(
        self,
        opening_odds: int,
        current_odds: int,
        handle_percentage: float
    ) -> int:
        """
        Analyze line movement for additional trap signals.

        Returns adjustment to trap score (-20 to +20).
        """
        movement = current_odds - opening_odds

        # No movement = normal
        if movement == 0:
            return 0

        # Reverse line movement (RLM) = strongest signal
        # Line moves AGAINST the handle = sharps on other side
        if movement > 0:  # Line moved toward underdog
            if handle_percentage > 0.6:  # But public on favorite
                # REVERSE LINE MOVEMENT = Strong trap
                return -20
        else:  # Line moved toward favorite
            if handle_percentage < 0.4:  # But public on underdog
                # REVERSE LINE MOVEMENT = Strong trap (other direction)
                return -20

        # Steam move (line moves WITH sharp money)
        if abs(movement) > 50:  # Big line move (0.5+ points)
            if movement > 0 and handle_percentage < 0.4:
                # Line and sharps both on underdog
                return +15
            elif movement < 0 and handle_percentage > 0.6:
                # Line and sharps both on favorite
                return +15

        return 0

    def _classify_severity(self, trap_score: int) -> str:
        """Classify trap severity."""
        if trap_score <= -80:
            return "EXTREME TRAP"
        elif trap_score <= -60:
            return "STRONG TRAP"
        elif trap_score <= -30:
            return "MODERATE TRAP"
        elif trap_score <= -10:
            return "SLIGHT TRAP"
        elif trap_score >= 80:
            return "STRONG SHARP CONSENSUS"
        elif trap_score >= 60:
            return "SHARP CONSENSUS"
        elif trap_score >= 30:
            return "SLIGHT SHARP LEAN"
        else:
            return "NORMAL MARKET"

    def _get_recommendation(self, trap_score: int) -> str:
        """Get betting recommendation based on trap score."""
        if trap_score <= -80:
            return "STRONG FADE - Bet opposite of public"
        elif trap_score <= -60:
            return "FADE PUBLIC - Consider opposite side"
        elif trap_score <= -30:
            return "Be cautious betting with public"
        elif trap_score >= 60:
            return "RIDE SHARP CONSENSUS - Bet with sharps"
        elif trap_score >= 30:
            return "Slight edge with sharps"
        else:
            return "No clear edge - standard analysis applies"

    def _explain_trap(self, divergence: float, trap_score: int) -> List[str]:
        """Generate reasoning for trap detection."""
        reasoning = []

        if divergence > 15:
            reasoning.append(
                f"Public handle {divergence:.1f}% above expected - EXTREME overload"
            )
            reasoning.append(
                "This level of divergence indicates sharps are heavily fading public"
            )
        elif divergence > 10:
            reasoning.append(
                f"Public handle {divergence:.1f}% above expected - Significant overload"
            )
        elif divergence > 5:
            reasoning.append(
                f"Public handle {divergence:.1f}% above expected - Moderate overload"
            )
        elif divergence < -10:
            reasoning.append(
                f"Sharp handle {abs(divergence):.1f}% above expected - Sharp consensus"
            )
            reasoning.append(
                "Sharps are aligned with the line - high confidence in this side"
            )
        elif divergence < -5:
            reasoning.append(
                f"Sharp handle {abs(divergence):.1f}% above expected - Slight sharp lean"
            )
        else:
            reasoning.append(
                "Handle distribution matches expected pattern - normal market"
            )

        return reasoning

    def detect_trap_patterns(
        self,
        game: str,
        home_ml: int,
        away_ml: int,
        home_handle: float,
        away_handle: float,
        opening_home_ml: Optional[int] = None,
        opening_away_ml: Optional[int] = None
    ) -> Dict:
        """
        Detect all trap patterns for a game.

        Args:
            game: Game identifier (e.g., "BAL @ PIT")
            home_ml: Current home moneyline odds
            away_ml: Current away moneyline odds
            home_handle: Handle % on home team (0.0-1.0)
            away_handle: Handle % on away team (0.0-1.0)
            opening_home_ml: Opening home ML (optional)
            opening_away_ml: Opening away ML (optional)

        Returns:
            Dict with trap analysis for both sides
        """
        # Analyze home team
        home_score, home_details = self.calculate_trap_score(
            home_ml,
            home_handle,
            opening_home_ml
        )

        # Analyze away team
        away_score, away_details = self.calculate_trap_score(
            away_ml,
            away_handle,
            opening_away_ml
        )

        # Determine overall trap signal
        if abs(home_score) > abs(away_score):
            primary_trap = 'home' if home_score < 0 else 'away'
            trap_strength = abs(home_score)
        else:
            primary_trap = 'away' if away_score < 0 else 'home'
            trap_strength = abs(away_score)

        return {
            'game': game,
            'home_analysis': home_details,
            'away_analysis': away_details,
            'primary_trap_side': primary_trap,
            'trap_strength': trap_strength,
            'recommendation': self._get_game_recommendation(
                home_score,
                away_score,
                game
            ),
            'timestamp': datetime.now().isoformat()
        }

    def _get_game_recommendation(
        self,
        home_score: int,
        away_score: int,
        game: str
    ) -> str:
        """Get overall game recommendation."""
        parts = game.split('@')
        away_team = parts[0].strip() if len(parts) == 2 else "AWAY"
        home_team = parts[1].strip() if len(parts) == 2 else "HOME"

        if home_score <= -60:
            return f"FADE {home_team} - Bet {away_team} (trap detected on home)"
        elif away_score <= -60:
            return f"FADE {away_team} - Bet {home_team} (trap detected on away)"
        elif home_score >= 60:
            return f"BET {home_team} - Sharp consensus on home"
        elif away_score >= 60:
            return f"BET {away_team} - Sharp consensus on away"
        else:
            return "No strong trap signal - rely on standard analysis"


def main():
    parser = argparse.ArgumentParser(
        description="Detect trap games using sharp vs public money divergence"
    )
    parser.add_argument(
        "--game",
        help="Game to analyze (e.g., 'BAL @ PIT')"
    )
    parser.add_argument(
        "--home-ml",
        type=int,
        help="Current home moneyline odds (e.g., -150)"
    )
    parser.add_argument(
        "--away-ml",
        type=int,
        help="Current away moneyline odds (e.g., +130)"
    )
    parser.add_argument(
        "--home-handle",
        type=float,
        help="Handle % on home team (0.0-1.0, e.g., 0.85 for 85%%)"
    )
    parser.add_argument(
        "--away-handle",
        type=float,
        help="Handle % on away team (0.0-1.0)"
    )
    parser.add_argument(
        "--opening-home-ml",
        type=int,
        help="Opening home moneyline (optional, for line movement analysis)"
    )
    parser.add_argument(
        "--opening-away-ml",
        type=int,
        help="Opening away moneyline (optional)"
    )

    args = parser.parse_args()

    if not all([args.game, args.home_ml, args.away_ml, args.home_handle, args.away_handle]):
        parser.print_help()
        print("\n💡 EXAMPLE:")
        print("   python trap_detector.py --game 'BAL @ PIT' \\")
        print("       --home-ml -150 --away-ml +130 \\")
        print("       --home-handle 0.85 --away-handle 0.15")
        return

    detector = TrapDetector()

    result = detector.detect_trap_patterns(
        game=args.game,
        home_ml=args.home_ml,
        away_ml=args.away_ml,
        home_handle=args.home_handle,
        away_handle=args.away_handle,
        opening_home_ml=args.opening_home_ml,
        opening_away_ml=args.opening_away_ml
    )

    # Print results
    print("=" * 70)
    print(f"🎯 TRAP DETECTION: {result['game']}")
    print("=" * 70)
    print()

    print("HOME TEAM ANALYSIS:")
    home = result['home_analysis']
    print(f"   Odds: {home['current_odds']}")
    print(f"   Handle: {home['handle_percentage']:.1%} (Expected: {home['expected_handle']:.1%})")
    print(f"   Divergence: {home['divergence']:+.1f}%")
    print(f"   Trap Score: {home['trap_score']}")
    print(f"   Severity: {home['severity']}")
    print()

    print("AWAY TEAM ANALYSIS:")
    away = result['away_analysis']
    print(f"   Odds: {away['current_odds']}")
    print(f"   Handle: {away['handle_percentage']:.1%} (Expected: {away['expected_handle']:.1%})")
    print(f"   Divergence: {away['divergence']:+.1f}%")
    print(f"   Trap Score: {away['trap_score']}")
    print(f"   Severity: {away['severity']}")
    print()

    print("=" * 70)
    print("🎯 RECOMMENDATION:")
    print(f"   {result['recommendation']}")
    print("=" * 70)

    # Save result
    output_file = detector.data_dir / f"trap_analysis_{args.game.replace(' ', '_').replace('@', 'at')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved to {output_file}")


if __name__ == "__main__":
    main()
