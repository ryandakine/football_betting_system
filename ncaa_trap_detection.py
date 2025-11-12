#!/usr/bin/env python3
"""
NCAA Trap Detection - Sharp Money vs Public Money
==================================================

PRINCIPLE: "Smarter than the data" - detect when sharp money
           diverges from public betting patterns

WHAT THIS DETECTS:
1. Expected handle % based on odds (from market efficiency research)
2. Actual handle % from betting data sources
3. Divergence = trap signal (sharps vs public)

SHARP MONEY SIGNALS:
- Line moves AGAINST public (reverse line movement)
- Handle % way above expected (public overload = trap)
- Handle % way below expected (sharp consensus)

USAGE:
    from ncaa_trap_detection import NCAATrapDetector

    detector = NCAATrapDetector()
    trap_signal = detector.analyze_game(
        home_ml=-150,
        actual_handle=0.85,  # 85% on favorite
        line_opened=-130,
        line_current=-150
    )

    if trap_signal['trap_score'] < -60:
        print("STRONG TRAP - Fade the public!")

DATA SOURCES:
- Action Network (best - has money % and bet %)
- BetOnline (public betting %)
- VegasInsider (line movement)

Based on market efficiency research showing expected handle
distribution at different odds levels.
"""

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class TrapStrength(Enum):
    """Trap signal strength"""
    STRONG_TRAP = -100      # Fade public aggressively
    MODERATE_TRAP = -60     # Fade public cautiously
    SLIGHT_TRAP = -30       # Lean fade
    NORMAL = 0              # No trap signal
    SLIGHT_SHARP = 30       # Sharps slightly favoring
    SHARP_CONSENSUS = 60    # Sharps heavily favoring
    EXTREME_SHARP = 100     # Sharps all-in


@dataclass
class TrapSignal:
    """Trap detection result"""
    trap_score: int  # -100 to +100
    strength: TrapStrength
    signal: str
    sharp_side: str  # 'favorite' or 'underdog'
    divergence: float  # % points divergence
    expected_handle: float
    actual_handle: float
    confidence: float  # 0-1
    reverse_line_movement: bool
    reasoning: str


class NCAATrapDetector:
    """
    Detects trap games where sharp money diverges from public

    Based on expected handle % at different odds levels.
    When actual handle significantly diverges from expected,
    that's a signal of sharp money on the other side.
    """

    # Expected handle % for different moneyline odds
    # Based on market efficiency research
    EXPECTED_HANDLE = {
        -300: 0.75,  # Heavy favorite should get 75% of money
        -250: 0.71,
        -200: 0.67,
        -175: 0.64,
        -150: 0.60,
        -130: 0.57,
        -110: 0.52,
        -105: 0.51,
        100: 0.50,   # Pick'em
        105: 0.49,
        110: 0.48,
        130: 0.43,
        150: 0.40,
        175: 0.36,
        200: 0.33,
        250: 0.29,
        300: 0.25,   # Heavy underdog gets 25% of money
    }

    # Divergence thresholds
    STRONG_TRAP_THRESHOLD = 0.15      # 15%+ divergence
    MODERATE_TRAP_THRESHOLD = 0.10    # 10%+ divergence
    SLIGHT_TRAP_THRESHOLD = 0.05      # 5%+ divergence

    SHARP_CONSENSUS_THRESHOLD = -0.10  # 10%+ below expected
    EXTREME_SHARP_THRESHOLD = -0.15    # 15%+ below expected

    def __init__(self):
        pass

    def analyze_game(
        self,
        home_ml: float,
        actual_handle: float,
        line_opened: Optional[float] = None,
        line_current: Optional[float] = None,
        game_info: Optional[Dict] = None
    ) -> TrapSignal:
        """
        Analyze game for trap signals

        Args:
            home_ml: Current home team moneyline (e.g., -150)
            actual_handle: Actual handle % on favorite (0-1)
            line_opened: Opening moneyline (optional)
            line_current: Current moneyline (optional)
            game_info: Additional game context (optional)

        Returns:
            TrapSignal with trap score and reasoning
        """

        # Step 1: Get expected handle for this line
        expected_handle = self._get_expected_handle(home_ml)

        # Step 2: Calculate divergence
        divergence = actual_handle - expected_handle

        # Step 3: Check for reverse line movement
        reverse_line_movement = False
        if line_opened and line_current:
            reverse_line_movement = self._detect_reverse_line_movement(
                line_opened, line_current, actual_handle
            )

        # Step 4: Calculate trap score
        trap_score, strength = self._calculate_trap_score(
            divergence, reverse_line_movement
        )

        # Step 5: Determine sharp side
        if trap_score < 0:
            # Negative trap score = public overloaded on favorite
            # Sharps on underdog
            sharp_side = 'underdog'
        elif trap_score > 0:
            # Positive score = sharp consensus on favorite
            sharp_side = 'favorite'
        else:
            sharp_side = 'none'

        # Step 6: Generate signal and reasoning
        signal, reasoning = self._generate_reasoning(
            trap_score, divergence, expected_handle, actual_handle,
            reverse_line_movement, sharp_side, game_info
        )

        # Step 7: Calculate confidence
        confidence = self._calculate_confidence(
            divergence, reverse_line_movement
        )

        return TrapSignal(
            trap_score=trap_score,
            strength=strength,
            signal=signal,
            sharp_side=sharp_side,
            divergence=divergence,
            expected_handle=expected_handle,
            actual_handle=actual_handle,
            confidence=confidence,
            reverse_line_movement=reverse_line_movement,
            reasoning=reasoning
        )

    def _get_expected_handle(self, moneyline: float) -> float:
        """
        Get expected handle % for given moneyline

        Interpolates between known values for accuracy
        """
        # Handle pick'em
        if abs(moneyline - 100) < 10:
            return 0.50

        # Find closest odds levels
        sorted_odds = sorted(self.EXPECTED_HANDLE.keys())

        # If outside range, use boundary values
        if moneyline <= sorted_odds[0]:
            return self.EXPECTED_HANDLE[sorted_odds[0]]
        if moneyline >= sorted_odds[-1]:
            return self.EXPECTED_HANDLE[sorted_odds[-1]]

        # Interpolate between two closest odds levels
        lower = max([o for o in sorted_odds if o <= moneyline])
        upper = min([o for o in sorted_odds if o >= moneyline])

        if lower == upper:
            return self.EXPECTED_HANDLE[lower]

        # Linear interpolation
        lower_handle = self.EXPECTED_HANDLE[lower]
        upper_handle = self.EXPECTED_HANDLE[upper]

        weight = (moneyline - lower) / (upper - lower)
        expected = lower_handle + weight * (upper_handle - lower_handle)

        return expected

    def _detect_reverse_line_movement(
        self,
        line_opened: float,
        line_current: float,
        actual_handle: float
    ) -> bool:
        """
        Detect reverse line movement (RLM)

        RLM = line moves AGAINST public betting
        Strong indicator of sharp money
        """
        # Determine which side has heavy public action
        public_on_favorite = actual_handle > 0.60

        # Did line move toward or away from public side?
        # Negative line = favorite
        # If favorite, more negative = stronger

        line_moved_toward_favorite = line_current < line_opened

        if public_on_favorite and not line_moved_toward_favorite:
            # Public on favorite, but line moved toward underdog
            # = Reverse line movement (sharps on underdog)
            return True

        if not public_on_favorite and line_moved_toward_favorite:
            # Public on underdog, but line moved toward favorite
            # = Reverse line movement (sharps on favorite)
            return True

        return False

    def _calculate_trap_score(
        self,
        divergence: float,
        reverse_line_movement: bool
    ) -> Tuple[int, TrapStrength]:
        """
        Calculate trap score from -100 to +100

        Negative = trap (fade public)
        Positive = sharp consensus (ride with sharps)
        """

        # Base score from divergence
        if divergence >= self.STRONG_TRAP_THRESHOLD:
            score = -100
            strength = TrapStrength.STRONG_TRAP
        elif divergence >= self.MODERATE_TRAP_THRESHOLD:
            score = -60
            strength = TrapStrength.MODERATE_TRAP
        elif divergence >= self.SLIGHT_TRAP_THRESHOLD:
            score = -30
            strength = TrapStrength.SLIGHT_TRAP
        elif divergence <= self.EXTREME_SHARP_THRESHOLD:
            score = 100
            strength = TrapStrength.EXTREME_SHARP
        elif divergence <= self.SHARP_CONSENSUS_THRESHOLD:
            score = 60
            strength = TrapStrength.SHARP_CONSENSUS
        elif divergence <= -self.SLIGHT_TRAP_THRESHOLD:
            score = 30
            strength = TrapStrength.SLIGHT_SHARP
        else:
            score = 0
            strength = TrapStrength.NORMAL

        # Boost score if reverse line movement detected
        if reverse_line_movement:
            if score < 0:
                score = max(-100, score - 20)  # Stronger trap
            elif score > 0:
                score = min(100, score + 20)   # Stronger sharp consensus

        # Re-classify strength after adjustment
        if abs(score) >= 80:
            strength = TrapStrength.STRONG_TRAP if score < 0 else TrapStrength.EXTREME_SHARP
        elif abs(score) >= 50:
            strength = TrapStrength.MODERATE_TRAP if score < 0 else TrapStrength.SHARP_CONSENSUS
        elif abs(score) >= 25:
            strength = TrapStrength.SLIGHT_TRAP if score < 0 else TrapStrength.SLIGHT_SHARP
        else:
            strength = TrapStrength.NORMAL

        return score, strength

    def _generate_reasoning(
        self,
        trap_score: int,
        divergence: float,
        expected_handle: float,
        actual_handle: float,
        reverse_line_movement: bool,
        sharp_side: str,
        game_info: Optional[Dict]
    ) -> Tuple[str, str]:
        """Generate human-readable signal and reasoning"""

        if trap_score <= -80:
            signal = "🚨 STRONG TRAP - FADE PUBLIC"
            reasoning = (
                f"Public overload detected: {actual_handle:.1%} on favorite "
                f"(expected {expected_handle:.1%}). Divergence of {divergence:+.1%}. "
            )
            if reverse_line_movement:
                reasoning += "Reverse line movement confirms sharp action on underdog. "
            reasoning += f"RECOMMENDATION: Bet {sharp_side}."

        elif trap_score <= -50:
            signal = "⚠️  MODERATE TRAP - Lean fade"
            reasoning = (
                f"Above-average public betting: {actual_handle:.1%} "
                f"(expected {expected_handle:.1%}). "
            )
            if reverse_line_movement:
                reasoning += "Line moving against public suggests sharp money. "
            reasoning += f"Consider {sharp_side}."

        elif trap_score <= -25:
            signal = "📊 SLIGHT TRAP - Monitor"
            reasoning = (
                f"Slightly elevated public betting: {actual_handle:.1%} "
                f"(expected {expected_handle:.1%}). "
                f"Small edge on {sharp_side}."
            )

        elif trap_score >= 80:
            signal = "💎 EXTREME SHARP CONSENSUS"
            reasoning = (
                f"Sharp money heavily favoring {sharp_side}. "
                f"Only {actual_handle:.1%} of handle (expected {expected_handle:.1%}). "
            )
            if reverse_line_movement:
                reasoning += "Line movement confirms sharp action. "
            reasoning += "STRONG BET."

        elif trap_score >= 50:
            signal = "✅ SHARP CONSENSUS"
            reasoning = (
                f"Sharp money on {sharp_side}. "
                f"Handle at {actual_handle:.1%} vs expected {expected_handle:.1%}. "
                f"Good bet opportunity."
            )

        elif trap_score >= 25:
            signal = "📈 SLIGHT SHARP - Lean with"
            reasoning = (
                f"Sharps slightly favoring {sharp_side}. "
                f"Handle below expected. Minor edge."
            )

        else:
            signal = "⚪ NORMAL MARKET"
            reasoning = (
                f"Handle at {actual_handle:.1%} matches expected {expected_handle:.1%}. "
                f"No trap signal detected."
            )

        # Add game context if available
        if game_info:
            if game_info.get('is_maction'):
                reasoning += " [MACtion game - softer lines, trap signals more reliable]"
            if game_info.get('big_name_school'):
                reasoning += " [Big name school - public bias likely]"

        return signal, reasoning

    def _calculate_confidence(
        self,
        divergence: float,
        reverse_line_movement: bool
    ) -> float:
        """
        Calculate confidence in trap signal (0-1)

        Higher divergence + RLM = higher confidence
        """
        # Base confidence from divergence magnitude
        base_confidence = min(abs(divergence) / 0.20, 1.0)

        # Boost if reverse line movement
        if reverse_line_movement:
            base_confidence = min(base_confidence * 1.25, 1.0)

        return base_confidence

    def get_recommended_bet_adjustment(self, trap_signal: TrapSignal) -> Dict:
        """
        Get betting adjustment recommendation based on trap signal

        Returns: Adjustment to apply to model predictions
        """
        if trap_signal.trap_score <= -80:
            return {
                'action': 'FADE',
                'confidence_boost': 0.15,  # +15% confidence
                'recommended_side': trap_signal.sharp_side,
                'reasoning': 'Strong trap - sharp money on other side'
            }

        elif trap_signal.trap_score <= -50:
            return {
                'action': 'LEAN_FADE',
                'confidence_boost': 0.08,  # +8% confidence
                'recommended_side': trap_signal.sharp_side,
                'reasoning': 'Moderate trap - consider fading'
            }

        elif trap_signal.trap_score >= 80:
            return {
                'action': 'RIDE_SHARPS',
                'confidence_boost': 0.15,  # +15% confidence
                'recommended_side': trap_signal.sharp_side,
                'reasoning': 'Extreme sharp consensus - ride with them'
            }

        elif trap_signal.trap_score >= 50:
            return {
                'action': 'LEAN_SHARPS',
                'confidence_boost': 0.08,  # +8% confidence
                'recommended_side': trap_signal.sharp_side,
                'reasoning': 'Sharp consensus - good spot'
            }

        else:
            return {
                'action': 'NONE',
                'confidence_boost': 0.0,
                'recommended_side': 'none',
                'reasoning': 'Normal market - no adjustment'
            }


def main():
    """Demo trap detection"""
    detector = NCAATrapDetector()

    print("NCAA Trap Detection Demo\n")
    print("=" * 80)

    # Example 1: Strong trap
    print("\n📌 Example 1: Toledo -150 vs Bowling Green")
    print("-" * 80)
    trap = detector.analyze_game(
        home_ml=-150,
        actual_handle=0.85,  # 85% on Toledo (way above expected 60%)
        line_opened=-130,
        line_current=-150,   # Moved toward favorite despite heavy public
        game_info={'is_maction': True}
    )

    print(f"Signal: {trap.signal}")
    print(f"Trap Score: {trap.trap_score}")
    print(f"Expected handle: {trap.expected_handle:.1%}")
    print(f"Actual handle: {trap.actual_handle:.1%}")
    print(f"Divergence: {trap.divergence:+.1%}")
    print(f"Sharp side: {trap.sharp_side}")
    print(f"Confidence: {trap.confidence:.1%}")
    print(f"Reasoning: {trap.reasoning}")

    adjustment = detector.get_recommended_bet_adjustment(trap)
    print(f"\n💡 Betting Adjustment:")
    print(f"   Action: {adjustment['action']}")
    print(f"   Confidence boost: +{adjustment['confidence_boost']:.1%}")
    print(f"   Bet on: {adjustment['recommended_side']}")

    # Example 2: Normal market
    print("\n\n📌 Example 2: Alabama -200 vs Auburn")
    print("-" * 80)
    trap = detector.analyze_game(
        home_ml=-200,
        actual_handle=0.68,  # Expected 67%, actual 68% (normal)
        line_opened=-195,
        line_current=-200,
        game_info={'big_name_school': True}
    )

    print(f"Signal: {trap.signal}")
    print(f"Trap Score: {trap.trap_score}")
    print(f"Reasoning: {trap.reasoning}")

    # Example 3: Sharp consensus
    print("\n\n📌 Example 3: Kent State +150 vs Ohio")
    print("-" * 80)
    trap = detector.analyze_game(
        home_ml=150,
        actual_handle=0.25,  # Expected 40% on underdog, only 25% (sharps on favorite)
        line_opened=140,
        line_current=150,   # Moved toward underdog
    )

    print(f"Signal: {trap.signal}")
    print(f"Trap Score: {trap.trap_score}")
    print(f"Sharp side: {trap.sharp_side}")
    print(f"Reasoning: {trap.reasoning}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
