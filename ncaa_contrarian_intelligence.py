#!/usr/bin/env python3
"""
NCAA Contrarian Intelligence - Fade The Public
===============================================

PRINCIPLE: System-Level Detection (Not Agent Memory)
Investment → System: Contrarian signals detected automatically, not manually checked

WHY THIS EXISTS:
- Public bets heavily on home favorites and big name schools
- 70%+ public on one side = sharp opportunity
- Reverse line movement = sharp money detected
- MACtion games especially vulnerable to public bias

INTEGRATION:
- Auto-runs in ncaa_daily_predictions.py
- Alerts in bet placement workflow
- Integrated into Tuesday MACtion skill

Based on NFL contrarian system - adapted for college football
"""

import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass


@dataclass
class ContrarianSignal:
    """Contrarian betting signal"""
    strength: int  # 0-5 stars
    recommendation: str  # "FADE HOME", "FADE AWAY", "NO SIGNAL"
    public_percentage: float
    line_movement: float  # Opening - Current
    sharp_money_detected: bool
    reasons: List[str]

    def __str__(self):
        stars = "⭐" * self.strength
        return f"{stars} ({self.strength}/5) - {self.recommendation}"


class NCAAContrarianIntelligence:
    """
    Detects contrarian betting opportunities in NCAA football

    PRINCIPLE: Automated Constraints
    System automatically checks for public bias - agent can't forget
    """

    def __init__(self, odds_api_key: str):
        self.odds_api_key = odds_api_key
        self.base_url = "https://api.the-odds-api.com/v4"

        # NCAA-specific thresholds
        self.PUBLIC_HEAVY_THRESHOLD = 0.65  # 65% (lower than NFL's 70%)
        self.EXTREME_PUBLIC_THRESHOLD = 0.75  # 75%
        self.SHARP_LINE_MOVE_THRESHOLD = 1.0  # 1 point movement

    def analyze_game(
        self,
        game: Dict,
        home_team: str,
        away_team: str
    ) -> ContrarianSignal:
        """
        Analyze single game for contrarian signals

        WHY: System detects bias automatically - no manual checking

        Returns: ContrarianSignal with strength 0-5
        """

        reasons = []
        strength = 0
        recommendation = "NO SIGNAL"

        # Get line movement
        opening_spread = game.get('opening_spread')
        current_spread = game.get('current_spread')

        if opening_spread is not None and current_spread is not None:
            line_movement = abs(opening_spread - current_spread)
        else:
            line_movement = 0.0

        # Estimate public betting percentage (from market indicators)
        public_pct = self._estimate_public_percentage(game, home_team, away_team)

        # SIGNAL 1: Public overload on home team
        if public_pct > self.EXTREME_PUBLIC_THRESHOLD:
            strength += 2
            reasons.append(f"Public extremely heavy on home ({public_pct:.0%})")
            recommendation = "FADE HOME - Take AWAY team"
        elif public_pct > self.PUBLIC_HEAVY_THRESHOLD:
            strength += 1
            reasons.append(f"Public heavy on home ({public_pct:.0%})")
            recommendation = "FADE HOME - Take AWAY team"

        # SIGNAL 2: Public overload on away team (less common in NCAA)
        elif public_pct < (1 - self.EXTREME_PUBLIC_THRESHOLD):
            strength += 2
            reasons.append(f"Public extremely heavy on away ({(1-public_pct):.0%})")
            recommendation = "FADE AWAY - Take HOME team"
        elif public_pct < (1 - self.PUBLIC_HEAVY_THRESHOLD):
            strength += 1
            reasons.append(f"Public heavy on away ({(1-public_pct):.0%})")
            recommendation = "FADE AWAY - Take HOME team"

        # SIGNAL 3: Sharp line movement
        sharp_money_detected = False
        if line_movement >= self.SHARP_LINE_MOVE_THRESHOLD:
            # Check for reverse line movement (most powerful signal)
            if opening_spread is not None and current_spread is not None:
                # Line moved toward less popular side = sharp money
                if public_pct > 0.65 and current_spread < opening_spread:
                    # Public on home, line moved toward away
                    strength += 2
                    sharp_money_detected = True
                    reasons.append("Sharp money detected on away (reverse movement)")
                elif public_pct < 0.35 and current_spread > opening_spread:
                    # Public on away, line moved toward home
                    strength += 2
                    sharp_money_detected = True
                    reasons.append("Sharp money detected on home (reverse movement)")
                else:
                    # Regular line movement (with public)
                    strength += 1
                    reasons.append(f"Line movement: {line_movement:.1f} points")

        # SIGNAL 4: Big name school bias (Alabama, Ohio State, etc.)
        big_name_schools = [
            'Alabama', 'Ohio State', 'Georgia', 'Michigan',
            'Notre Dame', 'Texas', 'USC', 'Oklahoma'
        ]

        if any(school in home_team for school in big_name_schools):
            if public_pct > 0.70:
                strength += 1
                reasons.append("Big name school + heavy public = fade opportunity")

        # SIGNAL 5: MACtion/Tuesday games (softer lines)
        if game.get('day_of_week') in ['Tuesday', 'Wednesday']:
            if game.get('conference') == 'MAC':
                strength += 1
                reasons.append("MACtion game - public often overreacts")

        # Cap at 5 stars
        strength = min(strength, 5)

        return ContrarianSignal(
            strength=strength,
            recommendation=recommendation,
            public_percentage=public_pct,
            line_movement=line_movement,
            sharp_money_detected=sharp_money_detected,
            reasons=reasons
        )

    def _estimate_public_percentage(
        self,
        game: Dict,
        home_team: str,
        away_team: str
    ) -> float:
        """
        Estimate public betting percentage on home team

        WHY: Real public data costs money - estimate from market indicators

        NCAA patterns:
        - Home teams get 55-65% public money (HFA bias)
        - Big name schools get 5-10% extra public
        - Conference favorites get extra public

        Returns: Estimated % of public on home team (0.0-1.0)
        """

        # Start with baseline home advantage
        public_home = 0.58  # 58% public on home team (average)

        # Adjust for spread (favorites get more public)
        spread = game.get('current_spread', 0)
        if spread < -7:  # Home favored by 7+
            public_home += 0.10
        elif spread < -3:  # Home favored by 3-7
            public_home += 0.05
        elif spread > 7:  # Home underdog by 7+
            public_home -= 0.10
        elif spread > 3:  # Home underdog by 3-7
            public_home -= 0.05

        # Adjust for big name schools
        big_name_schools = [
            'Alabama', 'Ohio State', 'Georgia', 'Michigan',
            'Notre Dame', 'Texas', 'USC', 'Oklahoma',
            'LSU', 'Florida', 'Penn State', 'Clemson'
        ]

        if any(school in home_team for school in big_name_schools):
            public_home += 0.08  # Big name at home = more public
        if any(school in away_team for school in big_name_schools):
            public_home -= 0.08  # Big name on road = less public on home

        # Adjust for conference (some conferences get more public attention)
        sec_teams = ['Alabama', 'Georgia', 'LSU', 'Florida', 'Tennessee', 'Auburn']
        big_ten_teams = ['Ohio State', 'Michigan', 'Penn State', 'Wisconsin']

        if any(team in home_team for team in sec_teams + big_ten_teams):
            public_home += 0.03

        # Clamp to 0.20-0.80 (20%-80% range)
        public_home = max(0.20, min(0.80, public_home))

        return public_home

    def fetch_line_movement(
        self,
        game_id: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Fetch opening and current spread from Odds API

        WHY: Line movement critical for detecting sharp money

        Returns: (opening_spread, current_spread)
        """

        try:
            # Fetch current odds
            url = f"{self.base_url}/sports/americanfootball_ncaaf/events/{game_id}/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'spreads',
                'oddsFormat': 'american'
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                return None, None

            data = response.json()

            # Extract spreads from different books
            opening_spreads = []
            current_spreads = []

            for bookmaker in data.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'home':
                                # Get opening line if available
                                if 'opening' in outcome:
                                    opening_spreads.append(outcome['opening']['point'])
                                current_spreads.append(outcome['point'])

            if not current_spreads:
                return None, None

            # Average across books
            opening = sum(opening_spreads) / len(opening_spreads) if opening_spreads else None
            current = sum(current_spreads) / len(current_spreads)

            return opening, current

        except Exception as e:
            print(f"⚠️  Error fetching line movement: {e}")
            return None, None

    def print_contrarian_analysis(
        self,
        signal: ContrarianSignal,
        home_team: str,
        away_team: str
    ):
        """
        Display contrarian analysis to user

        WHY: Context Embedded - shows reasoning inline
        """

        print(f"\n{'='*80}")
        print(f"🎯 CONTRARIAN INTELLIGENCE: {home_team} vs {away_team}")
        print(f"{'='*80}\n")

        print(f"📊 Contrarian Strength: {signal}")
        print(f"💡 Recommendation: {signal.recommendation}")

        if signal.reasons:
            print(f"\n📋 Signals Detected:")
            for reason in signal.reasons:
                print(f"   • {reason}")

        print(f"\n📈 Metrics:")
        print(f"   Public on home: {signal.public_percentage:.0%}")
        print(f"   Line movement: {signal.line_movement:.1f} points")
        print(f"   Sharp money: {'✅ YES' if signal.sharp_money_detected else '❌ NO'}")

        if signal.strength >= 3:
            print(f"\n🚨 STRONG CONTRARIAN SIGNAL")
            print(f"   Consider fading the public on this game!")

        print()


def main():
    """Test contrarian intelligence"""

    import sys

    if len(sys.argv) < 2:
        print("Usage: python ncaa_contrarian_intelligence.py <ODDS_API_KEY>")
        return

    api_key = sys.argv[1]
    contrarian = NCAAContrarianIntelligence(api_key)

    # Test game
    test_game = {
        'home_team': 'Alabama',
        'away_team': 'Auburn',
        'opening_spread': -14.5,
        'current_spread': -13.0,
        'day_of_week': 'Saturday',
        'conference': 'SEC'
    }

    signal = contrarian.analyze_game(
        test_game,
        test_game['home_team'],
        test_game['away_team']
    )

    contrarian.print_contrarian_analysis(
        signal,
        test_game['home_team'],
        test_game['away_team']
    )


if __name__ == "__main__":
    main()
