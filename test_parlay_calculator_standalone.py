#!/usr/bin/env python3
"""
Standalone test script for the ParlayCalculator
"""

import re
from typing import List, Dict, Any, Tuple

class ParlayCalculator:
    """Advanced parlay calculation system with odds conversion and correlation analysis"""

    def __init__(self):
        self.correlation_warnings = []
        self.risk_factors = []

    @staticmethod
    def american_to_decimal(american_odds: float) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100.0) + 1
        else:
            return (100.0 / abs(american_odds)) + 1

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> float:
        """Convert decimal odds to American odds"""
        if decimal_odds >= 2.0:
            return (decimal_odds - 1) * 100
        else:
            return -100 / (decimal_odds - 1)

    def parse_bet_string(self, bet_string: str) -> Dict[str, Any]:
        """Parse a bet string to extract team, type, and odds"""
        # Example formats:
        # "Kansas City Chiefs -120" (moneyline)
        # "Kansas City Chiefs ML -120"
        # "Chiefs vs Raiders - ML -150"

        bet_info = {
            'team': '',
            'bet_type': 'moneyline',
            'american_odds': 0.0,
            'decimal_odds': 1.0,
            'game_info': bet_string
        }

        # Try to extract odds (American format: +150, -120, etc.)
        odds_match = re.search(r'([+-]\d+)', bet_string)
        if odds_match:
            bet_info['american_odds'] = float(odds_match.group(1))
            bet_info['decimal_odds'] = self.american_to_decimal(bet_info['american_odds'])

        # Extract team name (everything before the odds)
        if odds_match:
            team_part = bet_string[:odds_match.start()].strip()
            # Clean up common patterns
            team_part = re.sub(r'\s*(vs|@|ML|MLB|NFL|NCAAF)\s*$', '', team_part)
            bet_info['team'] = team_part.strip()

        return bet_info

    def calculate_parlay_odds(self, bet_strings: List[str]) -> Dict[str, Any]:
        """Calculate parlay odds from a list of bet strings"""
        if not bet_strings:
            return {'decimal_odds': 1.0, 'american_odds': 0.0, 'legs': 0}

        parsed_bets = []
        decimal_multiplier = 1.0
        self.correlation_warnings = []
        self.risk_factors = []

        for bet_string in bet_strings:
            bet_info = self.parse_bet_string(bet_string)
            parsed_bets.append(bet_info)

            if bet_info['decimal_odds'] > 1.0:
                decimal_multiplier *= bet_info['decimal_odds']
            else:
                # If we can't parse odds, assume 2.0 (even money)
                decimal_multiplier *= 2.0
                self.risk_factors.append(f"Could not parse odds for: {bet_string}")

        # Analyze correlations
        self._analyze_correlations(parsed_bets)

        american_odds = self.decimal_to_american(decimal_multiplier)

        return {
            'decimal_odds': round(decimal_multiplier, 2),
            'american_odds': round(american_odds),
            'legs': len(bet_strings),
            'parsed_bets': parsed_bets,
            'correlation_warnings': self.correlation_warnings,
            'risk_factors': self.risk_factors,
            'implied_probability': round(1.0 / decimal_multiplier * 100, 2)
        }

    def _analyze_correlations(self, parsed_bets: List[Dict]) -> None:
        """Analyze correlations between bets for risk assessment"""
        if len(parsed_bets) < 2:
            return

        teams = [bet['team'].lower() for bet in parsed_bets]

        # Check for same team multiple times
        for i, team1 in enumerate(teams):
            for j, team2 in enumerate(teams):
                if i != j and team1 == team2:
                    self.correlation_warnings.append(
                        f"⚠️ Same team appears multiple times: {parsed_bets[i]['team']}"
                    )
                    break

        # Check for obvious correlations (same game)
        for bet in parsed_bets:
            game_info = bet.get('game_info', '').lower()
            if 'vs' in game_info or '@' in game_info:
                # This is a moneyline bet from same game - high correlation
                self.correlation_warnings.append(
                    f"⚠️ Multiple legs from same game: {bet['game_info'][:50]}..."
                )

        # Check for over-concentration in same conference/sport
        conference_indicators = ['chiefs', 'raiders', 'chargers', 'broncos', 'afc', 'nfc']
        nfl_count = sum(1 for team in teams if any(conf in team for conf in conference_indicators))
        if nfl_count >= 3:
            self.correlation_warnings.append(
                f"⚠️ High NFL concentration ({nfl_count}/{len(teams)} legs)"
            )

    def calculate_payout(self, parlay_odds: Dict, stake: float = 10.0) -> Dict[str, float]:
        """Calculate payout for a given stake"""
        decimal_odds = parlay_odds.get('decimal_odds', 1.0)
        payout = stake * decimal_odds
        profit = payout - stake

        return {
            'stake': stake,
            'payout': round(payout, 2),
            'profit': round(profit, 2),
            'roi_percent': round((profit / stake) * 100, 2)
        }

    def get_risk_assessment(self, parlay_result: Dict) -> str:
        """Provide risk assessment for the parlay"""
        legs = parlay_result.get('legs', 0)
        warnings = len(parlay_result.get('correlation_warnings', []))
        risk_factors = len(parlay_result.get('risk_factors', []))

        if legs <= 2:
            risk_level = "LOW"
            color = "🟢"
        elif legs <= 4 and warnings == 0:
            risk_level = "MODERATE"
            color = "🟡"
        elif legs <= 6 and warnings <= 1:
            risk_level = "HIGH"
            color = "🟠"
        else:
            risk_level = "EXTREME"
            color = "🔴"

        if warnings > 0:
            risk_level += f" (+{warnings} warnings)"

        return f"{color} {risk_level}"


def test_parlay_calculator():
    """Test the parlay calculator functionality"""

    calculator = ParlayCalculator()

    # Test American to decimal conversion
    print("=== Testing Odds Conversion ===")
    result1 = calculator.american_to_decimal(-150)
    assert abs(result1 - 1.6666666666666667) < 0.01, f"American -150 to decimal failed: {result1}"

    result2 = calculator.american_to_decimal(+200)
    assert abs(result2 - 3.0) < 0.01, f"American +200 to decimal failed: {result2}"

    result3 = calculator.decimal_to_american(1.6666666666666667)
    assert abs(result3 - (-150)) < 1, f"Decimal to American -150 failed: {result3}"

    result4 = calculator.decimal_to_american(3.0)
    assert abs(result4 - 200) < 1, f"Decimal to American +200 failed: {result4}"
    print("✅ Odds conversion tests passed")

    # Test parlay calculation
    print("\n=== Testing Parlay Calculation ===")
    test_bets = [
        "Kansas City Chiefs -150",
        "Buffalo Bills +120",
        "San Francisco 49ers -180"
    ]

    result = calculator.calculate_parlay_odds(test_bets)
    print(f"Test parlay result: {result}")

    # Expected calculation:
    # Chiefs -150 = 1.667
    # Bills +120 = 2.2
    # 49ers -180 = 1.556
    # Total decimal: 1.667 * 2.2 * 1.556 ≈ 5.71

    expected_decimal = 1.667 * 2.2 * 1.556
    assert abs(result['decimal_odds'] - expected_decimal) < 0.1, f"Decimal odds calculation failed: {result['decimal_odds']} vs {expected_decimal}"
    assert result['legs'] == 3, f"Legs count failed: {result['legs']}"
    print("✅ Parlay calculation tests passed")

    # Test correlation detection
    print("\n=== Testing Correlation Detection ===")
    correlated_bets = [
        "Kansas City Chiefs -150",
        "Kansas City Chiefs -140",  # Same team twice
        "Buffalo Bills +120"
    ]

    correlated_result = calculator.calculate_parlay_odds(correlated_bets)
    warnings = correlated_result.get('correlation_warnings', [])
    assert len(warnings) > 0, f"Correlation detection failed: {warnings}"
    print(f"✅ Correlation warnings: {warnings}")

    # Test payout calculation
    print("\n=== Testing Payout Calculation ===")
    payout_result = calculator.calculate_payout(result, stake=10.0)
    print(f"Payout result: {payout_result}")
    expected_payout = expected_decimal * 10
    assert abs(payout_result['payout'] - expected_payout) < 1.0, f"Payout calculation failed: {payout_result['payout']} vs {expected_payout}"
    expected_profit = expected_payout - 10
    assert abs(payout_result['profit'] - expected_profit) < 1.0, f"Profit calculation failed: {payout_result['profit']} vs {expected_profit}"
    print("✅ Payout calculation tests passed")

    # Test risk assessment
    print("\n=== Testing Risk Assessment ===")
    risk = calculator.get_risk_assessment(result)
    print(f"Risk assessment: {risk}")
    assert "MODERATE" in risk or "HIGH" in risk, f"Risk assessment failed: {risk}"
    print("✅ Risk assessment tests passed")

    print("\n🎉 All ParlayCalculator tests passed!")

if __name__ == "__main__":
    test_parlay_calculator()
