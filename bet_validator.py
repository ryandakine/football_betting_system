#!/usr/bin/env python3
"""
Bet Validator - Ensures NO Mock Data in Production Betting

WHY THIS EXISTS:
CRITICAL: We were betting real money based on mock/sample data.
This validator BLOCKS all bets unless data is verified as REAL.

DESIGN PHILOSOPHY: Fail Fast, Fail Loud
- If data is invalid → ERROR (don't fake it)
- If data is unavailable → ERROR (don't fake it)
- If data is mock → ERROR (don't fake it)
- NO fallbacks to sample data
- NO "just use this for now" logic

USAGE:
    from bet_validator import BetValidator

    validator = BetValidator()

    # Validate before placing bet
    is_valid, errors = validator.validate_bet(
        game="PHI @ GB",
        referee="Shawn Hochuli",
        bankroll=100.0,
        amount=5.0
    )

    if not is_valid:
        print("❌ BET BLOCKED:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BetValidator:
    """
    Validates betting data to ensure NO mock/sample data is used.

    CRITICAL RULES:
    1. All data must come from real sources (APIs, scrapers)
    2. No fallback to mock/sample data
    3. If validation fails → Block bet and error
    4. If data unavailable → Block bet and error
    """

    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        self.bankroll_file = Path(__file__).parent / "bankroll.json"
        self.bet_log_file = Path(__file__).parent / "bet_log.json"

        # Mock data patterns that should NEVER appear in production
        self.mock_patterns = [
            r'sample',
            r'mock',
            r'fake',
            r'test',
            r'dummy',
            r'placeholder',
            r'SAMPLE',
            r'MOCK',
            r'FAKE',
            r'TEST',
            r'DUMMY'
        ]

        # Invalid game patterns
        self.invalid_game_patterns = [
            r'^TEAM\d+',  # TEAM1, TEAM2, etc
            r'^HOME',     # HOME, AWAY placeholders
            r'^AWAY',
            r'TBD',
            r'Unknown'
        ]

    def validate_bet(
        self,
        game: str,
        referee: Optional[str],
        bankroll: float,
        amount: float,
        odds_data: Optional[Dict] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate all components of a bet.

        Args:
            game: Game string (e.g., "PHI @ GB")
            referee: Referee name
            bankroll: Current bankroll
            amount: Bet amount
            odds_data: Odds data from API

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # 1. Validate game
        game_valid, game_errors = self.validate_game(game)
        if not game_valid:
            errors.extend(game_errors)

        # 2. Validate referee
        referee_valid, referee_errors = self.validate_referee(referee, game)
        if not referee_valid:
            errors.extend(referee_errors)

        # 3. Validate bankroll
        bankroll_valid, bankroll_errors = self.validate_bankroll(bankroll, amount)
        if not bankroll_valid:
            errors.extend(bankroll_errors)

        # 4. Validate odds if provided
        if odds_data:
            odds_valid, odds_errors = self.validate_odds(odds_data)
            if not odds_valid:
                errors.extend(odds_errors)

        # 5. Check for mock data in strings
        mock_check_valid, mock_errors = self._check_for_mock_data(
            game=game,
            referee=referee,
            odds_data=odds_data
        )
        if not mock_check_valid:
            errors.extend(mock_errors)

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_game(self, game: str) -> Tuple[bool, List[str]]:
        """
        Validate game string.

        Args:
            game: Game string like "PHI @ GB"

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check if empty
        if not game or game.strip() == "":
            errors.append("Game is empty or None")
            return False, errors

        # Check for invalid patterns
        for pattern in self.invalid_game_patterns:
            if re.search(pattern, game, re.IGNORECASE):
                errors.append(f"Game contains invalid pattern: {pattern} (found in '{game}')")

        # Validate format (should be "TEAM @ TEAM")
        if '@' not in game and ' at ' not in game.lower():
            errors.append(f"Game format invalid: '{game}' (should be 'AWAY @ HOME')")

        # Extract team names
        parts = game.replace(' at ', ' @ ').split('@')
        if len(parts) != 2:
            errors.append(f"Game format invalid: '{game}' (should have exactly 2 teams)")
        else:
            away = parts[0].strip()
            home = parts[1].strip()

            # Validate team names (should be 2-3 letter abbreviations or full names)
            if len(away) < 2:
                errors.append(f"Away team name too short: '{away}'")
            if len(home) < 2:
                errors.append(f"Home team name too short: '{home}'")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_referee(self, referee: Optional[str], game: str) -> Tuple[bool, List[str]]:
        """
        Validate referee assignment.

        Args:
            referee: Referee name
            game: Game string (for context)

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Referee can be None (not all games have ref data available yet)
        if referee is None:
            # This is OK - just no bonus bet
            return True, []

        # Check if empty string or placeholder
        if referee.strip() == "":
            errors.append("Referee is empty string")
            return False, errors

        # Check for placeholder values
        invalid_values = ['Unknown', 'TBD', 'N/A', 'None', 'null']
        if referee in invalid_values:
            errors.append(f"Referee is placeholder value: '{referee}'")

        # Referee name should have at least first and last name
        parts = referee.strip().split()
        if len(parts) < 2:
            errors.append(f"Referee name invalid: '{referee}' (should be 'First Last')")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_bankroll(self, bankroll: float, amount: float) -> Tuple[bool, List[str]]:
        """
        Validate bankroll and bet amount.

        Args:
            bankroll: Current bankroll
            amount: Bet amount

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check bankroll is positive
        if bankroll <= 0:
            errors.append(f"Bankroll is zero or negative: ${bankroll:.2f}")

        # Check bet amount is positive
        if amount <= 0:
            errors.append(f"Bet amount is zero or negative: ${amount:.2f}")

        # Check sufficient bankroll
        if amount > bankroll:
            errors.append(f"Insufficient bankroll: ${amount:.2f} bet > ${bankroll:.2f} available")

        # Check for suspicious round numbers (might be mock data)
        if bankroll == 100.0 and amount in [5.0, 10.0]:
            # This is actually OK for initial bankroll
            pass

        # Warn if bankroll file doesn't exist
        if not self.bankroll_file.exists():
            errors.append(f"Bankroll file doesn't exist: {self.bankroll_file}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_odds(self, odds_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate odds data from API.

        Args:
            odds_data: Odds data dict

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check if empty
        if not odds_data:
            errors.append("Odds data is empty")
            return False, errors

        # Check for required fields
        required_fields = ['home_team', 'away_team']
        for field in required_fields:
            if field not in odds_data:
                errors.append(f"Odds data missing required field: '{field}'")

        # Check team names aren't placeholders
        if 'home_team' in odds_data:
            for pattern in self.invalid_game_patterns:
                if re.search(pattern, odds_data['home_team'], re.IGNORECASE):
                    errors.append(f"Home team is placeholder: '{odds_data['home_team']}'")

        if 'away_team' in odds_data:
            for pattern in self.invalid_game_patterns:
                if re.search(pattern, odds_data['away_team'], re.IGNORECASE):
                    errors.append(f"Away team is placeholder: '{odds_data['away_team']}'")

        # Check odds values are reasonable (-2000 to +2000)
        if 'home_ml' in odds_data:
            ml = odds_data['home_ml']
            if abs(ml) > 2000:
                errors.append(f"Home moneyline unreasonable: {ml}")

        if 'away_ml' in odds_data:
            ml = odds_data['away_ml']
            if abs(ml) > 2000:
                errors.append(f"Away moneyline unreasonable: {ml}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def _check_for_mock_data(
        self,
        game: str,
        referee: Optional[str],
        odds_data: Optional[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Check for mock data patterns in all fields.

        Args:
            game: Game string
            referee: Referee name
            odds_data: Odds data

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check game string
        for pattern in self.mock_patterns:
            if re.search(pattern, game, re.IGNORECASE):
                errors.append(f"MOCK DATA DETECTED in game: '{game}' (pattern: {pattern})")

        # Check referee
        if referee:
            for pattern in self.mock_patterns:
                if re.search(pattern, referee, re.IGNORECASE):
                    errors.append(f"MOCK DATA DETECTED in referee: '{referee}' (pattern: {pattern})")

        # Check odds data
        if odds_data:
            odds_str = json.dumps(odds_data)
            for pattern in self.mock_patterns:
                if re.search(pattern, odds_str, re.IGNORECASE):
                    errors.append(f"MOCK DATA DETECTED in odds data (pattern: {pattern})")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_data_source(self, data: Dict, source_name: str) -> Tuple[bool, List[str]]:
        """
        Validate that data came from a real source (not hardcoded).

        Args:
            data: Data dict
            source_name: Name of the data source for error messages

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check for timestamp (real data should have fetch timestamps)
        if 'timestamp' not in data and 'fetched_at' not in data:
            errors.append(f"{source_name}: Missing timestamp (might be hardcoded data)")

        # Check for source URL (real data should reference source)
        if 'source_url' not in data and 'api_source' not in data:
            errors.append(f"{source_name}: Missing source reference (might be hardcoded data)")

        is_valid = len(errors) == 0
        return is_valid, errors

    def block_bet_with_error(self, errors: List[str]) -> None:
        """
        Block bet and print errors.

        Args:
            errors: List of validation errors
        """
        print("=" * 70)
        print("❌ BET BLOCKED - VALIDATION FAILED")
        print("=" * 70)
        print()
        print("The following validation errors occurred:")
        print()

        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")

        print()
        print("🚨 CRITICAL: Do NOT override this validation!")
        print("   Fix the data source, don't fake the data.")
        print()
        print("Common fixes:")
        print("  - Ensure ODDS_API_KEY is set")
        print("  - Check API is returning real data")
        print("  - Verify referee scraper is working")
        print("  - Don't use betting cards with mock data")
        print()
        print("=" * 70)


def main():
    """Test the validator"""
    import argparse

    parser = argparse.ArgumentParser(description="Test bet validator")
    parser.add_argument("--game", required=True, help="Game to validate")
    parser.add_argument("--referee", help="Referee name")
    parser.add_argument("--bankroll", type=float, default=100.0, help="Current bankroll")
    parser.add_argument("--amount", type=float, default=5.0, help="Bet amount")

    args = parser.parse_args()

    validator = BetValidator()

    is_valid, errors = validator.validate_bet(
        game=args.game,
        referee=args.referee,
        bankroll=args.bankroll,
        amount=args.amount
    )

    if is_valid:
        print("✅ Validation passed - bet can proceed")
    else:
        validator.block_bet_with_error(errors)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
