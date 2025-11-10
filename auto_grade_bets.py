#!/usr/bin/env python3
"""
Automatic Bet Result Grading
=============================

WHY THIS EXISTS:
----------------
Manual grading causes errors:
- Forgot to check result
- Miscalculated winnings
- Updated bankroll wrong

INVESTMENT → SYSTEM:
--------------------
System automatically:
- Fetches game results from API
- Determines WIN/LOSS/PUSH
- Updates bankroll
- Sends notification

ERROR PREVENTION:
-----------------
Agent cannot:
- Forget to grade bets
- Grade incorrectly
- Calculate winnings wrong

OPERATIONAL COST:
-----------------
Initial: 1 hour (this file)
Ongoing: 0 (runs automatically via cron or post-game hook)
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bankroll_tracker import BankrollTracker

BET_LOG_FILE = Path(__file__).parent / "data" / "bet_log.json"


class AutoBetGrader:
    """
    Automatically grades bets by fetching actual game results.

    WHY: Eliminates manual work and human error.
    """

    def __init__(self):
        self.bet_log = BET_LOG_FILE
        self.tracker = BankrollTracker()
        self.api_key = os.getenv("THE_ODDS_API_KEY")

    def fetch_game_result(self, game: str, game_date: str) -> Optional[Dict]:
        """
        Fetch game result from The Odds API.

        Args:
            game: Game string (e.g., "PHI @ GB")
            game_date: Date string (e.g., "2025-11-10")

        Returns:
            Dict with final scores or None if not finished

        WHY: Single source of truth for game results.
        """
        if not self.api_key:
            print("⚠️  THE_ODDS_API_KEY not set - cannot fetch results")
            print("   Set in .env file or grade manually with:")
            print(f"   python bankroll_tracker.py --grade \"{game}\" WIN|LOSS|PUSH")
            return None

        try:
            import requests

            # Parse teams from game string
            if '@' in game:
                away, home = game.split('@')
                away_team = away.strip()
                home_team = home.strip()
            else:
                print(f"⚠️  Invalid game format: {game}")
                return None

            # Fetch scores from The Odds API
            # Note: The Odds API has scores endpoint for completed games
            url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/scores/"
            params = {
                'apiKey': self.api_key,
                'daysFrom': 3  # Look back 3 days
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"⚠️  API error: {response.status_code}")
                return None

            games = response.json()

            # Find matching game
            for g in games:
                if g.get('completed'):
                    # Match team names (may need fuzzy matching)
                    if (away_team.upper() in g.get('away_team', '').upper() and
                        home_team.upper() in g.get('home_team', '').upper()):

                        return {
                            'home_score': g.get('scores', [{}])[0].get('score'),
                            'away_score': g.get('scores', [{}])[1].get('score') if len(g.get('scores', [])) > 1 else None,
                            'completed': True
                        }

            # Game not found or not completed
            return None

        except Exception as e:
            print(f"❌ Error fetching game result: {e}")
            return None

    def grade_spread_bet(self, pick: str, home_score: int, away_score: int, game: str) -> str:
        """
        Determine if spread bet won/lost/pushed.

        Args:
            pick: Bet pick (e.g., "GB -1.5", "PHI +3")
            home_score: Home team final score
            away_score: Away team final score
            game: Game string for team identification

        Returns:
            "WIN", "LOSS", or "PUSH"

        WHY: Eliminates manual calculation errors.
        """
        # Parse pick
        if '@' in game:
            away_team, home_team = [t.strip() for t in game.split('@')]
        else:
            return "ERROR"

        # Determine which team was bet on
        if away_team.upper()[:3] in pick.upper():
            # Bet on away team
            spread = float(pick.split('+' if '+' in pick else ' ')[1])
            adjusted_score = away_score + spread
            result = "WIN" if adjusted_score > home_score else "LOSS" if adjusted_score < home_score else "PUSH"

        elif home_team.upper()[:3] in pick.upper():
            # Bet on home team
            spread_str = pick.split('-' if '-' in pick else ' ')[-1]
            spread = float(spread_str) if spread_str else 0
            adjusted_score = home_score - spread
            result = "WIN" if adjusted_score > away_score else "LOSS" if adjusted_score < away_score else "PUSH"

        else:
            return "ERROR"

        return result

    def grade_total_bet(self, pick: str, home_score: int, away_score: int) -> str:
        """
        Determine if total bet won/lost/pushed.

        Args:
            pick: Bet pick (e.g., "OVER 45.5", "UNDER 42")
            home_score: Home team final score
            away_score: Away team final score

        Returns:
            "WIN", "LOSS", or "PUSH"

        WHY: Automated, error-free totals grading.
        """
        total_score = home_score + away_score

        if 'OVER' in pick.upper():
            line = float(pick.split()[-1])
            result = "WIN" if total_score > line else "LOSS" if total_score < line else "PUSH"

        elif 'UNDER' in pick.upper():
            line = float(pick.split()[-1])
            result = "WIN" if total_score < line else "LOSS" if total_score > line else "PUSH"

        else:
            return "ERROR"

        return result

    def grade_moneyline_bet(self, pick: str, home_score: int, away_score: int, game: str) -> str:
        """
        Determine if moneyline bet won/lost.

        Args:
            pick: Bet pick (e.g., "GB ML", "PHI MONEYLINE")
            home_score: Home team final score
            away_score: Away team final score
            game: Game string

        Returns:
            "WIN" or "LOSS"

        WHY: Simple win/loss, no calculations.
        """
        if '@' in game:
            away_team, home_team = [t.strip() for t in game.split('@')]
        else:
            return "ERROR"

        # Determine which team was bet on
        if away_team.upper()[:3] in pick.upper():
            result = "WIN" if away_score > home_score else "LOSS"
        elif home_team.upper()[:3] in pick.upper():
            result = "WIN" if home_score > away_score else "LOSS"
        else:
            return "ERROR"

        return result

    def grade_pending_bets(self, dry_run: bool = False):
        """
        Grade all pending bets automatically.

        Args:
            dry_run: If True, show what would happen but don't update

        WHY: Fully automated bet grading.
        """
        pending = self.tracker.get_pending_bets()

        if not pending:
            print("✅ No pending bets to grade")
            return

        print(f"\n{'='*60}")
        print(f"🔍 GRADING {len(pending)} PENDING BET(S)")
        print(f"{'='*60}\n")

        graded = 0

        for bet in pending:
            game = bet['game']
            pick = bet['pick']
            bet_type = bet.get('bet_type', 'spread')
            game_date = bet.get('placed_at', datetime.now().isoformat())[:10]

            print(f"Checking: {game} - {pick}...")

            # Fetch result
            result_data = self.fetch_game_result(game, game_date)

            if not result_data or not result_data.get('completed'):
                print(f"  ⏳ Game not finished yet\n")
                continue

            home_score = result_data['home_score']
            away_score = result_data['away_score']

            print(f"  📊 Final Score: {away_score} - {home_score}")

            # Determine result based on bet type
            if bet_type == 'spread':
                result = self.grade_spread_bet(pick, home_score, away_score, game)
            elif bet_type == 'total':
                result = self.grade_total_bet(pick, home_score, away_score)
            elif bet_type == 'moneyline':
                result = self.grade_moneyline_bet(pick, home_score, away_score, game)
            else:
                print(f"  ❌ Unknown bet type: {bet_type}\n")
                continue

            if result == "ERROR":
                print(f"  ❌ Could not determine result\n")
                continue

            emoji = "🎉" if result == "WIN" else "😞" if result == "LOSS" else "🤝"
            print(f"  {emoji} {result}")

            if not dry_run:
                # Update bankroll
                try:
                    new_bankroll = self.tracker.record_result(game, result, bet.get('odds', -110))
                    print(f"  💰 New bankroll: ${new_bankroll:.2f}")
                    graded += 1
                except Exception as e:
                    print(f"  ❌ Error grading: {e}")

            print()

        if dry_run:
            print(f"{'='*60}")
            print(f"DRY RUN: Would have graded {graded} bet(s)")
            print(f"{'='*60}\n")
        else:
            print(f"{'='*60}")
            print(f"✅ Graded {graded} bet(s) successfully")
            print(f"{'='*60}\n")

            # Show updated stats
            stats = self.tracker.get_stats()
            print(f"Current Bankroll: ${stats['current_bankroll']:.2f}")
            print(f"Win Rate: {stats['win_rate']:.2%}")
            print(f"ROI: {stats['roi']:.2f}%\n")


def main():
    """CLI interface for auto grading."""
    import argparse

    parser = argparse.ArgumentParser(description="Auto Grade Bets")
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would happen without updating')
    parser.add_argument('--game', type=str,
                       help='Grade specific game only')

    args = parser.parse_args()

    grader = AutoBetGrader()

    if args.game:
        # Grade specific game
        pending = grader.tracker.get_pending_bets()
        found = False

        for bet in pending:
            if args.game.upper() in bet['game'].upper():
                found = True
                result_data = grader.fetch_game_result(bet['game'], bet.get('placed_at', '')[:10])

                if not result_data or not result_data.get('completed'):
                    print(f"⏳ {bet['game']} not finished yet")
                    return

                print(f"Final Score: {result_data['away_score']} - {result_data['home_score']}")

                if bet['bet_type'] == 'spread':
                    result = grader.grade_spread_bet(bet['pick'], result_data['home_score'],
                                                    result_data['away_score'], bet['game'])
                elif bet['bet_type'] == 'total':
                    result = grader.grade_total_bet(bet['pick'], result_data['home_score'],
                                                   result_data['away_score'])
                else:
                    result = grader.grade_moneyline_bet(bet['pick'], result_data['home_score'],
                                                       result_data['away_score'], bet['game'])

                if not args.dry_run and result != "ERROR":
                    new_balance = grader.tracker.record_result(bet['game'], result, bet.get('odds', -110))
                    print(f"✅ {result} - New bankroll: ${new_balance:.2f}")
                elif result != "ERROR":
                    print(f"DRY RUN: Would grade as {result}")

        if not found:
            print(f"❌ No pending bet found for: {args.game}")

    else:
        # Grade all pending
        grader.grade_pending_bets(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
