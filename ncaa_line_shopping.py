#!/usr/bin/env python3
"""
NCAA Line Shopping Module
==========================

Find the best available line across multiple sportsbooks

WHY THIS MATTERS:
- Getting -2.5 instead of -3.0 = +2-3% win rate boost
- 0.5 point improvement = ~$500/year on $100 bets
- Key numbers (3, 7, 10) worth even more

Example:
  DraftKings: Toledo -3.0 (-110)
  FanDuel: Toledo -2.5 (-110)  ← BEST LINE (+0.5 pts!)
  BetMGM: Toledo -3.5 (-110)

  Bet on FanDuel = Extra 0.5 points of value

USAGE:
    from ncaa_line_shopping import LineShoppingModule

    shopper = LineShoppingModule()

    # Add lines from different books
    shopper.add_line('DraftKings', 'Toledo', -3.0, -110)
    shopper.add_line('FanDuel', 'Toledo', -2.5, -110)
    shopper.add_line('BetMGM', 'Toledo', -3.5, -110)

    # Get best line
    best = shopper.get_best_line('Toledo', 'spread')
    print(f"Best line: {best['book']} at {best['spread']}")
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class BookLine:
    """Single line from a sportsbook"""
    book: str
    team: str
    line_type: str  # 'spread', 'moneyline', 'total'
    value: float  # Spread value or ML odds
    odds: float  # American odds (e.g., -110)
    timestamp: str
    juice: float = 0  # Calculated juice


@dataclass
class BestLineResult:
    """Best available line result"""
    book: str
    team: str
    line_type: str
    value: float
    odds: float
    advantage: float  # How much better than worst line
    all_lines: List[BookLine]
    crosses_key_number: bool = False
    key_number_advantage: Optional[str] = None


class LineShoppingModule:
    """
    Find best available lines across multiple sportsbooks

    Tracks lines in real-time and finds optimal betting opportunities
    """

    # Major sportsbooks
    SPORTSBOOKS = [
        'DraftKings',
        'FanDuel',
        'BetMGM',
        'Caesars',
        'BetRivers',
        'PointsBet',
        'Barstool',
        'WynnBET'
    ]

    # Key numbers in football
    KEY_NUMBERS = {
        3: 'Most common margin (FG)',
        7: 'TD margin',
        10: 'TD + FG',
        6: 'Two FGs',
        4: 'TD with missed PAT',
        14: 'Two TDs'
    }

    def __init__(self, data_file: str = "data/line_shopping_history.json"):
        self.data_file = Path(data_file)
        self.lines: Dict[str, List[BookLine]] = {}  # game_key -> list of lines
        self._load_history()

    def _load_history(self):
        """Load line history from file"""
        if self.data_file.exists():
            with open(self.data_file) as f:
                data = json.load(f)
                # Reconstruct lines dict
                for game_key, lines in data.get('lines', {}).items():
                    self.lines[game_key] = [BookLine(**line) for line in lines]

    def _save_history(self):
        """Save line history to file"""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        lines_dict = {}
        for game_key, lines in self.lines.items():
            lines_dict[game_key] = [asdict(line) for line in lines]

        with open(self.data_file, 'w') as f:
            json.dump({
                'last_updated': datetime.now().isoformat(),
                'lines': lines_dict
            }, f, indent=2)

    def add_line(
        self,
        book: str,
        team: str,
        value: float,
        odds: float = -110,
        line_type: str = 'spread',
        game_key: Optional[str] = None
    ):
        """
        Add line from a sportsbook

        Args:
            book: Sportsbook name
            team: Team name
            value: Spread value or ML odds
            odds: American odds (default -110)
            line_type: 'spread', 'moneyline', or 'total'
            game_key: Optional game identifier
        """

        if game_key is None:
            game_key = f"{team}_{datetime.now().strftime('%Y%m%d')}"

        # Calculate juice
        juice = self._calculate_juice(odds)

        line = BookLine(
            book=book,
            team=team,
            line_type=line_type,
            value=value,
            odds=odds,
            timestamp=datetime.now().isoformat(),
            juice=juice
        )

        if game_key not in self.lines:
            self.lines[game_key] = []

        self.lines[game_key].append(line)
        self._save_history()

    def _calculate_juice(self, odds: float) -> float:
        """Calculate juice (vig) from American odds"""
        if odds < 0:
            implied_prob = abs(odds) / (abs(odds) + 100)
        else:
            implied_prob = 100 / (odds + 100)

        # Juice = implied prob - 50%
        juice = (implied_prob - 0.5) * 100
        return juice

    def get_best_line(
        self,
        team: str,
        line_type: str = 'spread'
    ) -> Optional[BestLineResult]:
        """
        Get best available line for team

        Args:
            team: Team name
            line_type: 'spread', 'moneyline', or 'total'

        Returns:
            BestLineResult with best line and analysis
        """

        # Find all lines for this team
        all_team_lines = []
        for game_key, lines in self.lines.items():
            for line in lines:
                if line.team.lower() == team.lower() and line.line_type == line_type:
                    all_team_lines.append(line)

        if not all_team_lines:
            return None

        # For spreads: More negative is worse for favorite, more positive better for underdog
        # We want the BEST VALUE for the bettor
        if line_type == 'spread':
            # Determine if team is favorite or underdog
            avg_value = sum(l.value for l in all_team_lines) / len(all_team_lines)

            if avg_value < 0:
                # Team is favorite - want least negative (closest to 0)
                # -2.5 is better than -3.0
                best_line = max(all_team_lines, key=lambda l: l.value)
                worst_line = min(all_team_lines, key=lambda l: l.value)
            else:
                # Team is underdog - want most positive
                # +3.0 is better than +2.5
                best_line = max(all_team_lines, key=lambda l: l.value)
                worst_line = min(all_team_lines, key=lambda l: l.value)

            advantage = abs(best_line.value - worst_line.value)

            # Check if crosses key number
            crosses_key, key_advantage = self._check_key_number_cross(
                worst_line.value,
                best_line.value
            )

        elif line_type == 'moneyline':
            # For ML: More positive is always better (better payout)
            best_line = max(all_team_lines, key=lambda l: l.value)
            worst_line = min(all_team_lines, key=lambda l: l.value)
            advantage = best_line.value - worst_line.value
            crosses_key = False
            key_advantage = None

        else:  # total
            # For totals, depends on over/under
            best_line = max(all_team_lines, key=lambda l: l.value)
            worst_line = min(all_team_lines, key=lambda l: l.value)
            advantage = abs(best_line.value - worst_line.value)
            crosses_key = False
            key_advantage = None

        return BestLineResult(
            book=best_line.book,
            team=team,
            line_type=line_type,
            value=best_line.value,
            odds=best_line.odds,
            advantage=advantage,
            all_lines=all_team_lines,
            crosses_key_number=crosses_key,
            key_number_advantage=key_advantage
        )

    def _check_key_number_cross(
        self,
        worst_value: float,
        best_value: float
    ) -> Tuple[bool, Optional[str]]:
        """Check if best line crosses a key number"""

        # Check each key number
        for key_num, description in self.KEY_NUMBERS.items():
            # Check if key number is between worst and best
            if worst_value < key_num < best_value or best_value < key_num < worst_value:
                return True, f"Crosses key number {key_num} ({description})"

        return False, None

    def print_line_comparison(self, team: str, line_type: str = 'spread'):
        """Print line comparison across all books"""

        result = self.get_best_line(team, line_type)

        if not result:
            print(f"No lines found for {team}")
            return

        print(f"\n{'='*80}")
        print(f"📊 LINE SHOPPING: {team} ({line_type.upper()})")
        print(f"{'='*80}\n")

        # Sort lines by value (best first)
        sorted_lines = sorted(
            result.all_lines,
            key=lambda l: l.value,
            reverse=True
        )

        for i, line in enumerate(sorted_lines):
            icon = '🏆' if i == 0 else '  '
            print(f"{icon} {line.book:15s}: {line.value:+6.1f} ({line.odds:+4.0f})")

        print()
        print(f"✅ BEST LINE: {result.book} at {result.value:+.1f}")
        print(f"   Advantage: {result.advantage:.1f} points better than worst")

        if result.crosses_key_number:
            print(f"   🔥 {result.key_number_advantage} - SIGNIFICANT EDGE!")

        print(f"\n{'='*80}\n")

    def get_line_shopping_summary(self) -> Dict:
        """Get summary of line shopping opportunities"""

        total_games = len(self.lines)
        total_lines = sum(len(lines) for lines in self.lines.values())

        return {
            'total_games_tracked': total_games,
            'total_lines': total_lines,
            'avg_lines_per_game': total_lines / total_games if total_games > 0 else 0,
            'sportsbooks_tracked': len(set(
                line.book for lines in self.lines.values() for line in lines
            ))
        }


def main():
    """Demo line shopping"""

    print("NCAA Line Shopping Demo\n")

    shopper = LineShoppingModule()

    # Example 1: Toledo spread across books
    print("Example 1: Line Shopping - Toledo Spread\n")

    shopper.add_line('DraftKings', 'Toledo', -3.0, -110, 'spread', 'toledo_bg_20251112')
    shopper.add_line('FanDuel', 'Toledo', -2.5, -110, 'spread', 'toledo_bg_20251112')
    shopper.add_line('BetMGM', 'Toledo', -3.5, -110, 'spread', 'toledo_bg_20251112')
    shopper.add_line('Caesars', 'Toledo', -3.0, -115, 'spread', 'toledo_bg_20251112')

    shopper.print_line_comparison('Toledo', 'spread')

    # Example 2: Crossing key number 3
    print("\nExample 2: Key Number Advantage - Crossing 3\n")

    shopper.add_line('DraftKings', 'Alabama', -3.5, -110, 'spread', 'bama_auburn_20251112')
    shopper.add_line('FanDuel', 'Alabama', -3.0, -110, 'spread', 'bama_auburn_20251112')
    shopper.add_line('BetMGM', 'Alabama', -2.5, -110, 'spread', 'bama_auburn_20251112')

    shopper.print_line_comparison('Alabama', 'spread')

    # Example 3: Moneyline shopping
    print("\nExample 3: Moneyline Shopping\n")

    shopper.add_line('DraftKings', 'Ohio', 150, -110, 'moneyline', 'ohio_kent_20251112')
    shopper.add_line('FanDuel', 'Ohio', 165, -110, 'moneyline', 'ohio_kent_20251112')
    shopper.add_line('BetMGM', 'Ohio', 155, -110, 'moneyline', 'ohio_kent_20251112')

    shopper.print_line_comparison('Ohio', 'moneyline')

    # Summary
    summary = shopper.get_line_shopping_summary()
    print(f"\n📊 LINE SHOPPING SUMMARY:")
    print(f"   Games tracked: {summary['total_games_tracked']}")
    print(f"   Total lines: {summary['total_lines']}")
    print(f"   Avg lines per game: {summary['avg_lines_per_game']:.1f}")
    print(f"   Sportsbooks tracked: {summary['sportsbooks_tracked']}")
    print()


if __name__ == "__main__":
    main()
