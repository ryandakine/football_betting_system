#!/usr/bin/env python3
"""
NFL Parlay Maker
================
Builds optimized parlays from NFL betting edges.

Features:
- 2-leg through 10-leg parlays
- Smart selection (avoid correlated bets)
- Filters by confidence, edge size, bet type
- Ranks by EV, confidence, payout
- Diversifies across games

Usage:
    python nfl_parlay_maker.py --week 10
    python nfl_parlay_maker.py --week 10 --min-confidence 0.70 --max-legs 4
    python nfl_parlay_maker.py --week 10 --parlay-size 3 --top 10
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class BettingEdge:
    """Represents a single betting edge."""
    game: str
    edge_type: str
    pick: str
    confidence: float
    edge_size: str
    reasoning: str
    odds: int = -110  # Default American odds

    @property
    def decimal_odds(self) -> float:
        """Convert American odds to decimal."""
        if self.odds > 0:
            return (self.odds / 100) + 1
        else:
            return (100 / abs(self.odds)) + 1

    def __str__(self) -> str:
        return f"{self.pick} ({self.confidence:.0%})"


@dataclass
class Parlay:
    """Represents a parlay combination."""
    legs: List[BettingEdge]

    @property
    def size(self) -> int:
        """Number of legs in parlay."""
        return len(self.legs)

    @property
    def combined_odds(self) -> float:
        """Combined decimal odds for parlay."""
        odds = 1.0
        for leg in self.legs:
            odds *= leg.decimal_odds
        return odds

    @property
    def american_odds(self) -> int:
        """Combined odds in American format."""
        decimal = self.combined_odds
        if decimal >= 2.0:
            return int((decimal - 1) * 100)
        else:
            return int(-100 / (decimal - 1))

    @property
    def combined_confidence(self) -> float:
        """Combined probability of all legs hitting."""
        prob = 1.0
        for leg in self.legs:
            prob *= leg.confidence
        return prob

    @property
    def expected_value(self) -> float:
        """Expected value per $1 wagered."""
        # EV = (probability of winning * payout) - (probability of losing * stake)
        win_prob = self.combined_confidence
        lose_prob = 1 - win_prob
        payout = self.combined_odds - 1  # Profit on $1 bet
        return (win_prob * payout) - (lose_prob * 1.0)

    @property
    def games_involved(self) -> set:
        """Set of unique games in parlay."""
        return {leg.game for leg in self.legs}

    @property
    def average_confidence(self) -> float:
        """Average confidence across legs."""
        return sum(leg.confidence for leg in self.legs) / len(self.legs)

    def __str__(self) -> str:
        return f"{self.size}-Leg Parlay ({self.american_odds:+d}, {self.combined_confidence:.0%} confidence)"


class NFLParlayMaker:
    """Builds optimized parlays from NFL betting edges."""

    def __init__(self):
        self.edges: List[BettingEdge] = []

    def load_edges_from_week(self, week: int) -> List[BettingEdge]:
        """Fetch edges from auto_weekly_analyzer.py."""
        print(f"\n🔍 Fetching NFL edges for Week {week}...")

        try:
            # Run analyzer with JSON output
            result = subprocess.run(
                ['python', 'auto_weekly_analyzer.py', '--week', str(week), '--json'],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                print(f"❌ Failed to fetch edges: {result.stderr}")
                return []

            # Parse JSON
            data = json.loads(result.stdout)

            edges = []
            for edge_data in data.get('top_edges', []):
                edge = BettingEdge(
                    game=edge_data['game'],
                    edge_type=edge_data['edge_type'],
                    pick=edge_data['pick'],
                    confidence=edge_data['confidence'],
                    edge_size=edge_data['edge_size'],
                    reasoning=edge_data.get('reasoning', ''),
                    odds=-110  # Default, could be enhanced later
                )
                edges.append(edge)

            print(f"✅ Found {len(edges)} betting edges")
            self.edges = edges
            return edges

        except subprocess.TimeoutExpired:
            print("❌ Analyzer timed out")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse analyzer output: {e}")
            return []
        except Exception as e:
            print(f"❌ Error fetching edges: {e}")
            return []

    def filter_edges(
        self,
        min_confidence: float = 0.60,
        edge_sizes: Optional[List[str]] = None,
        bet_types: Optional[List[str]] = None
    ) -> List[BettingEdge]:
        """Filter edges by criteria."""
        filtered = self.edges

        # Filter by confidence
        filtered = [e for e in filtered if e.confidence >= min_confidence]

        # Filter by edge size
        if edge_sizes:
            filtered = [e for e in filtered if e.edge_size in edge_sizes]

        # Filter by bet type
        if bet_types:
            filtered = [e for e in filtered if e.edge_type in bet_types]

        return filtered

    def generate_parlays(
        self,
        edges: List[BettingEdge],
        min_legs: int = 2,
        max_legs: int = 5,
        avoid_same_game: bool = True
    ) -> List[Parlay]:
        """Generate all possible parlay combinations."""
        all_parlays = []

        for num_legs in range(min_legs, max_legs + 1):
            # Generate all combinations of this size
            for combo in combinations(edges, num_legs):
                parlay = Parlay(list(combo))

                # Skip if same game appears twice (correlated)
                if avoid_same_game and len(parlay.games_involved) < parlay.size:
                    continue

                all_parlays.append(parlay)

        return all_parlays

    def rank_parlays(
        self,
        parlays: List[Parlay],
        sort_by: str = 'ev'
    ) -> List[Parlay]:
        """Rank parlays by specified criteria."""
        if sort_by == 'ev':
            return sorted(parlays, key=lambda p: p.expected_value, reverse=True)
        elif sort_by == 'confidence':
            return sorted(parlays, key=lambda p: p.combined_confidence, reverse=True)
        elif sort_by == 'payout':
            return sorted(parlays, key=lambda p: p.combined_odds, reverse=True)
        else:
            return parlays

    def generate_report(
        self,
        week: int,
        parlays: List[Parlay],
        top_n: int = 10
    ) -> str:
        """Generate human-readable parlay report."""
        report = []
        report.append("=" * 80)
        report.append(f"🎯 NFL PARLAY MAKER - WEEK {week}")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Parlays Generated: {len(parlays)}")
        report.append("")

        if not parlays:
            report.append("❌ No parlays generated!")
            report.append("\nPossible reasons:")
            report.append("  - Not enough edges found")
            report.append("  - Filters too restrictive")
            report.append("  - No games with referee assignments yet")
            return "\n".join(report)

        # Show top parlays
        report.append(f"🏆 TOP {min(top_n, len(parlays))} PARLAYS (Ranked by Expected Value)")
        report.append("")

        for i, parlay in enumerate(parlays[:top_n], 1):
            report.append(f"{i}. {parlay}")
            report.append(f"   Expected Value: ${parlay.expected_value:.2f} per $1 wagered")
            report.append(f"   $100 bet → Win ${(parlay.combined_odds - 1) * 100:.2f} ({parlay.combined_confidence:.1%} probability)")
            report.append("")

            for j, leg in enumerate(parlay.legs, 1):
                report.append(f"   Leg {j}: {leg.pick}")
                report.append(f"          {leg.edge_type} | {leg.confidence:.0%} confidence | {leg.edge_size}")
                if leg.reasoning:
                    report.append(f"          💡 {leg.reasoning}")
            report.append("")
            report.append("-" * 80)
            report.append("")

        # Statistics
        report.append("=" * 80)
        report.append("📊 PARLAY STATISTICS")
        report.append("=" * 80)
        report.append("")

        by_size = {}
        for parlay in parlays:
            by_size[parlay.size] = by_size.get(parlay.size, 0) + 1

        report.append("Parlay Distribution:")
        for size in sorted(by_size.keys()):
            report.append(f"  {size}-leg: {by_size[size]} parlays")
        report.append("")

        avg_ev = sum(p.expected_value for p in parlays) / len(parlays)
        avg_conf = sum(p.combined_confidence for p in parlays) / len(parlays)

        report.append(f"Average Expected Value: ${avg_ev:.2f}")
        report.append(f"Average Combined Confidence: {avg_conf:.1%}")
        report.append("")

        # Recommendations
        report.append("=" * 80)
        report.append("💡 RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")

        best_ev = parlays[0] if parlays else None
        if best_ev:
            report.append(f"✅ Best Value: {best_ev.size}-leg parlay")
            report.append(f"   EV: ${best_ev.expected_value:.2f}, Confidence: {best_ev.combined_confidence:.1%}")
            report.append("")

        safe_parlays = [p for p in parlays if p.combined_confidence >= 0.50]
        if safe_parlays:
            report.append(f"✅ Highest Probability Parlays: {len(safe_parlays)} with >50% hit rate")
            report.append("")

        report.append("=" * 80)
        report.append("⚠️  DISCLAIMER")
        report.append("=" * 80)
        report.append("")
        report.append("Parlays are high-risk, high-reward bets.")
        report.append("ALL legs must hit for the parlay to win.")
        report.append("Only bet what you can afford to lose.")
        report.append("Gamble responsibly.")
        report.append("")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Generate optimized NFL parlays from betting edges"
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="NFL week number (1-18)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Minimum edge confidence (default: 0.60)"
    )
    parser.add_argument(
        "--min-legs",
        type=int,
        default=2,
        help="Minimum parlay legs (default: 2)"
    )
    parser.add_argument(
        "--max-legs",
        type=int,
        default=5,
        help="Maximum parlay legs (default: 5)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show top N parlays (default: 10)"
    )
    parser.add_argument(
        "--sort-by",
        choices=['ev', 'confidence', 'payout'],
        default='ev',
        help="Sort parlays by (default: ev)"
    )
    parser.add_argument(
        "--edge-sizes",
        nargs='+',
        choices=['MASSIVE', 'LARGE', 'MEDIUM'],
        help="Filter by edge sizes (default: all)"
    )
    parser.add_argument(
        "--allow-same-game",
        action="store_true",
        help="Allow multiple bets from same game (not recommended)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save report to file (optional)"
    )

    args = parser.parse_args()

    # Create parlay maker
    maker = NFLParlayMaker()

    # Load edges from analyzer
    edges = maker.load_edges_from_week(args.week)

    if not edges:
        print("\n❌ No edges found! Cannot generate parlays.")
        print("\nTry:")
        print("  1. Run auto_weekly_analyzer.py --week", args.week, "first")
        print("  2. Check if referee assignments are posted (usually Thursday)")
        print("  3. Lower --min-confidence threshold")
        sys.exit(1)

    # Filter edges
    filtered = maker.filter_edges(
        min_confidence=args.min_confidence,
        edge_sizes=args.edge_sizes
    )

    print(f"✅ {len(filtered)} edges passed filters")

    if len(filtered) < args.min_legs:
        print(f"\n❌ Not enough edges ({len(filtered)}) to make {args.min_legs}-leg parlays!")
        print(f"\nTry lowering --min-confidence or --min-legs")
        sys.exit(1)

    # Generate parlays
    print(f"\n🔨 Generating {args.min_legs}-{args.max_legs} leg parlays...")
    parlays = maker.generate_parlays(
        filtered,
        min_legs=args.min_legs,
        max_legs=args.max_legs,
        avoid_same_game=not args.allow_same_game
    )

    print(f"✅ Generated {len(parlays)} parlay combinations")

    # Rank parlays
    ranked = maker.rank_parlays(parlays, sort_by=args.sort_by)

    # Generate report
    report = maker.generate_report(args.week, ranked, top_n=args.top)

    # Print to console
    print("\n" + report)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n💾 Report saved to: {args.output}")

    print("\n" + "=" * 80)
    print("✅ PARLAY GENERATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
