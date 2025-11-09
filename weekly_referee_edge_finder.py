#!/usr/bin/env python3
"""
Weekly Referee Edge Finder
===========================
Analyzes this week's referee assignments to find betting edges.

Usage:
    python weekly_referee_edge_finder.py --week 10

Outputs:
    - Top 5 plays based on referee assignments
    - Confidence ratings for each edge
    - Historical evidence for each play
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from referee_intelligence_model import RefereeIntelligenceModel
from parse_team_referee_pairings import TeamRefereeParser


@dataclass
class GameWithReferee:
    """Game info with referee assignment."""
    game_id: str
    home_team: str
    away_team: str
    referee: str
    spread: float
    total: float
    home_ml: Optional[int] = None
    away_ml: Optional[int] = None
    kickoff_time: Optional[str] = None
    network: Optional[str] = None


@dataclass
class RefereeEdgePlay:
    """A betting play identified by referee intelligence."""
    game: GameWithReferee
    bet_type: str  # SPREAD, TOTAL, MONEYLINE
    pick: str  # HOME, AWAY, OVER, UNDER
    confidence: float
    edge_size: str  # SMALL, MEDIUM, LARGE, MASSIVE
    reasoning: str
    signals: List[str]
    supporting_data: Dict[str, Any]

    def __str__(self) -> str:
        """Format as readable string."""
        game_str = f"{self.game.away_team} @ {self.game.home_team}"
        if self.game.kickoff_time:
            game_str += f" ({self.game.kickoff_time})"

        bet_str = f"{self.bet_type} {self.pick}"
        if self.bet_type == "SPREAD":
            if self.pick == "HOME":
                bet_str += f" {self.game.spread}"
            else:
                bet_str += f" +{abs(self.game.spread)}"
        elif self.bet_type == "TOTAL":
            bet_str += f" {self.game.total}"

        confidence_stars = "⭐" * int(self.confidence * 5)

        return f"""
{'='*80}
🎯 {game_str}
{'='*80}
Referee: {self.game.referee}
Play: {bet_str}
Confidence: {self.confidence:.0%} {confidence_stars}
Edge Size: {self.edge_size}

💰 Reasoning:
{self.reasoning}

📊 Signals: {', '.join(self.signals)}

📈 Supporting Data:
{json.dumps(self.supporting_data, indent=2)}
"""


class WeeklyRefereeEdgeFinder:
    """Finds betting edges based on weekly referee assignments."""

    def __init__(self):
        self.referee_intel = RefereeIntelligenceModel()
        self.team_referee_parser = TeamRefereeParser()

        # Load all team-referee pairing data
        try:
            self.team_referee_parser.parse_all_teams()
            print("✅ Loaded team-referee pairing data for all 32 teams")
        except Exception as e:
            print(f"⚠️  Warning: Could not load team-referee data: {e}")

    def analyze_game(self, game: GameWithReferee) -> List[RefereeEdgePlay]:
        """Analyze a single game for referee edges."""
        plays = []

        # Get general referee edges
        general_edges = self.referee_intel.detect_referee_edges(
            game.referee,
            game.spread,
            game.total,
            game.home_team,
            game.away_team,
        )

        # Get team-specific referee edges
        team_edges = self.team_referee_parser.detect_team_referee_edges(
            game.referee,
            game.home_team,
            game.away_team,
            game.spread,
            game.total,
        )

        # Combine all edges
        all_edges = general_edges + team_edges

        # Convert to RefereeEdgePlay objects
        for edge in all_edges:
            # Get supporting data
            ref_profile = self.referee_intel.get_referee_profile(game.referee)
            home_ref_stats = self.team_referee_parser.get_referee_team_bias(
                game.referee, game.home_team
            )
            away_ref_stats = self.team_referee_parser.get_referee_team_bias(
                game.referee, game.away_team
            )

            supporting_data = {
                'referee_overall': ref_profile or {},
                'referee_vs_home_team': home_ref_stats,
                'referee_vs_away_team': away_ref_stats,
            }

            play = RefereeEdgePlay(
                game=game,
                bet_type=edge['type'],
                pick=edge['pick'],
                confidence=edge['confidence'],
                edge_size=edge['edge_size'],
                reasoning=edge['reason'],
                signals=[edge['signal']],
                supporting_data=supporting_data,
            )

            plays.append(play)

        return plays

    def analyze_week(self, games: List[GameWithReferee]) -> List[RefereeEdgePlay]:
        """Analyze all games for the week."""
        all_plays = []

        for game in games:
            game_plays = self.analyze_game(game)
            all_plays.extend(game_plays)

        # Sort by confidence * edge_size
        edge_size_values = {
            'SMALL': 1.0,
            'MEDIUM': 1.5,
            'LARGE': 2.0,
            'MASSIVE': 3.0,
        }

        all_plays.sort(
            key=lambda p: p.confidence * edge_size_values.get(p.edge_size, 1.0),
            reverse=True
        )

        return all_plays

    def get_top_plays(
        self,
        games: List[GameWithReferee],
        top_n: int = 5
    ) -> List[RefereeEdgePlay]:
        """Get top N plays for the week."""
        all_plays = self.analyze_week(games)
        return all_plays[:top_n]

    def generate_weekly_report(
        self,
        games: List[GameWithReferee],
        week: int,
        output_path: Optional[str] = None
    ) -> str:
        """Generate full weekly report."""
        top_plays = self.get_top_plays(games, top_n=10)

        report = f"""
{'='*80}
🏈 NFL WEEK {week} - REFEREE EDGE REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Games Analyzed: {len(games)}
Edges Found: {len(top_plays)}

MODEL 11: REFEREE INTELLIGENCE - THE CENTERPIECE
This report identifies betting edges based on historical referee tendencies
and team-specific referee bias patterns from 2018-2024.

"""

        if not top_plays:
            report += "\n⚠️  No strong referee edges found this week.\n"
        else:
            report += f"\n🎯 TOP {min(len(top_plays), 10)} PLAYS:\n"
            for i, play in enumerate(top_plays, 1):
                report += f"\n\n#{i} - {play}"

        # Add summary stats
        report += f"""

{'='*80}
📊 SUMMARY STATISTICS
{'='*80}

Edge Distribution:
"""
        # Count edge types
        spread_edges = sum(1 for p in top_plays if p.bet_type == "SPREAD")
        total_edges = sum(1 for p in top_plays if p.bet_type == "TOTAL")
        ml_edges = sum(1 for p in top_plays if p.bet_type == "MONEYLINE")

        report += f"  Spread Edges: {spread_edges}\n"
        report += f"  Total Edges: {total_edges}\n"
        report += f"  Moneyline Edges: {ml_edges}\n"

        # Count edge sizes
        massive_edges = sum(1 for p in top_plays if p.edge_size == "MASSIVE")
        large_edges = sum(1 for p in top_plays if p.edge_size == "LARGE")
        medium_edges = sum(1 for p in top_plays if p.edge_size == "MEDIUM")

        report += f"\nEdge Size Distribution:\n"
        report += f"  MASSIVE: {massive_edges}\n"
        report += f"  LARGE: {large_edges}\n"
        report += f"  MEDIUM: {medium_edges}\n"

        # Average confidence
        if top_plays:
            avg_confidence = sum(p.confidence for p in top_plays) / len(top_plays)
            report += f"\nAverage Confidence: {avg_confidence:.0%}\n"

        report += f"""

{'='*80}
⚠️  DISCLAIMER
{'='*80}
These picks are based on historical referee patterns and should be combined
with other analysis. Past performance does not guarantee future results.
Gamble responsibly.

Powered by Model 11: Referee Intelligence System
Data: 2018-2024 NFL referee assignments (32 teams × 20+ referees)
"""

        # Save to file if requested
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\n✅ Report saved to {output_file}")

        return report


def load_games_from_json(json_path: str) -> List[GameWithReferee]:
    """Load games from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    games = []
    for game_data in data['games']:
        game = GameWithReferee(
            game_id=game_data['game_id'],
            home_team=game_data['home_team'],
            away_team=game_data['away_team'],
            referee=game_data['referee'],
            spread=float(game_data['spread']),
            total=float(game_data['total']),
            home_ml=game_data.get('home_ml'),
            away_ml=game_data.get('away_ml'),
            kickoff_time=game_data.get('kickoff_time'),
            network=game_data.get('network'),
        )
        games.append(game)

    return games


def create_sample_week() -> List[GameWithReferee]:
    """Create sample games for testing."""
    return [
        GameWithReferee(
            game_id="KC_BUF_W10",
            home_team="KC",
            away_team="BUF",
            referee="Brad Rogers",
            spread=-2.5,
            total=48.5,
            home_ml=-140,
            away_ml=120,
            kickoff_time="SNF 8:20 PM",
            network="NBC",
        ),
        GameWithReferee(
            game_id="BAL_CIN_W10",
            home_team="CIN",
            away_team="BAL",
            referee="Carl Cheffers",
            spread=3.5,
            total=42.0,
            home_ml=155,
            away_ml=-180,
            kickoff_time="TNF 8:15 PM",
            network="Prime",
        ),
        GameWithReferee(
            game_id="DET_GB_W10",
            home_team="GB",
            away_team="DET",
            referee="Bill Vinovich",
            spread=7.0,
            total=51.5,
            home_ml=280,
            away_ml=-350,
            kickoff_time="Sun 1:00 PM",
            network="FOX",
        ),
        GameWithReferee(
            game_id="PHI_DAL_W10",
            home_team="DAL",
            away_team="PHI",
            referee="John Hussey",
            spread=-6.5,
            total=45.0,
            home_ml=-280,
            away_ml=230,
            kickoff_time="Sun 4:25 PM",
            network="CBS",
        ),
        GameWithReferee(
            game_id="SF_TB_W10",
            home_team="TB",
            away_team="SF",
            referee="Shawn Hochuli",
            spread=3.0,
            total=47.5,
            home_ml=135,
            away_ml=-160,
            kickoff_time="Sun 1:00 PM",
            network="FOX",
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Find betting edges based on weekly referee assignments"
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="NFL week number"
    )
    parser.add_argument(
        "--games",
        type=str,
        help="Path to JSON file with game data (optional - uses sample if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save report (optional)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top plays to show (default: 5)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"🏈 NFL WEEK {args.week} - REFEREE EDGE FINDER")
    print("="*80 + "\n")

    # Initialize finder
    finder = WeeklyRefereeEdgeFinder()

    # Load games
    if args.games:
        print(f"Loading games from {args.games}...")
        games = load_games_from_json(args.games)
    else:
        print("No game data provided - using sample games for demo...")
        games = create_sample_week()

    print(f"✅ Loaded {len(games)} games\n")

    # Generate report
    report = finder.generate_weekly_report(
        games,
        week=args.week,
        output_path=args.output
    )

    # Print to console
    print(report)


if __name__ == "__main__":
    main()
