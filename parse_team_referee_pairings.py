#!/usr/bin/env python3
"""
Parse Team-Referee Pairing Data from Conspiracy Reports
=========================================================
Extracts team-specific referee tendencies from the goldmine
of data in reports/referee_conspiracy/*.md files.

This is the REAL edge - not just referee tendencies overall,
but how each referee treats SPECIFIC TEAMS.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamRefereeParser:
    """Parse team-specific referee autopsy reports."""

    def __init__(self, reports_dir: str = "reports/referee_conspiracy"):
        self.reports_dir = Path(reports_dir)
        self.team_referee_data: Dict[str, Dict[str, Any]] = {}

    def parse_all_teams(self) -> Dict[str, Dict[str, Any]]:
        """Parse all team referee reports."""
        report_files = list(self.reports_dir.glob("*.md"))

        # Filter out duplicates (remove " (copy).md" files)
        report_files = [f for f in report_files if "(copy)" not in f.name]

        logger.info(f"Found {len(report_files)} team reports")

        for report_file in report_files:
            team = report_file.stem  # e.g., "KC"
            logger.info(f"Parsing {team}...")

            try:
                data = self.parse_team_report(report_file)
                self.team_referee_data[team] = data
            except Exception as e:
                logger.error(f"Error parsing {team}: {e}")

        return self.team_referee_data

    def parse_team_report(self, report_path: Path) -> Dict[str, Any]:
        """Parse individual team report."""
        with open(report_path, 'r') as f:
            content = f.read()

        team = report_path.stem

        # Extract style impact section (team-specific referee stats)
        style_section = self._extract_section(content, "## Style Impact")

        referee_stats = {}
        if style_section:
            referee_stats = self._parse_style_impact(style_section)

        # Extract narrative correlations (specific game events)
        narrative_section = self._extract_section(content, "## Narrative Correlations")

        narratives = []
        if narrative_section:
            narratives = self._parse_narratives(narrative_section)

        # Extract broadcast behavior
        broadcast_section = self._extract_section(content, "## Broadcast Behavior")

        broadcast_stats = {}
        if broadcast_section:
            broadcast_stats = self._parse_broadcast_behavior(broadcast_section)

        return {
            'team': team,
            'referee_stats': referee_stats,
            'narratives': narratives,
            'broadcast_behavior': broadcast_stats,
        }

    def _extract_section(self, content: str, header: str) -> str:
        """Extract section between header and next ##."""
        pattern = f"{re.escape(header)}(.*?)(?=##|$)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _parse_style_impact(self, section: str) -> Dict[str, Dict[str, float]]:
        """
        Parse style impact lines like:
        - Carl Cheffers (overtime_frequency_gt_15pct): 13 games, avg margin +3.4,
          penalties on team 7.3, penalty diff +0.8, odds delta -3.0, overtime rate 8.62%
        """
        referee_stats = {}

        # Pattern to match referee lines
        pattern = r"- ([^(]+)\s*\([^)]+\):\s*(\d+)\s*games,\s*avg margin\s*([-+]?\d+\.?\d*),\s*penalties on team\s*(\d+\.?\d*),\s*penalty diff\s*([-+]?\d+\.?\d*),\s*odds delta\s*([-+]?\d+\.?\d*),\s*overtime rate\s*(\d+\.?\d*)%"

        for match in re.finditer(pattern, section):
            ref_name = match.group(1).strip()
            games = int(match.group(2))
            avg_margin = float(match.group(3))
            penalties = float(match.group(4))
            penalty_diff = float(match.group(5))
            odds_delta = float(match.group(6))
            ot_rate = float(match.group(7))

            referee_stats[ref_name] = {
                'games': games,
                'avg_margin': avg_margin,
                'penalties_on_team': penalties,
                'penalty_diff': penalty_diff,
                'odds_delta': odds_delta,
                'overtime_rate': ot_rate,
            }

        return referee_stats

    def _parse_narratives(self, section: str) -> List[Dict[str, Any]]:
        """
        Parse narrative lines like:
        - Week 7 2018 vs CIN: surge of +29.3 points with Bill Vinovich stepping in (final margin +35.0).
        """
        narratives = []

        # Pattern for surge/collapse lines
        pattern = r"- Week\s+(\d+)\s+(\d+)\s+vs\s+([A-Z]+):\s+(surge|collapse)\s+of\s+([-+]?\d+\.?\d*)\s+points\s+with\s+([^(]+)\s+stepping in\s+\(final margin\s+([-+]?\d+\.?\d*)\)"

        for match in re.finditer(pattern, section):
            week = int(match.group(1))
            year = int(match.group(2))
            opponent = match.group(3)
            event_type = match.group(4)  # "surge" or "collapse"
            point_delta = float(match.group(5))
            referee = match.group(6).strip()
            final_margin = float(match.group(7))

            narratives.append({
                'week': week,
                'year': year,
                'opponent': opponent,
                'type': event_type,
                'point_delta': point_delta,
                'referee': referee,
                'final_margin': final_margin,
            })

        # Pattern for overtime gut punch lines
        pattern_ot = r"- Week\s+(\d+)\s+(\d+)\s+vs\s+([A-Z]+):\s+overtime gut punch\s+([-+]?\d+\.?\d*)\s+under\s+(.+)"

        for match in re.finditer(pattern_ot, section):
            week = int(match.group(1))
            year = int(match.group(2))
            opponent = match.group(3)
            margin = float(match.group(4))
            referee = match.group(5).strip().rstrip('.')

            narratives.append({
                'week': week,
                'year': year,
                'opponent': opponent,
                'type': 'overtime',
                'point_delta': margin,
                'referee': referee,
                'final_margin': margin,
            })

        return narratives

    def _parse_broadcast_behavior(self, section: str) -> Dict[str, Dict[str, float]]:
        """
        Parse broadcast behavior like:
        - Primetime: 6.1 penalties on team, 48.7 total points
        """
        broadcast_stats = {}

        pattern = r"- ([^:]+):\s*(\d+\.?\d*)\s*penalties on team,\s*(\d+\.?\d*)\s*total points"

        for match in re.finditer(pattern, section):
            window = match.group(1).strip()
            penalties = float(match.group(2))
            total_points = float(match.group(3))

            broadcast_stats[window] = {
                'penalties': penalties,
                'total_points': total_points,
            }

        return broadcast_stats

    def save_to_json(self, output_path: str):
        """Save parsed data to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.team_referee_data, f, indent=2)

        logger.info(f"Saved team-referee data to {output_file}")

    def get_referee_team_bias(
        self, referee: str, team: str
    ) -> Dict[str, float]:
        """
        Get specific referee bias for a team.

        Returns dict with avg_margin, penalties, ot_rate, etc.
        """
        if team not in self.team_referee_data:
            return {}

        ref_stats = self.team_referee_data[team]['referee_stats']

        if referee not in ref_stats:
            return {}

        return ref_stats[referee]

    def detect_team_referee_edges(
        self,
        referee: str,
        home_team: str,
        away_team: str,
        spread: float,
        total: float,
    ) -> List[Dict[str, Any]]:
        """
        Detect edges based on team-referee pairings.

        This is THE GOLD - combining:
        - Referee's overall tendencies
        - Referee's bias toward specific teams
        - Historical game narratives
        """
        edges = []

        # Get home team's history with this referee
        home_stats = self.get_referee_team_bias(referee, home_team)
        away_stats = self.get_referee_team_bias(referee, away_team)

        # EDGE 1: Strong home team bias with this ref
        if home_stats and home_stats.get('avg_margin', 0) > 7.0:
            games = home_stats.get('games', 0)
            margin = home_stats['avg_margin']

            if games >= 5:  # Need sufficient sample
                edges.append({
                    'type': 'SPREAD',
                    'pick': 'HOME',
                    'confidence': min(0.80, margin / 15.0),
                    'reason': f"{referee} favors {home_team} by +{margin:.1f} pts ({games} games). Historical bias.",
                    'edge_size': 'LARGE' if margin > 10 else 'MEDIUM',
                    'signal': 'TEAM_REF_HOME_BIAS',
                })

                # Also bet home ML if they're underdogs
                if spread > 0:
                    edges.append({
                        'type': 'MONEYLINE',
                        'pick': 'HOME',
                        'confidence': min(0.75, margin / 12.0),
                        'reason': f"{home_team} underdog but {referee} +{margin:.1f} bias. HUGE VALUE.",
                        'edge_size': 'MASSIVE',
                        'signal': 'TEAM_REF_UNDERDOG_VALUE',
                    })

        # EDGE 2: Strong away team bias with this ref
        if away_stats and away_stats.get('avg_margin', 0) < -7.0:
            games = away_stats.get('games', 0)
            margin = away_stats['avg_margin']

            if games >= 5:
                edges.append({
                    'type': 'SPREAD',
                    'pick': 'AWAY',
                    'confidence': min(0.80, abs(margin) / 15.0),
                    'reason': f"{referee} favors {away_team} opponent by {margin:.1f} pts ({games} games). Bet away.",
                    'edge_size': 'LARGE' if abs(margin) > 10 else 'MEDIUM',
                    'signal': 'TEAM_REF_AWAY_BIAS',
                })

        # EDGE 3: Referee causes lower scoring for home team
        if home_stats and home_stats.get('penalties_on_team', 0) > 7.0:
            penalties = home_stats['penalties_on_team']
            games = home_stats.get('games', 0)

            if games >= 5 and total > 45:
                edges.append({
                    'type': 'TOTAL',
                    'pick': 'UNDER',
                    'confidence': 0.65,
                    'reason': f"{referee} calls {penalties:.1f} penalties on {home_team}. Slows game down.",
                    'edge_size': 'MEDIUM',
                    'signal': 'TEAM_REF_HIGH_PENALTIES_UNDER',
                })

        # EDGE 4: Referee causes higher OT rate for this team
        if home_stats and home_stats.get('overtime_rate', 0) > 10.0:
            ot_rate = home_stats['overtime_rate']
            games = home_stats.get('games', 0)

            if games >= 5 and total < 46:
                edges.append({
                    'type': 'TOTAL',
                    'pick': 'OVER',
                    'confidence': min(0.75, ot_rate / 15.0),
                    'reason': f"{referee} + {home_team} = {ot_rate:.1f}% OT rate. Overtime adds points.",
                    'edge_size': 'LARGE',
                    'signal': 'TEAM_REF_OVERTIME_SURGE',
                })

        # EDGE 5: Historical narrative matches current game
        if home_team in self.team_referee_data:
            narratives = self.team_referee_data[home_team]['narratives']

            # Find surges with this referee
            surges = [n for n in narratives if n['referee'] == referee and n['type'] == 'surge']

            if len(surges) >= 2:  # Multiple historical surges
                avg_surge = sum(n['point_delta'] for n in surges) / len(surges)

                if avg_surge > 20:
                    edges.append({
                        'type': 'SPREAD',
                        'pick': 'HOME',
                        'confidence': 0.70,
                        'reason': f"{home_team} + {referee} = historical surges (avg +{avg_surge:.1f} pts). Pattern.",
                        'edge_size': 'LARGE',
                        'signal': 'TEAM_REF_SURGE_PATTERN',
                    })

        return edges


def main():
    """Parse all team-referee reports and save to JSON."""
    parser = TeamRefereeParser()

    print("=" * 70)
    print("PARSING TEAM-REFEREE CONSPIRACY REPORTS")
    print("=" * 70)

    data = parser.parse_all_teams()

    print(f"\n✅ Parsed {len(data)} teams")

    # Save to JSON
    parser.save_to_json("data/team_referee_pairings.json")

    # Show sample
    print("\n" + "=" * 70)
    print("SAMPLE: Kansas City Chiefs")
    print("=" * 70)

    kc_data = data.get('KC', {})

    print("\n📊 Referee Stats for KC:")
    for ref, stats in list(kc_data['referee_stats'].items())[:5]:
        print(f"\n{ref}:")
        print(f"  Games: {stats['games']}")
        print(f"  Avg Margin: {stats['avg_margin']:+.1f}")
        print(f"  Penalties: {stats['penalties_on_team']:.1f}")
        print(f"  OT Rate: {stats['overtime_rate']:.1f}%")

    print("\n📖 Historical Narratives:")
    for narrative in kc_data['narratives'][:5]:
        print(f"\n  {narrative['year']} Week {narrative['week']} vs {narrative['opponent']}:")
        print(f"    Type: {narrative['type']}")
        print(f"    Delta: {narrative['point_delta']:+.1f}")
        print(f"    Referee: {narrative['referee']}")

    # Test edge detection
    print("\n" + "=" * 70)
    print("EDGE DETECTION TEST")
    print("=" * 70)
    print("\nGame: Dolphins @ Chiefs, Spread: KC -7.5, Total: 48.5")
    print("Referee: Brad Rogers\n")

    edges = parser.detect_team_referee_edges("Brad Rogers", "KC", "MIA", -7.5, 48.5)

    for edge in edges:
        print(f"✅ {edge['type']} {edge['pick']}")
        print(f"   Confidence: {edge['confidence']:.0%}")
        print(f"   Reason: {edge['reason']}")
        print(f"   Edge Size: {edge['edge_size']}")
        print(f"   Signal: {edge['signal']}\n")

    print("=" * 70)
    print("✅ DONE - Team-referee data saved!")
    print("=" * 70)


if __name__ == "__main__":
    main()
