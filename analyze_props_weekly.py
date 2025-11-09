#!/usr/bin/env python3
"""
Weekly Prop Analyzer
====================
Analyzes player props for the week using Model 12 + Referee Intelligence.

Usage:
    python analyze_props_weekly.py --week 11
    python analyze_props_weekly.py --week 11 --output reports/week11_props.txt

Analyzes:
- QB passing yards/TDs
- RB rushing yards/TDs
- WR receiving yards/TDs
- All other prop types

Factors considered:
- Player historical stats (7 years)
- Matchup vs opponent defense
- Referee impact (penalties = more passing)
- Game script (spread/total)
- Home/away splits
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from prop_intelligence_model import PropIntelligenceModel, PropType, PropPrediction
from referee_intelligence_model import RefereeIntelligenceModel


class WeeklyPropAnalyzer:
    """Analyzes all props for a given week."""

    def __init__(self):
        self.prop_model = PropIntelligenceModel()
        self.ref_model = RefereeIntelligenceModel()

    def analyze_week(self, week: int) -> List[PropPrediction]:
        """
        Analyze all props for the week.

        In production:
        1. Scrape DraftKings/FanDuel for this week's prop lines
        2. For each prop, run prediction
        3. Return only high-confidence picks
        """

        print(f"\n{'='*80}")
        print(f"🏈 ANALYZING WEEK {week} PLAYER PROPS")
        print(f"{'='*80}\n")

        # Sample props for Week 11
        sample_props = self._get_sample_props(week)

        print(f"Found {len(sample_props)} props to analyze\n")

        predictions = []

        for prop in sample_props:
            # Get referee profile
            referee_profile = self.ref_model.get_referee_profile(prop['referee'])

            # Make prediction
            pred = self.prop_model.predict_prop(
                player_name=prop['player'],
                prop_type=PropType(prop['prop_type']),
                line=prop['line'],
                opponent_team=prop['opponent'],
                is_home=prop['is_home'],
                spread=prop['spread'],
                total=prop['total'],
                referee_name=prop['referee'],
                referee_profile=referee_profile,
            )

            predictions.append(pred)

        return predictions

    def _get_sample_props(self, week: int) -> List[Dict[str, Any]]:
        """Get sample props for testing."""
        return [
            {
                'player': 'Patrick Mahomes',
                'prop_type': 'passing_yards',
                'line': 275.5,
                'opponent': 'BUF',
                'is_home': True,
                'spread': -2.5,
                'total': 48.5,
                'referee': 'Brad Rogers',
                'game': 'BUF @ KC',
            },
            {
                'player': 'Josh Allen',
                'prop_type': 'passing_yards',
                'line': 255.5,
                'opponent': 'KC',
                'is_home': False,
                'spread': 2.5,
                'total': 48.5,
                'referee': 'Brad Rogers',
                'game': 'BUF @ KC',
            },
            {
                'player': 'Patrick Mahomes',
                'prop_type': 'passing_tds',
                'line': 1.5,
                'opponent': 'BUF',
                'is_home': True,
                'spread': -2.5,
                'total': 48.5,
                'referee': 'Brad Rogers',
                'game': 'BUF @ KC',
            },
            {
                'player': 'Christian McCaffrey',
                'prop_type': 'rushing_yards',
                'line': 95.5,
                'opponent': 'TB',
                'is_home': False,
                'spread': 3.0,
                'total': 47.5,
                'referee': 'Shawn Hochuli',
                'game': 'SF @ TB',
            },
            {
                'player': 'Tyreek Hill',
                'prop_type': 'receiving_yards',
                'line': 75.5,
                'opponent': 'LV',
                'is_home': True,
                'spread': -7.5,
                'total': 44.5,
                'referee': 'John Parry',
                'game': 'LV @ MIA',
            },
            {
                'player': 'Travis Kelce',
                'prop_type': 'receptions',
                'line': 5.5,
                'opponent': 'BUF',
                'is_home': True,
                'spread': -2.5,
                'total': 48.5,
                'referee': 'Brad Rogers',
                'game': 'BUF @ KC',
            },
        ]

    def generate_report(
        self,
        week: int,
        predictions: List[PropPrediction],
        output_path: str = None,
        min_confidence: float = 0.60
    ) -> str:
        """Generate report of top prop picks."""

        # Filter to high-confidence picks
        top_picks = [
            p for p in predictions
            if p.confidence >= min_confidence
        ]

        # Sort by confidence
        top_picks.sort(key=lambda p: p.confidence, reverse=True)

        report = []
        report.append("="*80)
        report.append(f"🎯 WEEK {week} - TOP PROP PICKS")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: Model 12 (Prop Intelligence)")
        report.append(f"Min Confidence: {min_confidence:.0%}")
        report.append("")
        report.append(f"Total Props Analyzed: {len(predictions)}")
        report.append(f"High-Confidence Picks: {len(top_picks)}")
        report.append("")

        if not top_picks:
            report.append("⚠️  NO HIGH-CONFIDENCE PROPS FOUND")
            report.append("\nNo props met the minimum confidence threshold.")
            report.append("Try lowering --min-confidence or wait for better spots.")
        else:
            report.append(f"🎯 {len(top_picks)} PROPS TO BET:")
            report.append("")

            for i, pick in enumerate(top_picks, 1):
                report.append("\n" + "="*80)
                report.append(f"PROP #{i}: {pick.player_name} - {pick.prop_type.value.replace('_', ' ').title()}")
                report.append("="*80)

                # Display
                prop_display = f"{pick.prop_type.value.replace('_', ' ').title()}"
                report.append(f"Line: {pick.line}")
                report.append(f"Prediction: {pick.prediction:.1f}")
                report.append(f"Pick: {pick.pick}")
                report.append("")

                # Confidence & Edge
                stars = "⭐" * int(pick.confidence * 5)
                report.append(f"Confidence: {pick.confidence:.0%} {stars}")
                report.append(f"Edge Size: {pick.edge_size}")
                report.append(f"Edge: {abs(pick.prediction - pick.line):.1f} {pick.prop_type.value.split('_')[-1]}")
                report.append("")

                # Breakdown
                report.append("Breakdown:")
                report.append(f"  Base Projection: {pick.base_prediction:.1f}")
                if abs(pick.matchup_adjustment) > 1:
                    report.append(f"  Matchup: {pick.matchup_adjustment:+.1f}")
                if abs(pick.referee_adjustment) > 1:
                    report.append(f"  Referee Impact: {pick.referee_adjustment:+.1f}")
                if abs(pick.gamescript_adjustment) > 1:
                    report.append(f"  Game Script: {pick.gamescript_adjustment:+.1f}")
                if abs(pick.venue_adjustment) > 1:
                    report.append(f"  Venue: {pick.venue_adjustment:+.1f}")
                report.append("")

                # Reasoning
                report.append("Why we like it:")
                for reason in pick.reasoning:
                    report.append(f"  • {reason}")
                report.append("")

                # Bet sizing
                if pick.edge_size == 'MASSIVE':
                    report.append(f"💰 RECOMMENDATION: MAX BET (5 units)")
                elif pick.edge_size == 'LARGE':
                    report.append(f"💰 RECOMMENDATION: BET STRONG (3-4 units)")
                elif pick.edge_size == 'MEDIUM':
                    report.append(f"💰 RECOMMENDATION: BET MODERATE (2-3 units)")
                else:
                    report.append(f"💰 RECOMMENDATION: BET SMALL (1 unit)")

        # Summary
        report.append("\n" + "="*80)
        report.append("📊 SUMMARY")
        report.append("="*80)
        report.append("")

        if top_picks:
            # Group by prop type
            by_type = {}
            for pick in top_picks:
                prop_type = pick.prop_type.value
                if prop_type not in by_type:
                    by_type[prop_type] = []
                by_type[prop_type].append(pick)

            report.append("Props by Type:")
            for prop_type, picks in by_type.items():
                avg_conf = sum(p.confidence for p in picks) / len(picks)
                report.append(f"  {prop_type.replace('_', ' ').title()}: {len(picks)} picks (avg {avg_conf:.0%} confidence)")

            report.append("")
            avg_edge = sum(abs(p.prediction - p.line) for p in top_picks) / len(top_picks)
            report.append(f"Average Edge: {avg_edge:.1f}")

        report.append("")
        report.append("="*80)
        report.append("⚠️  DISCLAIMER")
        report.append("="*80)
        report.append("")
        report.append("Props are predictions based on historical data.")
        report.append("Players can overperform or underperform any given week.")
        report.append("Only bet what you can afford to lose.")
        report.append("Gamble responsibly.")
        report.append("")

        report_text = "\n".join(report)

        # Save if requested
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\n✅ Report saved to {output_file}")

        return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Analyze player props for the week"
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="NFL week number (1-18)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.60,
        help="Minimum confidence threshold (default: 0.60)"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = WeeklyPropAnalyzer()

    # Analyze props
    predictions = analyzer.analyze_week(args.week)

    # Generate report
    report = analyzer.generate_report(
        args.week,
        predictions,
        args.output,
        args.min_confidence
    )

    # Print to console
    print("\n" + report)

    print("\n" + "="*80)
    print("✅ PROP ANALYSIS COMPLETE!")
    print("="*80)

    # Show top 3
    top_picks = sorted(
        [p for p in predictions if p.confidence >= args.min_confidence],
        key=lambda p: p.confidence,
        reverse=True
    )

    if top_picks:
        print(f"\n🎯 TOP 3 PROPS FOR WEEK {args.week}:\n")
        for i, pick in enumerate(top_picks[:3], 1):
            print(f"   {i}. {pick.player_name} - {pick.prop_type.value.replace('_', ' ').title()}")
            print(f"      {pick.pick} {pick.line} ({pick.confidence:.0%} confidence, {pick.edge_size})\n")


if __name__ == "__main__":
    main()
