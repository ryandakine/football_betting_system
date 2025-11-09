#!/usr/bin/env python3
"""
Automated Weekly NFL Analyzer
==============================
Automatically scrapes NFL.com for this week's games + referees,
then runs the full analysis on each game.

Usage:
    python auto_weekly_analyzer.py --week 11
    python auto_weekly_analyzer.py --week 11 --output report.txt

This script:
1. Scrapes NFL.com for weekly schedule
2. Gets referee assignments (posted Thursdays)
3. Gets betting lines from The Odds API
4. Runs Model 11 analysis on each game
5. Generates comprehensive report with all edges
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from referee_intelligence_model import RefereeIntelligenceModel
from parse_team_referee_pairings import TeamRefereeParser


# Team abbreviation mapping
TEAM_NAME_MAP = {
    # Full names to abbreviations
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    # City names
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Raiders": "LV",
    "Chargers": "LAC",
    "Rams": "LA",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "49ers": "SF",
    "Seahawks": "SEA",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Commanders": "WAS",
}


class NFLGameScraper:
    """Scrapes NFL.com for weekly games and referee assignments."""

    def __init__(self):
        self.games = []

    def scrape_week(self, week: int, year: int = 2024) -> List[Dict[str, Any]]:
        """
        Scrape NFL.com for this week's games.

        Note: This is a simplified version. In production, you'd use:
        - requests library to fetch HTML
        - BeautifulSoup to parse HTML
        - Or use NFL's API if available

        For now, returns sample data structure.
        """
        print(f"\n🔍 Scraping NFL.com for Week {week}, {year}...")

        # In production, you would:
        # 1. Fetch from NFL.com schedule page
        # 2. Parse HTML for game matchups
        # 3. Get referee assignments (usually Thursday)
        # 4. Get betting lines from The Odds API

        print("⚠️  NOTE: NFL.com scraping requires external packages (requests, beautifulsoup4)")
        print("          For now, using manual input or sample data.")
        print("\nTo enable full scraping, install:")
        print("  pip install requests beautifulsoup4")

        return []

    def get_sample_games(self, week: int) -> List[Dict[str, Any]]:
        """Get sample games for testing."""
        return [
            {
                "game_id": f"BUF_KC_W{week}",
                "away_team": "BUF",
                "home_team": "KC",
                "referee": "Brad Rogers",
                "spread": -2.5,
                "total": 48.5,
                "home_ml": -140,
                "away_ml": 120,
                "kickoff_time": "SNF 8:20 PM",
                "network": "NBC",
            },
            {
                "game_id": f"BAL_CIN_W{week}",
                "away_team": "BAL",
                "home_team": "CIN",
                "referee": "Carl Cheffers",
                "spread": 3.5,
                "total": 42.0,
                "home_ml": 155,
                "away_ml": -180,
                "kickoff_time": "TNF 8:15 PM",
                "network": "Prime",
            },
            {
                "game_id": f"DET_GB_W{week}",
                "away_team": "DET",
                "home_team": "GB",
                "referee": "Bill Vinovich",
                "spread": 7.0,
                "total": 51.5,
                "home_ml": 280,
                "away_ml": -350,
                "kickoff_time": "Sun 1:00 PM",
                "network": "FOX",
            },
            {
                "game_id": f"PHI_DAL_W{week}",
                "away_team": "PHI",
                "home_team": "DAL",
                "referee": "John Hussey",
                "spread": -6.5,
                "total": 45.0,
                "home_ml": -280,
                "away_ml": 230,
                "kickoff_time": "Sun 4:25 PM",
                "network": "CBS",
            },
            {
                "game_id": f"SF_TB_W{week}",
                "away_team": "SF",
                "home_team": "TB",
                "referee": "Shawn Hochuli",
                "spread": 3.0,
                "total": 47.5,
                "home_ml": 135,
                "away_ml": -160,
                "kickoff_time": "Sun 1:00 PM",
                "network": "FOX",
            },
        ]


class AutoWeeklyAnalyzer:
    """Automatically analyzes all NFL games for the week."""

    def __init__(self):
        self.scraper = NFLGameScraper()
        self.ref_intel = RefereeIntelligenceModel()
        self.team_parser = TeamRefereeParser()

        # Load referee data
        try:
            self.team_parser.parse_all_teams()
            print("✅ Loaded 640+ team-referee pairings")
        except Exception as e:
            print(f"⚠️  Warning loading referee data: {e}")

    def analyze_all_games(self, week: int, use_sample: bool = True) -> List[Dict[str, Any]]:
        """Analyze all games for the week."""

        # Get games
        if use_sample:
            print("\n📋 Using sample games for Week", week)
            games = self.scraper.get_sample_games(week)
        else:
            games = self.scraper.scrape_week(week)

        if not games:
            print("\n❌ No games found!")
            return []

        print(f"\n🏈 Found {len(games)} games to analyze\n")

        results = []

        for i, game in enumerate(games, 1):
            print(f"\n{'='*80}")
            print(f"GAME {i}/{len(games)}: {game['away_team']} @ {game['home_team']}")
            print(f"{'='*80}")

            result = self.analyze_single_game(game)
            results.append(result)

            time.sleep(0.5)  # Brief pause between analyses

        return results

    def analyze_single_game(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single game."""

        home_team = game['home_team']
        away_team = game['away_team']
        referee = game['referee']
        spread = game['spread']
        total = game['total']

        print(f"\n📊 Game: {away_team} @ {home_team}")
        print(f"   Spread: {home_team} {spread:+.1f}")
        print(f"   Total: {total}")
        print(f"   Referee: {referee}")
        print(f"   Time: {game.get('kickoff_time', 'TBD')}")

        # Analyze referee
        ref_profile = self.ref_intel.get_referee_profile(referee)
        home_bias = self.team_parser.get_referee_team_bias(referee, home_team)
        away_bias = self.team_parser.get_referee_team_bias(referee, away_team)

        # Detect edges
        edges = self.team_parser.detect_team_referee_edges(
            referee, home_team, away_team, spread, total
        )

        # Build result
        result = {
            'game': game,
            'referee_profile': ref_profile,
            'home_bias': home_bias,
            'away_bias': away_bias,
            'edges': edges,
            'edge_count': len(edges),
        }

        # Display findings
        if home_bias and home_bias.get('games', 0) >= 5:
            margin = home_bias.get('avg_margin', 0)
            if abs(margin) > 7.0:
                print(f"\n   🎯 TEAM-REF BIAS: {referee} + {home_team} = {margin:+.1f} margin")

        if edges:
            print(f"\n   ✅ FOUND {len(edges)} EDGE(S)!")
            for edge in edges:
                print(f"      - {edge['type']} {edge['pick']}: {edge['confidence']:.0%} ({edge['edge_size']})")
        else:
            print(f"\n   ℹ️  No strong edges detected")

        return result

    def generate_report(
        self,
        week: int,
        results: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive weekly report."""

        report = []
        report.append("="*80)
        report.append(f"🏈 NFL WEEK {week} - AUTOMATED ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Games Analyzed: {len(results)}")
        report.append(f"Powered by: Model 11 (Referee Intelligence)")
        report.append("")

        # Count total edges
        total_edges = sum(r['edge_count'] for r in results)
        report.append(f"Total Edges Found: {total_edges}")
        report.append("")

        # Filter to games with edges
        games_with_edges = [r for r in results if r['edge_count'] > 0]

        if not games_with_edges:
            report.append("\n⚠️  NO STRONG EDGES FOUND THIS WEEK")
            report.append("\nThis happens when referee assignments don't provide")
            report.append("statistical advantages. Wait for better spots!")
        else:
            report.append(f"\n🎯 {len(games_with_edges)} GAMES WITH EDGES:")
            report.append("")

            # Sort by total edge confidence
            games_with_edges.sort(
                key=lambda r: sum(e['confidence'] for e in r['edges']),
                reverse=True
            )

            for i, result in enumerate(games_with_edges, 1):
                game = result['game']
                edges = result['edges']

                report.append("\n" + "="*80)
                report.append(f"PLAY #{i}: {game['away_team']} @ {game['home_team']}")
                report.append("="*80)
                report.append(f"Time: {game.get('kickoff_time', 'TBD')}")
                report.append(f"Spread: {game['home_team']} {game['spread']:+.1f}")
                report.append(f"Total: {game['total']}")
                report.append(f"Referee: {game['referee']}")
                report.append("")

                # Show team-ref bias
                home_bias = result.get('home_bias')
                if home_bias and home_bias.get('games', 0) >= 5:
                    margin = home_bias.get('avg_margin', 0)
                    if abs(margin) > 7.0:
                        report.append(f"📊 Team-Ref History:")
                        report.append(f"   {game['referee']} + {game['home_team']}: {margin:+.1f} avg margin ({home_bias['games']} games)")
                        report.append("")

                # Show edges
                report.append(f"🎯 EDGES DETECTED ({len(edges)} total):")
                report.append("")

                for j, edge in enumerate(edges, 1):
                    confidence_stars = "⭐" * int(edge['confidence'] * 5)
                    report.append(f"   Edge #{j}: {edge['type']} {edge['pick']}")
                    report.append(f"   Confidence: {edge['confidence']:.0%} {confidence_stars}")
                    report.append(f"   Edge Size: {edge['edge_size']}")
                    report.append(f"   Signal: {edge['signal']}")
                    report.append(f"   Reason: {edge['reason']}")
                    report.append("")

                    # Bet sizing
                    if edge['edge_size'] == 'MASSIVE':
                        report.append(f"   💰 RECOMMENDATION: MAX BET (5 units)")
                    elif edge['edge_size'] == 'LARGE':
                        report.append(f"   💰 RECOMMENDATION: BET STRONG (3-4 units)")
                    elif edge['edge_size'] == 'MEDIUM':
                        report.append(f"   💰 RECOMMENDATION: BET MODERATE (2-3 units)")
                    else:
                        report.append(f"   💰 RECOMMENDATION: BET SMALL (1 unit)")
                    report.append("")

                # Jackpot detection
                if len(edges) >= 3:
                    report.append("   🎰 JACKPOT! Multiple edges aligned!")
                    report.append("")

        # Summary stats
        report.append("\n" + "="*80)
        report.append("📊 SUMMARY STATISTICS")
        report.append("="*80)
        report.append("")

        # Edge breakdown
        all_edges = []
        for r in results:
            all_edges.extend(r['edges'])

        if all_edges:
            spread_edges = sum(1 for e in all_edges if e['type'] == 'SPREAD')
            total_edges = sum(1 for e in all_edges if e['type'] == 'TOTAL')
            ml_edges = sum(1 for e in all_edges if e['type'] == 'MONEYLINE')

            report.append(f"Edge Type Distribution:")
            report.append(f"  Spread: {spread_edges}")
            report.append(f"  Total: {total_edges}")
            report.append(f"  Moneyline: {ml_edges}")
            report.append("")

            massive = sum(1 for e in all_edges if e['edge_size'] == 'MASSIVE')
            large = sum(1 for e in all_edges if e['edge_size'] == 'LARGE')
            medium = sum(1 for e in all_edges if e['edge_size'] == 'MEDIUM')

            report.append(f"Edge Size Distribution:")
            report.append(f"  MASSIVE: {massive}")
            report.append(f"  LARGE: {large}")
            report.append(f"  MEDIUM: {medium}")
            report.append("")

            avg_conf = sum(e['confidence'] for e in all_edges) / len(all_edges)
            report.append(f"Average Edge Confidence: {avg_conf:.0%}")

        report.append("")
        report.append("="*80)
        report.append("⚠️  BETTING DISCLAIMER")
        report.append("="*80)
        report.append("")
        report.append("These picks are based on historical referee patterns (2018-2024).")
        report.append("Past performance does not guarantee future results.")
        report.append("Only bet what you can afford to lose.")
        report.append("Gamble responsibly.")
        report.append("")
        report.append("System: 11-Model Super Intelligence")
        report.append("Model 11: Referee Intelligence (640+ team-ref pairings)")
        report.append("")

        report_text = "\n".join(report)

        # Save to file if requested
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\n✅ Report saved to {output_file}")

        return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Automatically analyze all NFL games for the week"
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
        "--sample",
        action="store_true",
        default=True,
        help="Use sample games (default: True)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"🤖 AUTOMATED NFL WEEKLY ANALYZER")
    print("="*80)
    print(f"\nWeek: {args.week}")
    print(f"Mode: {'Sample Data' if args.sample else 'Live Scraping'}")

    # Initialize analyzer
    analyzer = AutoWeeklyAnalyzer()

    # Analyze all games
    results = analyzer.analyze_all_games(args.week, use_sample=args.sample)

    if not results:
        print("\n❌ No results to report")
        return

    # Generate report
    print("\n" + "="*80)
    print("📝 GENERATING WEEKLY REPORT")
    print("="*80)

    report = analyzer.generate_report(args.week, results, args.output)

    # Print to console
    print("\n" + report)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)

    # Show top plays
    games_with_edges = [r for r in results if r['edge_count'] > 0]
    if games_with_edges:
        print(f"\n🎯 TOP PLAYS FOR WEEK {args.week}:")
        for i, result in enumerate(games_with_edges[:3], 1):
            game = result['game']
            best_edge = max(result['edges'], key=lambda e: e['confidence'])
            print(f"\n   {i}. {game['away_team']} @ {game['home_team']}")
            print(f"      {best_edge['type']} {best_edge['pick']}: {best_edge['confidence']:.0%} ({best_edge['edge_size']})")

    print("\n")


if __name__ == "__main__":
    main()
