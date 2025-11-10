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
import asyncio
import json
import os
import re
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from referee_intelligence_model import RefereeIntelligenceModel
from parse_team_referee_pairings import TeamRefereeParser

# Try to import odds integration
try:
    from nfl_odds_integration import fetch_and_integrate_nfl_odds
    HAS_ODDS_API = True
except ImportError:
    HAS_ODDS_API = False
    print("⚠️  NFL Odds Integration not available")

# Try to import referee assignment fetcher
try:
    from referee_assignment_fetcher import RefereeAssignmentsFetcher
    HAS_REF_FETCHER = True
except ImportError:
    HAS_REF_FETCHER = False
    print("⚠️  Referee Assignment Fetcher not available")


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


def get_nfl_week_dates(week: int, year: int = 2025) -> List[date]:
    """
    Convert NFL week number to list of dates for that week.

    NFL weeks typically run Thursday-Monday:
    - Week 1 starts ~Sept 5 (Thursday night)
    - Each subsequent week is 7 days later
    - Games span Thursday/Friday/Saturday/Sunday/Monday

    Args:
        week: NFL week number (1-18)
        year: NFL season year (default: 2025)

    Returns:
        List of dates for that week (Thursday through Monday)
    """
    # 2025 NFL Season: Week 1 starts Thursday, Sept 4
    # 2024 NFL Season: Week 1 started Thursday, Sept 5
    if year == 2025:
        week1_start = date(2025, 9, 4)  # Thursday, Sept 4, 2025
    elif year == 2024:
        week1_start = date(2024, 9, 5)  # Thursday, Sept 5, 2024
    else:
        # Default estimate: First Thursday of September
        week1_start = date(year, 9, 1)
        while week1_start.weekday() != 3:  # 3 = Thursday
            week1_start += timedelta(days=1)

    # Calculate Thursday of target week
    week_start = week1_start + timedelta(weeks=(week - 1))

    # NFL week spans Thursday-Monday (5 days)
    dates = []
    for day_offset in range(5):
        dates.append(week_start + timedelta(days=day_offset))

    return dates


class NFLGameScraper:
    """Scrapes NFL.com for weekly games and referee assignments."""

    def __init__(self):
        self.games = []
        self.ref_fetcher = RefereeAssignmentsFetcher() if HAS_REF_FETCHER else None

    def fetch_real_games(self, week: int, target_date: Optional[date] = None, year: int = 2025) -> List[Dict[str, Any]]:
        """
        Fetch real NFL games from The Odds API for the entire week.

        Args:
            week: NFL week number (1-18)
            target_date: Optional specific date (if None, fetches all dates for the week)
            year: NFL season year (default: 2025)

        Returns games in the format expected by the analyzer:
        [{
            'game_id': str,
            'away_team': str,
            'home_team': str,
            'referee': str,  # Will be 'TBD' until assignments posted
            'spread': float,
            'total': float,
            'home_ml': int,
            'away_ml': int,
            'kickoff_time': str,
            'network': str,
        }]
        """

        if not HAS_ODDS_API:
            print("❌ Odds API integration not available")
            return []

        print(f"\n🔍 Fetching real NFL games for Week {week}...")

        try:
            # Get all dates for this NFL week (Thursday-Monday)
            if target_date:
                dates_to_fetch = [target_date]
            else:
                dates_to_fetch = get_nfl_week_dates(week, year)
                print(f"   📅 Week {week} dates: {dates_to_fetch[0]} to {dates_to_fetch[-1]}")

            # Fetch odds for all dates in the week
            all_games = []
            seen_games = set()

            for fetch_date in dates_to_fetch:
                # Fetch odds from API for this date
                result = asyncio.run(fetch_and_integrate_nfl_odds(fetch_date))

                if not result or not result.get('odds'):
                    continue

                # Convert odds data to game format
                for odds in result['odds']:
                    # Create unique game key to avoid duplicates (one per bookmaker)
                    game_key = f"{odds.away_team}_{odds.home_team}"
                    if game_key in seen_games:
                        continue
                    seen_games.add(game_key)

                    game = {
                        'game_id': odds.game_id,
                        'away_team': odds.away_team,
                        'home_team': odds.home_team,
                        'referee': 'TBD',  # Updated separately when available
                        'spread': odds.spread_home if odds.spread_home else 0.0,
                        'total': odds.total if odds.total else 0.0,
                        'home_ml': odds.moneyline_home if odds.moneyline_home else 0,
                        'away_ml': odds.moneyline_away if odds.moneyline_away else 0,
                        'kickoff_time': odds.commence_time,
                        'network': 'TBD',
                        'bookmaker': odds.bookmaker,
                    }
                    all_games.append(game)

            print(f"✅ Found {len(all_games)} real games across {len(dates_to_fetch)} dates")

            # Fetch referee assignments and match to games
            if self.ref_fetcher and week:
                print(f"🔍 Fetching referee assignments for Week {week}...")
                all_games = self._add_referee_assignments(all_games, week, year)

            return all_games

        except Exception as e:
            print(f"❌ Error fetching real games: {e}")
            raise RuntimeError(f"Failed to fetch real NFL games: {e}") from e

    def _add_referee_assignments(self, games: List[Dict[str, Any]], week: int, year: int = 2025) -> List[Dict[str, Any]]:
        """Fetch referee assignments from Football Zebras and match to games."""
        try:

            # Fetch referee assignments
            ref_data = self.ref_fetcher.fetch_week(year, week)

            if not ref_data or not ref_data.get('ok'):
                print("⚠️  Referee assignments not yet posted (usually available Thursday)")
                return games

            ref_assignments = ref_data.get('games', {})
            print(f"✅ Found {len(ref_assignments)} referee assignments")

            # Match referees to games
            matched = 0
            for game in games:
                away = game['away_team']
                home = game['home_team']

                # Try different matchup formats
                matchup_keys = [
                    f"{away} @ {home}",
                    f"{away}@{home}",
                    f"{away} at {home}",
                ]

                for key in matchup_keys:
                    if key in ref_assignments:
                        game['referee'] = ref_assignments[key]
                        matched += 1
                        break
                    # Try case-insensitive
                    for ref_key, ref_name in ref_assignments.items():
                        if ref_key.upper().replace('@', ' @ ') == f"{away} @ {home}".upper():
                            game['referee'] = ref_name
                            matched += 1
                            break

            print(f"✅ Matched {matched}/{len(games)} games with referee assignments")

            if matched < len(games):
                print(f"⚠️  {len(games) - matched} games still have referee='TBD'")

            return games

        except Exception as e:
            print(f"⚠️  Error fetching referee assignments: {e}")
            print("   Continuing with referee='TBD' for all games")
            return games


class AutoWeeklyAnalyzer:
    """Automatically analyzes all NFL games for the week."""

    def __init__(self):
        self.scraper = NFLGameScraper()
        self.ref_intel = RefereeIntelligenceModel()
        self.team_parser = TeamRefereeParser()

        # Sentiment analysis
        try:
            from ai_council_with_sentiment import SentimentFeatureExtractor
            from intelligent_model_selector import IntelligentModelSelector
            self.sentiment_extractor = SentimentFeatureExtractor()
            self.model_selector = IntelligentModelSelector()
            self.has_sentiment = True
            print("✅ Sentiment analysis ENABLED")
        except ImportError as e:
            self.sentiment_extractor = None
            self.model_selector = None
            self.has_sentiment = False
            print(f"⚠️  Sentiment analysis DISABLED: {e}")

        # Load referee data
        try:
            self.team_parser.parse_all_teams()
            print("✅ Loaded 640+ team-referee pairings")
        except Exception as e:
            print(f"⚠️  Warning loading referee data: {e}")

    def analyze_all_games(self, week: int, target_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """Analyze all games for the week using REAL data only."""

        # Fetch real games from Odds API
        print(f"\n🔍 Fetching real games for Week {week}...")
        games = self.scraper.fetch_real_games(week, target_date)

        if not games:
            print("\n❌ No games found from Odds API!")
            print("Please check:")
            print("  1. THE_ODDS_API_KEY is set in .env")
            print("  2. Network connectivity")
            print("  3. API quota/limits")
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

        # Detect edges from BOTH sources
        # 1. Team-specific referee bias (from conspiracy reports)
        team_edges = self.team_parser.detect_team_referee_edges(
            referee, home_team, away_team, spread, total
        )

        # 2. General referee patterns (ALL bet types: spread, total, ML, 1H, team totals)
        ref_edges = self.ref_intel.detect_referee_edges(
            referee, spread, total, home_team, away_team
        )

        # Combine all edges
        edges = team_edges + ref_edges

        # Extract sentiment if available
        sentiment_data = None
        model_recommendation = None
        if self.has_sentiment and self.sentiment_extractor:
            sentiment_data = self.sentiment_extractor.extract_game_sentiment({
                'home_team': home_team,
                'away_team': away_team,
                'game_id': f"{away_team}_{home_team}_{game.get('game_date', 'TBD')}"
            })

            # Get referee edge strength for model selection
            ref_edge_strength = max([e['confidence'] for e in edges], default=0.5)

            # Use intelligent model selector
            if self.model_selector:
                model_recommendation = self.model_selector.select_model(
                    game=f"{away_team} @ {home_team}",
                    referee_edge=ref_edge_strength,
                    narrative_strength=0.0,  # TODO: Add narrative scraper
                    contrarian_score=sentiment_data.get('contrarian_opportunity', 0.0),
                    public_sentiment=sentiment_data.get('reddit_sentiment_score', 0.0)
                )

        # Build result
        result = {
            'game': game,
            'referee_profile': ref_profile,
            'home_bias': home_bias,
            'away_bias': away_bias,
            'edges': edges,
            'edge_count': len(edges),
            'sentiment': sentiment_data,
            'model_recommendation': model_recommendation,
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

        # Display sentiment and model recommendation
        if sentiment_data:
            contrarian = sentiment_data.get('contrarian_opportunity', 0.0)
            if contrarian > 0.70:
                print(f"\n   🎯 CONTRARIAN SIGNAL: {contrarian:.0%} (public vs sharp divergence!)")

        if model_recommendation:
            print(f"\n   🤖 MODEL RECOMMENDATION: {model_recommendation.primary_model.value.upper()}")
            print(f"      Confidence: {model_recommendation.confidence_score:.0%}")
            if model_recommendation.contrarian_score > 0:
                print(f"      Contrarian Score: {model_recommendation.contrarian_score:.0%}")

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

                    # Format bet type display
                    bet_display = f"{edge['type']} {edge['pick']}"

                    # Add line if available (team totals, 1H spread)
                    if 'line' in edge:
                        bet_display += f" {edge['line']:.1f}"

                    report.append(f"   Edge #{j}: {bet_display}")
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
            # Count all bet types
            spread_edges = sum(1 for e in all_edges if e['type'] == 'SPREAD')
            total_edges = sum(1 for e in all_edges if e['type'] == 'TOTAL')
            ml_edges = sum(1 for e in all_edges if e['type'] == 'MONEYLINE')
            first_half_spread_edges = sum(1 for e in all_edges if e['type'] == '1H_SPREAD')
            home_team_total_edges = sum(1 for e in all_edges if e['type'] == 'TEAM_TOTAL_HOME')
            away_team_total_edges = sum(1 for e in all_edges if e['type'] == 'TEAM_TOTAL_AWAY')

            report.append(f"Edge Type Distribution:")
            report.append(f"  Full Game Spread: {spread_edges}")
            report.append(f"  Full Game Total: {total_edges}")
            report.append(f"  Moneyline: {ml_edges}")
            report.append(f"  1st Half Spread: {first_half_spread_edges}")
            report.append(f"  Home Team Total: {home_team_total_edges}")
            report.append(f"  Away Team Total: {away_team_total_edges}")
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
        "--json",
        action="store_true",
        help="Output results as JSON (for autonomous agent)"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = AutoWeeklyAnalyzer()

    # Analyze all games - REAL DATA ONLY
    results = analyzer.analyze_all_games(args.week)

    if not results:
        if args.json:
            print(json.dumps({'games_analyzed': 0, 'edges_found': 0, 'top_edges': []}))
        else:
            print("\n❌ No results to report")
        return

    # JSON mode - output structured data for autonomous agent
    if args.json:
        all_edges = []
        for result in results:
            for edge in result.get('edges', []):
                all_edges.append({
                    'game': f"{result['game']['away_team']} @ {result['game']['home_team']}",
                    'edge_type': edge['type'],
                    'pick': edge['pick'],
                    'confidence': edge['confidence'],
                    'edge_size': edge['edge_size'],
                    'reasoning': edge.get('reasoning', '')
                })

        json_output = {
            'games_analyzed': len(results),
            'edges_found': sum(r['edge_count'] for r in results),
            'top_edges': sorted(all_edges, key=lambda e: e['confidence'], reverse=True)[:10]
        }
        print(json.dumps(json_output, indent=2))
        return

    # Human-readable mode
    print("\n" + "="*80)
    print(f"🤖 AUTOMATED NFL WEEKLY ANALYZER")
    print("="*80)
    print(f"\nWeek: {args.week}")
    print(f"Mode: Real Data (Odds API)")

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
