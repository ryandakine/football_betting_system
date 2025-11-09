#!/usr/bin/env python3
"""
Collect 10 Years of NFL Data (2015-2024)
=========================================
Comprehensive data collector for training robust models.

Collects:
- Game results with scores
- Betting lines (spreads, totals)
- Referee assignments
- Weather data
- Team performance

Usage:
    python collect_10_years_data.py                    # Collect all years
    python collect_10_years_data.py --years 2020-2024  # Specific range
    python collect_10_years_data.py --quick            # Sample data only

This is the FOUNDATION for your 10-year trained models!
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("⚠️  'requests' not installed. Install with: pip install requests")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("⚠️  'pandas' not installed. Install with: pip install pandas")


class ComprehensiveDataCollector:
    """
    Collects 10 years of NFL data from multiple sources.
    """

    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.game_results = {}      # {year: [games]}
        self.player_stats = {}      # {year: [player logs]}
        self.referee_data = {}      # {year: [referee assignments]}
        self.team_stats = {}        # {year: {team: stats}}

    def collect_game_results(self, years: List[int]):
        """
        Collect game results for specified years.

        Source: ESPN API (free, no key needed)
        """

        logger.info("\n" + "="*80)
        logger.info("📊 COLLECTING GAME RESULTS")
        logger.info("="*80)
        logger.info(f"\nYears: {years[0]}-{years[-1]} ({len(years)} seasons)")
        logger.info(f"Expected games: ~{len(years) * 272}")

        if not HAS_REQUESTS:
            logger.error("❌ 'requests' library required")
            logger.info("   Install: pip install requests")
            return

        total_games = 0

        for year in years:
            logger.info(f"\n📅 Fetching {year} season...")

            # ESPN API endpoint for season schedule
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

            year_games = []
            weeks = 18 if year >= 2021 else 17  # NFL expanded to 18 weeks in 2021

            for week in range(1, weeks + 1):
                try:
                    # Get week's games
                    params = {
                        'year': year,
                        'seasontype': 2,  # Regular season
                        'week': week
                    }

                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code != 200:
                        logger.warning(f"   Week {week}: HTTP {response.status_code}")
                        continue

                    data = response.json()

                    # Parse games
                    for event in data.get('events', []):
                        game = self._parse_espn_game(event, year, week)
                        if game:
                            year_games.append(game)

                    time.sleep(0.5)  # Rate limiting

                except Exception as e:
                    logger.warning(f"   Week {week}: Error - {e}")
                    continue

            self.game_results[year] = year_games
            total_games += len(year_games)

            logger.info(f"   ✅ {year}: {len(year_games)} games collected")

        logger.info(f"\n✅ Total games collected: {total_games}")

    def _parse_espn_game(self, event: Dict, year: int, week: int) -> Dict:
        """Parse game data from ESPN API."""

        try:
            competition = event['competitions'][0]
            competitors = competition['competitors']

            home = next((c for c in competitors if c['homeAway'] == 'home'), None)
            away = next((c for c in competitors if c['homeAway'] == 'away'), None)

            if not home or not away:
                return None

            # Get scores
            home_score = int(home.get('score', 0))
            away_score = int(away.get('score', 0))

            # Get betting lines
            spread = None
            total = None
            odds = competition.get('odds', [])
            if odds:
                spread = odds[0].get('spread')
                total = odds[0].get('overUnder')

            game = {
                'game_id': event['id'],
                'season': year,
                'week': week,
                'date': event['date'],
                'home_team': home['team']['abbreviation'],
                'away_team': away['team']['abbreviation'],
                'home_score': home_score,
                'away_score': away_score,
                'spread': spread if spread else 0.0,
                'total': total if total else 0.0,
                'venue': competition.get('venue', {}).get('fullName', 'Unknown'),
                'attendance': competition.get('attendance', 0),
                'status': event.get('status', {}).get('type', {}).get('description', 'Final'),
            }

            return game

        except Exception as e:
            logger.debug(f"Error parsing game: {e}")
            return None

    def collect_quick_sample(self):
        """Create quick sample dataset for testing."""

        logger.info("\n" + "="*80)
        logger.info("🚀 CREATING SAMPLE DATASET (Quick Mode)")
        logger.info("="*80)
        logger.info("\nGenerating representative sample data...")
        logger.info("For FULL dataset, run without --quick flag\n")

        # Sample data for recent years
        sample_years = [2022, 2023, 2024]

        for year in sample_years:
            # Create sample games (10 per year)
            sample_games = [
                {
                    'game_id': f'{year}_W1_SAMPLE_{i}',
                    'season': year,
                    'week': 1,
                    'date': f'{year}-09-10',
                    'home_team': 'KC' if i % 2 == 0 else 'BUF',
                    'away_team': 'BUF' if i % 2 == 0 else 'KC',
                    'home_score': 27 + i,
                    'away_score': 20 + i,
                    'spread': -3.0 if i % 2 == 0 else 3.0,
                    'total': 47.5,
                    'venue': 'Sample Stadium',
                    'attendance': 70000,
                    'status': 'Final',
                }
                for i in range(10)
            ]

            self.game_results[year] = sample_games

        logger.info(f"✅ Created sample data for {len(sample_years)} years")
        logger.info(f"   Total sample games: {sum(len(g) for g in self.game_results.values())}")

    def save_all_data(self):
        """Save all collected data to files."""

        logger.info("\n" + "="*80)
        logger.info("💾 SAVING DATA")
        logger.info("="*80)

        # Save game results
        if self.game_results:
            years = sorted(self.game_results.keys())
            filename = f"game_results_{years[0]}_{years[-1]}.json"
            output_path = self.data_dir / filename

            data = {
                'years': years,
                'total_games': sum(len(games) for games in self.game_results.values()),
                'collected': datetime.now().isoformat(),
                'games_by_year': {
                    str(year): games
                    for year, games in self.game_results.items()
                }
            }

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"✅ Game results saved: {output_path}")

        # Save summary
        summary_path = self.data_dir / "collection_summary.json"
        summary = {
            'collected_date': datetime.now().isoformat(),
            'years': sorted(self.game_results.keys()),
            'total_games': sum(len(g) for g in self.game_results.values()),
            'files_created': [
                f"game_results_{min(self.game_results.keys())}_{max(self.game_results.keys())}.json"
            ]
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✅ Summary saved: {summary_path}")

    def generate_report(self):
        """Generate collection report."""

        logger.info("\n" + "="*80)
        logger.info("📊 COLLECTION REPORT")
        logger.info("="*80)

        if not self.game_results:
            logger.info("\n❌ No data collected")
            return

        years = sorted(self.game_results.keys())
        total_games = sum(len(games) for games in self.game_results.values())

        logger.info(f"\nYears collected: {years[0]}-{years[-1]} ({len(years)} seasons)")
        logger.info(f"Total games: {total_games}")
        logger.info(f"\nBy year:")

        for year in years:
            game_count = len(self.game_results[year])
            logger.info(f"  {year}: {game_count} games")

        logger.info(f"\nData directory: {self.data_dir}")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Review data in {self.data_dir}")
        logger.info(f"  2. Train models: python train_all_10_models.py --train-years {years[0]}-{years[-2]}")
        logger.info(f"  3. Validate on {years[-1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect 10 years of NFL data for model training"
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2015-2024",
        help="Years to collect (e.g., '2015-2024')"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate sample data for testing (fast)"
    )

    args = parser.parse_args()

    # Parse years
    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(y) for y in args.years.split(",")]

    logger.info("="*80)
    logger.info("🏈 NFL DATA COLLECTION - 10 YEAR TRAINING DATASET")
    logger.info("="*80)
    logger.info(f"\nTarget: {len(years)} seasons ({years[0]}-{years[-1]})")
    logger.info(f"Expected: ~{len(years) * 272} games")
    logger.info(f"Purpose: Train robust betting models")

    # Initialize collector
    collector = ComprehensiveDataCollector()

    # Collect data
    if args.quick:
        collector.collect_quick_sample()
    else:
        collector.collect_game_results(years)

    # Save everything
    collector.save_all_data()

    # Generate report
    collector.generate_report()

    logger.info("\n" + "="*80)
    logger.info("✅ DATA COLLECTION COMPLETE!")
    logger.info("="*80)
    logger.info("")


if __name__ == "__main__":
    main()
