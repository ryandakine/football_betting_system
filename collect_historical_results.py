#!/usr/bin/env python3
"""
Collect Historical NFL Results (2024 Season)
=============================================
Fetches game results for weeks 1-9 from ESPN API to populate backtest.

Usage:
    python collect_historical_results.py                # Collect all weeks 1-9
    python collect_historical_results.py --weeks 1-5    # Specific weeks
    python collect_historical_results.py --week 1       # Single week

This script is FREE - no API key needed!
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
import time

# Try to import requests, fall back to urllib if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.parse
    HAS_REQUESTS = False
    print("⚠️  'requests' library not found. Using urllib (slower).")
    print("   Install requests: pip install requests")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# 2024 NFL Season week start dates
WEEK_DATES = {
    1: "20240905",  # Sept 5-9 (Thu-Mon)
    2: "20240912",  # Sept 12-16
    3: "20240919",  # Sept 19-23
    4: "20240926",  # Sept 26-30
    5: "20241003",  # Oct 3-7
    6: "20241010",  # Oct 10-14
    7: "20241017",  # Oct 17-21
    8: "20241024",  # Oct 24-28
    9: "20241031",  # Oct 31-Nov 4
}


class ResultsCollector:
    """Collects historical NFL results from ESPN API."""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def fetch_week_results(self, week: int) -> List[Dict[str, Any]]:
        """
        Fetch all game results for a specific week.

        Returns list of games with scores, spreads, totals.
        """

        date = WEEK_DATES.get(week)
        if not date:
            logger.error(f"❌ Week {week} not in range (1-9)")
            return []

        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={date}&limit=100"

        logger.info(f"📊 Fetching Week {week} results from ESPN...")
        logger.info(f"   Date: {date}")

        try:
            # Fetch data
            if HAS_REQUESTS:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    logger.error(f"   ❌ HTTP {response.status_code}")
                    return []
                data = response.json()
            else:
                # Use urllib as fallback
                with urllib.request.urlopen(url) as response:
                    data = json.loads(response.read().decode())

            # Parse games
            games = []
            for event in data.get('events', []):
                try:
                    game = self._parse_game(event, week)
                    if game:
                        games.append(game)
                except Exception as e:
                    logger.warning(f"   ⚠️  Error parsing game: {e}")
                    continue

            logger.info(f"   ✅ Found {len(games)} games")
            return games

        except Exception as e:
            logger.error(f"   ❌ Error fetching data: {e}")
            return []

    def _parse_game(self, event: Dict, week: int) -> Dict[str, Any]:
        """Parse a single game from ESPN data."""

        competition = event['competitions'][0]
        competitors = competition['competitors']

        # Find home and away teams
        home = next((c for c in competitors if c['homeAway'] == 'home'), None)
        away = next((c for c in competitors if c['homeAway'] == 'away'), None)

        if not home or not away:
            return None

        # Get scores
        home_score = int(home.get('score', 0))
        away_score = int(away.get('score', 0))

        # Calculate margin
        actual_margin = home_score - away_score

        # Get betting lines (if available)
        spread_line = None
        total_line = None

        odds = competition.get('odds', [])
        if odds and len(odds) > 0:
            # Parse spread (e.g., "KC -3.0")
            spread_str = odds[0].get('details', '')
            if spread_str:
                # Extract number from string like "KC -3.0"
                parts = spread_str.split()
                if len(parts) >= 2:
                    try:
                        spread_line = float(parts[1])
                        # Adjust sign if needed (home team perspective)
                        if home['team']['abbreviation'] not in parts[0]:
                            spread_line = -spread_line
                    except:
                        pass

            total_line = odds[0].get('overUnder')

        # Determine outcomes
        home_covered = None
        went_over = None

        if spread_line is not None:
            # Did home cover? (actual margin > spread line)
            home_covered = actual_margin > spread_line

        if total_line is not None:
            actual_total = home_score + away_score
            went_over = actual_total > total_line

        # Get referee (if available)
        # ESPN doesn't always provide referee in scoreboard API
        # You may need to scrape from another source
        referee = "Unknown"

        game = {
            'game_id': event['id'],
            'week': week,
            'date': event['date'],
            'home_team': home['team']['abbreviation'],
            'away_team': away['team']['abbreviation'],
            'home_score': home_score,
            'away_score': away_score,
            'actual_margin': actual_margin,
            'spread_line': spread_line if spread_line is not None else 0.0,
            'total_line': total_line if total_line is not None else 0.0,
            'home_covered': home_covered,
            'went_over': went_over,
            'referee': referee,
            'status': event.get('status', {}).get('type', {}).get('description', 'Final'),
        }

        return game

    def collect_multiple_weeks(self, weeks: List[int]) -> Dict[int, List[Dict]]:
        """Collect results for multiple weeks."""

        logger.info("\n" + "="*80)
        logger.info("🏈 COLLECTING HISTORICAL NFL RESULTS")
        logger.info("="*80)
        logger.info(f"\nWeeks: {weeks[0]}-{weeks[-1]}")
        logger.info(f"Source: ESPN API (Free)")
        logger.info("")

        all_results = {}

        for week in weeks:
            results = self.fetch_week_results(week)
            if results:
                all_results[week] = results
                time.sleep(1)  # Be nice to ESPN servers

        logger.info("\n" + "="*80)
        logger.info("📊 COLLECTION SUMMARY")
        logger.info("="*80)

        total_games = sum(len(games) for games in all_results.values())
        logger.info(f"\nTotal weeks collected: {len(all_results)}")
        logger.info(f"Total games collected: {total_games}")

        # Show week-by-week
        for week, games in sorted(all_results.items()):
            logger.info(f"  Week {week}: {len(games)} games")

        return all_results

    def save_results(self, results: Dict[int, List[Dict]], filename: str = None):
        """Save results to JSON file."""

        if filename is None:
            weeks = list(results.keys())
            filename = f"historical_results_weeks_{weeks[0]}-{weeks[-1]}.json"

        output_path = self.output_dir / filename

        # Format for saving
        data = {
            'season': 2024,
            'weeks': list(results.keys()),
            'total_games': sum(len(games) for games in results.values()),
            'collected_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'games_by_week': {
                f'week_{week}': games
                for week, games in results.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"\n✅ Results saved to: {output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Collect historical NFL results from ESPN API"
    )
    parser.add_argument(
        "--weeks",
        type=str,
        default="1-9",
        help="Weeks to collect (e.g., '1-9' or '1,2,3')"
    )
    parser.add_argument(
        "--week",
        type=int,
        help="Single week to collect"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename (default: auto-generated)"
    )

    args = parser.parse_args()

    # Parse weeks
    if args.week:
        weeks = [args.week]
    elif "-" in args.weeks:
        start, end = args.weeks.split("-")
        weeks = list(range(int(start), int(end) + 1))
    else:
        weeks = [int(w) for w in args.weeks.split(",")]

    # Collect results
    collector = ResultsCollector()
    results = collector.collect_multiple_weeks(weeks)

    if not results:
        logger.error("\n❌ No results collected!")
        return

    # Save to file
    output_path = collector.save_results(results, args.output)

    logger.info("\n" + "="*80)
    logger.info("✅ COLLECTION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\n📁 File: {output_path}")
    logger.info(f"📊 Total: {sum(len(g) for g in results.values())} games")
    logger.info("\nNext step:")
    logger.info(f"  python backtest_historical_weeks.py --weeks {weeks[0]}-{weeks[-1]}")
    logger.info("")


if __name__ == "__main__":
    main()
