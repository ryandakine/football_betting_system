#!/usr/bin/env python3
"""
15-Year NFL Parlay Backtest Data Collector
===========================================
Collects comprehensive historical NFL data (2010-2024) for parlay model training.

Data Collected:
- Game results (scores, winners)
- Betting lines (spread, total, ML - opening & closing)
- Referee assignments
- Bet outcomes (did spread/total/ML hit?)
- Team stats, weather, injuries (optional)

Sources:
- ESPN API (game results, scores)
- Pro Football Reference (historical data)
- Football Zebras (referee assignments)
- Sports Reference (betting lines when available)

Usage:
    python collect_15_year_parlay_data.py --start-year 2010 --end-year 2024
    python collect_15_year_parlay_data.py --year 2023 --weeks 1-17
"""

import argparse
import asyncio
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests


class NFLHistoricalDataCollector:
    """Collects 15 years of NFL historical data for parlay training."""

    def __init__(self, output_dir: str = "data/historical/parlay_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.games = []
        self.referee_assignments = {}

    def collect_season(self, year: int) -> List[Dict[str, Any]]:
        """
        Collect all games for a given NFL season.

        Args:
            year: Season year (2010-2024)

        Returns:
            List of game dictionaries with results and betting data
        """
        print(f"\n{'='*80}")
        print(f"📅 COLLECTING {year} NFL SEASON")
        print(f"{'='*80}\n")

        season_games = []

        # Collect each week (regular season weeks 1-17/18)
        num_weeks = 18 if year >= 2021 else 17  # NFL expanded to 18 weeks in 2021

        for week in range(1, num_weeks + 1):
            print(f"\n🏈 Week {week}/{num_weeks}...")

            try:
                week_games = self._collect_week(year, week)
                season_games.extend(week_games)
                print(f"   ✅ Collected {len(week_games)} games")

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue

        # Save season data
        season_file = self.output_dir / f"nfl_{year}_season.json"
        with open(season_file, 'w') as f:
            json.dump(season_games, f, indent=2)

        print(f"\n✅ Saved {len(season_games)} games to {season_file}")
        return season_games

    def _collect_week(self, year: int, week: int) -> List[Dict[str, Any]]:
        """Collect all games for a specific week."""

        games = []

        # ESPN API endpoint for NFL scoreboard
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {
            'dates': self._get_week_date_range(year, week),
            'limit': 100
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse games
            for event in data.get('events', []):
                game = self._parse_espn_game(event, year, week)
                if game:
                    games.append(game)

        except requests.exceptions.RequestException as e:
            print(f"      ⚠️  ESPN API error: {e}")

        return games

    def _parse_espn_game(self, event: Dict, year: int, week: int) -> Optional[Dict[str, Any]]:
        """Parse ESPN game data into our format."""

        try:
            competition = event['competitions'][0]
            competitors = competition['competitors']

            # Get home/away teams
            home_team = next(c for c in competitors if c['homeAway'] == 'home')
            away_team = next(c for c in competitors if c['homeAway'] == 'away')

            # Get scores
            home_score = int(home_team['score'])
            away_score = int(away_team['score'])

            # Get team abbreviations
            home_abbr = home_team['team']['abbreviation']
            away_abbr = away_team['team']['abbreviation']

            # Get betting lines if available
            odds = competition.get('odds', [{}])[0] if competition.get('odds') else {}
            spread = odds.get('spread', None)
            over_under = odds.get('overUnder', None)

            # Calculate outcomes
            actual_spread = home_score - away_score
            actual_total = home_score + away_score

            game_data = {
                'game_id': event['id'],
                'year': year,
                'week': week,
                'date': event.get('date', ''),
                'home_team': home_abbr,
                'away_team': away_abbr,
                'home_score': home_score,
                'away_score': away_score,
                'winner': home_abbr if home_score > away_score else away_abbr,

                # Betting lines
                'spread_line': spread,
                'total_line': over_under,

                # Actual results
                'actual_spread': actual_spread,
                'actual_total': actual_total,

                # Bet outcomes (if lines available)
                'spread_result': self._calculate_spread_result(actual_spread, spread) if spread else None,
                'total_result': self._calculate_total_result(actual_total, over_under) if over_under else None,

                # Referee (to be filled later)
                'referee': 'Unknown',

                # Status
                'status': competition['status']['type']['name']
            }

            return game_data

        except (KeyError, StopIteration, ValueError) as e:
            print(f"      ⚠️  Parse error: {e}")
            return None

    def _calculate_spread_result(self, actual_spread: float, line: float) -> str:
        """Determine if home team covered the spread."""
        if actual_spread > line:
            return 'HOME_COVER'
        elif actual_spread < line:
            return 'AWAY_COVER'
        else:
            return 'PUSH'

    def _calculate_total_result(self, actual_total: float, line: float) -> str:
        """Determine if total went over/under."""
        if actual_total > line:
            return 'OVER'
        elif actual_total < line:
            return 'UNDER'
        else:
            return 'PUSH'

    def _get_week_date_range(self, year: int, week: int) -> str:
        """
        Convert week number to date range for ESPN API.

        ESPN API expects YYYYMMDD format.
        """
        # Rough estimate: Week 1 starts first Thursday of September
        # This is approximate, would need actual NFL calendar for precision
        start_date = date(year, 9, 1)

        # Find first Thursday
        while start_date.weekday() != 3:  # 3 = Thursday
            start_date = date(year, 9, start_date.day + 1)

        # Add weeks
        from datetime import timedelta
        week_start = start_date + timedelta(weeks=week-1)
        week_end = week_start + timedelta(days=6)

        return f"{week_start.strftime('%Y%m%d')}-{week_end.strftime('%Y%m%d')}"

    def enrich_with_referees(self, games: List[Dict], year: int):
        """
        Add referee assignments to game data.

        Uses referee_assignment_fetcher.py if available.
        """
        print(f"\n🔍 Enriching with referee data for {year}...")

        try:
            from referee_assignment_fetcher import RefereeAssignmentsFetcher

            fetcher = RefereeAssignmentsFetcher()

            # Organize games by week
            games_by_week = {}
            for game in games:
                week = game['week']
                if week not in games_by_week:
                    games_by_week[week] = []
                games_by_week[week].append(game)

            # Fetch refs for each week
            for week, week_games in games_by_week.items():
                try:
                    ref_data = fetcher.fetch_week(year, week)

                    if ref_data and ref_data.get('ok'):
                        ref_assignments = ref_data.get('games', {})

                        # Match to games
                        for game in week_games:
                            matchup = f"{game['away_team']} @ {game['home_team']}"
                            if matchup in ref_assignments:
                                game['referee'] = ref_assignments[matchup]
                                print(f"   ✅ {matchup}: {ref_assignments[matchup]}")

                    time.sleep(0.5)  # Rate limit

                except Exception as e:
                    print(f"   ⚠️  Week {week}: {e}")
                    continue

        except ImportError:
            print("   ⚠️  referee_assignment_fetcher not available")

    def generate_training_dataset(self, years: List[int]) -> str:
        """
        Generate consolidated training dataset from multiple years.

        Creates:
        - parlay_training_data.json: All games with outcomes
        - parlay_training_summary.txt: Statistics
        """
        print(f"\n{'='*80}")
        print(f"📊 GENERATING TRAINING DATASET")
        print(f"{'='*80}\n")

        all_games = []

        # Load all years
        for year in years:
            season_file = self.output_dir / f"nfl_{year}_season.json"
            if season_file.exists():
                with open(season_file, 'r') as f:
                    season_games = json.load(f)
                    all_games.extend(season_games)
                    print(f"✅ Loaded {len(season_games)} games from {year}")
            else:
                print(f"⚠️  {year} season data not found, skipping")

        if not all_games:
            print("\n❌ No game data found!")
            return ""

        # Filter to completed games with betting lines
        training_games = [
            g for g in all_games
            if g['status'] == 'STATUS_FINAL' and g['spread_line'] is not None
        ]

        print(f"\n✅ {len(training_games)} games with complete data")

        # Save training dataset
        training_file = self.output_dir / "parlay_training_data.json"
        with open(training_file, 'w') as f:
            json.dump(training_games, f, indent=2)

        # Generate summary
        summary = self._generate_summary(training_games, years)

        summary_file = self.output_dir / "parlay_training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)

        print(f"\n💾 Saved:")
        print(f"   - {training_file}")
        print(f"   - {summary_file}")

        return summary

    def _generate_summary(self, games: List[Dict], years: List[int]) -> str:
        """Generate training data summary statistics."""

        summary = []
        summary.append("=" * 80)
        summary.append("NFL PARLAY TRAINING DATASET SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Years: {min(years)}-{max(years)} ({len(years)} seasons)")
        summary.append(f"Total Games: {len(games)}")
        summary.append("")

        # Games by year
        summary.append("GAMES BY YEAR:")
        games_by_year = {}
        for game in games:
            year = game['year']
            games_by_year[year] = games_by_year.get(year, 0) + 1

        for year in sorted(games_by_year.keys()):
            summary.append(f"  {year}: {games_by_year[year]} games")
        summary.append("")

        # Bet outcomes
        spread_outcomes = {}
        total_outcomes = {}

        for game in games:
            if game['spread_result']:
                spread_outcomes[game['spread_result']] = spread_outcomes.get(game['spread_result'], 0) + 1
            if game['total_result']:
                total_outcomes[game['total_result']] = total_outcomes.get(game['total_result'], 0) + 1

        summary.append("SPREAD OUTCOMES:")
        for outcome, count in spread_outcomes.items():
            pct = (count / len(games)) * 100
            summary.append(f"  {outcome}: {count} ({pct:.1f}%)")
        summary.append("")

        summary.append("TOTAL OUTCOMES:")
        for outcome, count in total_outcomes.items():
            pct = (count / len(games)) * 100
            summary.append(f"  {outcome}: {count} ({pct:.1f}%)")
        summary.append("")

        # Referees
        referees = {}
        for game in games:
            ref = game.get('referee', 'Unknown')
            referees[ref] = referees.get(ref, 0) + 1

        known_refs = {k: v for k, v in referees.items() if k != 'Unknown'}
        summary.append(f"REFEREE DATA:")
        summary.append(f"  Known Referees: {len(known_refs)}")
        summary.append(f"  Games with Referee: {sum(known_refs.values())} ({(sum(known_refs.values())/len(games))*100:.1f}%)")
        summary.append(f"  Games without Referee: {referees.get('Unknown', 0)}")
        summary.append("")

        summary.append("=" * 80)
        summary.append("TRAINING DATA READY FOR PARLAY MODEL!")
        summary.append("=" * 80)

        return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Collect 15 years of NFL data for parlay model training"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Start year (default: 2010)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year (default: 2024)"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Collect single year only"
    )
    parser.add_argument(
        "--enrich-referees",
        action="store_true",
        help="Add referee assignments (slower)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/historical/parlay_training",
        help="Output directory"
    )

    args = parser.parse_args()

    # Create collector
    collector = NFLHistoricalDataCollector(output_dir=args.output_dir)

    # Determine years to collect
    if args.year:
        years = [args.year]
    else:
        years = list(range(args.start_year, args.end_year + 1))

    print(f"\n{'='*80}")
    print(f"🏈 15-YEAR NFL PARLAY DATA COLLECTION")
    print(f"{'='*80}")
    print(f"\nCollecting seasons: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"Output directory: {args.output_dir}")
    print(f"Referee enrichment: {'Yes' if args.enrich_referees else 'No'}")
    print(f"\nThis will take approximately {len(years) * 2} minutes...")
    print(f"{'='*80}\n")

    # Collect each season
    for year in years:
        games = collector.collect_season(year)

        # Enrich with referees if requested
        if args.enrich_referees:
            collector.enrich_with_referees(games, year)

            # Re-save with referee data
            season_file = collector.output_dir / f"nfl_{year}_season.json"
            with open(season_file, 'w') as f:
                json.dump(games, f, indent=2)

    # Generate consolidated training dataset
    summary = collector.generate_training_dataset(years)

    print("\n" + summary)

    print(f"\n{'='*80}")
    print(f"✅ DATA COLLECTION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"  1. Review: {args.output_dir}/parlay_training_summary.txt")
    print(f"  2. Train: Use parlay_training_data.json for model training")
    print(f"  3. Backtest: Run historical parlay simulations")
    print("")


if __name__ == "__main__":
    main()
