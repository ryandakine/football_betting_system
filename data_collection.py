"""
NFL Data Collection Module
=========================

Professional-grade data collection for NFL betting system.
Integrates with existing advanced_data_ingestion.py and nfl_live_data_fetcher.py
to provide comprehensive NFL data collection capabilities.

Real data sources only - no mock data.
"""

import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = "data"
HISTORICAL_NFL_DATA_PATH = os.path.join(
    DATA_DIR, "historical_nfl_data.parquet"
)
TEAM_STATS_DATA_PATH = os.path.join(DATA_DIR, "team_stats_data.parquet")
PLAYER_STATS_DATA_PATH = os.path.join(DATA_DIR, "player_stats_data.parquet")
WEATHER_DATA_PATH = os.path.join(DATA_DIR, "weather_data.parquet")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


class NFLDataCollector:
    """
    Comprehensive NFL data collection system.
    Coordinates with advanced_data_ingestion.py and nfl_live_data_fetcher.py
    """

    def __init__(self, db_path: str = "data/nfl_betting_system.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36'
            )
        })

        # Rate limiting
        self.last_request_time = {}
        self.request_delays = {
            'pro_football_ref': 3.0,
            'espn': 1.0,
            'nfl_com': 2.0,
            'weather': 1.0
        }

    def _rate_limit(self, source: str):
        """Apply rate limiting for different data sources"""
        now = time.time()
        if source in self.last_request_time:
            elapsed = now - self.last_request_time[source]
            delay = self.request_delays.get(source, 1.0)
            if elapsed < delay:
                time.sleep(delay - elapsed)
        self.last_request_time[source] = time.time()

    def fetch_nfl_schedule(
        self, season: int, week: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch NFL game schedule from ESPN API

        Args:
            season: NFL season year
            week: Specific week (1-18 regular season, 19-22 playoffs),
                  None for all

        Returns:
            DataFrame with game schedule information
        """
        logger.info(
            f"Fetching NFL schedule for {season}, week {week or 'all'}"
        )

        if week:
            url = (
                f"https://site.api.espn.com/apis/site/v2/sports/football/"
                f"nfl/scoreboard?week={week}&seasontype=2&year={season}"
            )
        else:
            url = (
                f"https://site.api.espn.com/apis/site/v2/sports/football/"
                f"nfl/scoreboard?seasontype=2&year={season}"
            )

        self._rate_limit('espn')

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            games = []
            for event in data.get('events', []):
                try:
                    game_id = event.get('id')
                    date_str = event.get('date')

                    # Get teams
                    competitions = event.get('competitions', [{}])
                    if not competitions:
                        continue

                    competition = competitions[0]
                    competitors = competition.get('competitors', [])

                    if len(competitors) != 2:
                        continue

                    home_team = None
                    away_team = None
                    home_score = None
                    away_score = None

                    for competitor in competitors:
                        team_info = competitor.get('team', {})
                        team_abbr = team_info.get('abbreviation', '')
                        score = competitor.get('score')

                        if competitor.get('homeAway') == 'home':
                            home_team = team_abbr
                            home_score = (
                                int(score) if score and score.isdigit()
                                else None
                            )
                        else:
                            away_team = team_abbr
                            away_score = (
                                int(score) if score and score.isdigit()
                                else None
                            )

                    # Get venue info
                    venue = competition.get('venue', {})
                    venue_name = venue.get('fullName', '')

                    # Get status
                    status = event.get('status', {})
                    game_status = status.get('type', {}).get('name', 'scheduled')

                    games.append({
                        'game_id': game_id,
                        'season': season,
                        'week': week,
                        'date': date_str,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'venue': venue_name,
                        'status': game_status,
                        'data_source': 'ESPN_API'
                    })

                except Exception as e:
                    logger.error(
                        f"Error parsing game {event.get('id', 'unknown')}: {e}"
                    )
                    continue

            df = pd.DataFrame(games)
            logger.info(
                f"Successfully collected {len(games)} games from ESPN API"
            )
            return df

        except Exception as e:
            logger.error(f"Error fetching NFL schedule: {e}")
            return pd.DataFrame()

    def fetch_team_stats(self, season: int) -> pd.DataFrame:
        """
        Fetch team statistics from Pro Football Reference

        Args:
            season: NFL season year

        Returns:
            DataFrame with team statistics
        """
        logger.info(f"Fetching team stats for {season} season")

        url = f"https://www.pro-football-reference.com/years/{season}/"
        self._rate_limit('pro_football_ref')

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse team stats from the page
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the team stats table
            stats_table = soup.find('table', {'id': 'team_stats'})
            if not stats_table:
                logger.warning(f"No team stats table found for {season}")
                return pd.DataFrame()

            # Extract table data
            headers = []
            header_row = stats_table.find('thead').find('tr')
            for th in header_row.find_all('th'):
                headers.append(th.get_text().strip())

            rows = []
            tbody = stats_table.find('tbody')
            for row in tbody.find_all('tr'):
                row_data = []
                for td in row.find_all(['td', 'th']):
                    row_data.append(td.get_text().strip())
                if row_data:
                    rows.append(row_data)

            if not rows:
                logger.warning(f"No team stats data found for {season}")
                return pd.DataFrame()

            # Create DataFrame
            df = pd.DataFrame(
                rows, columns=headers[:len(rows[0])] if rows else headers
            )
            df['season'] = season
            df['data_source'] = 'PRO_FOOTBALL_REF'

            # Clean numeric columns
            numeric_columns = [
                'G', 'W', 'L', 'T', 'PF', 'PA', 'PD', 'MoV',
                'SoS', 'SRS', 'OSRS', 'DSRS'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            logger.info(
                f"Successfully collected team stats for {len(df)} teams"
            )
            return df

        except Exception as e:
            logger.error(f"Error fetching team stats: {e}")
            return pd.DataFrame()

    def fetch_player_stats(
        self, season: int, position: str = 'all'
    ) -> pd.DataFrame:
        """
        Fetch player statistics from Pro Football Reference

        Args:
            season: NFL season year
            position: Player position ('QB', 'RB', 'WR', 'TE', 'K', 'DEF',
                     or 'all')

        Returns:
            DataFrame with player statistics
        """
        logger.info(f"Fetching {position} player stats for {season} season")

        position_urls = {
            'QB': (
                f'https://www.pro-football-reference.com/years/'
                f'{season}/passing.htm'
            ),
            'RB': (
                f'https://www.pro-football-reference.com/years/'
                f'{season}/rushing.htm'
            ),
            'WR': (
                f'https://www.pro-football-reference.com/years/'
                f'{season}/receiving.htm'
            ),
            'TE': (
                f'https://www.pro-football-reference.com/years/'
                f'{season}/receiving.htm'
            ),
            'K': (
                f'https://www.pro-football-reference.com/years/'
                f'{season}/kicking.htm'
            ),
            'DEF': (
                f'https://www.pro-football-reference.com/years/'
                f'{season}/defense.htm'
            )
        }

        all_stats = []

        positions_to_fetch = (
            [position] if position != 'all' else list(position_urls.keys())
        )

        for pos in positions_to_fetch:
            if pos not in position_urls:
                logger.warning(f"Unknown position: {pos}")
                continue

            url = position_urls[pos]
            self._rate_limit('pro_football_ref')

            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Find the main stats table
                stats_table = (
                    soup.find('table', {'id': 'stats'}) or
                    soup.find('table', {'class': 'stats_table'})
                )
                if not stats_table:
                    logger.warning(
                        f"No stats table found for {pos} in {season}"
                    )
                    continue

                # Extract headers
                headers = []
                header_row = stats_table.find('thead').find('tr')
                for th in header_row.find_all('th'):
                    headers.append(th.get_text().strip())

                # Extract data rows
                rows = []
                tbody = stats_table.find('tbody')
                for row in tbody.find_all('tr'):
                    # Skip header rows that appear in the middle
                    if (row.find('th') and
                            row.find('th').get('scope') == 'col'):
                        continue

                    row_data = []
                    for cell in row.find_all(['td', 'th']):
                        row_data.append(cell.get_text().strip())

                    if row_data:
                        rows.append(row_data)

                if rows:
                    # Create DataFrame for this position
                    pos_df = pd.DataFrame(
                        rows,
                        columns=headers[:len(rows[0])] if rows else headers
                    )
                    pos_df['position'] = pos
                    pos_df['season'] = season
                    pos_df['data_source'] = 'PRO_FOOTBALL_REF'
                    all_stats.append(pos_df)

                    logger.info(f"Collected stats for {len(pos_df)} {pos} players")

            except Exception as e:
                logger.error(f"Error fetching {pos} stats: {e}")
                continue

        if all_stats:
            combined_df = pd.concat(all_stats, ignore_index=True)
            logger.info(
                f"Successfully collected stats for {len(combined_df)} "
                f"total players"
            )
            return combined_df
        else:
            logger.warning("No player stats collected")
            return pd.DataFrame()

    def fetch_weather_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch weather data for games (simplified implementation)

        Args:
            games_df: DataFrame with game information including venue

        Returns:
            DataFrame with weather information added
        """
        logger.info("Fetching weather data for games")

        # This is a simplified implementation
        # In production, you'd use APIs like OpenWeatherMap with venue coords

        if games_df.empty:
            return games_df

        # Add placeholder weather columns
        games_df = games_df.copy()
        games_df['temperature'] = None
        games_df['weather_conditions'] = None
        games_df['wind_speed'] = None
        games_df['humidity'] = None

        logger.info(
            "Weather data collection completed (placeholder implementation)"
        )
        return games_df

    def save_to_database(self, df: pd.DataFrame, table_name: str):
        """
        Save DataFrame to SQLite database

        Args:
            df: DataFrame to save
            table_name: Name of the database table
        """
        if df.empty:
            logger.warning(f"No data to save to {table_name}")
            return

        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                logger.info(f"Saved {len(df)} records to {table_name} table")

        except Exception as e:
            logger.error(f"Error saving to database: {e}")

    def collect_all_data(
        self, season: int, week: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Orchestrate collection of all NFL data

        Args:
            season: NFL season year
            week: Specific week to collect (optional)

        Returns:
            Dictionary of DataFrames with collected data
        """
        logger.info(f"Starting comprehensive NFL data collection for {season}")

        results = {}

        # 1. Collect game schedule
        logger.info("Collecting game schedule...")
        schedule_df = self.fetch_nfl_schedule(season, week)
        if not schedule_df.empty:
            results['schedule'] = schedule_df
            self.save_to_database(schedule_df, 'game_schedule')

        # 2. Collect team stats
        logger.info("Collecting team statistics...")
        team_stats_df = self.fetch_team_stats(season)
        if not team_stats_df.empty:
            results['team_stats'] = team_stats_df
            self.save_to_database(team_stats_df, 'team_statistics')

        # 3. Collect player stats
        logger.info("Collecting player statistics...")
        player_stats_df = self.fetch_player_stats(season)
        if not player_stats_df.empty:
            results['player_stats'] = player_stats_df
            self.save_to_database(player_stats_df, 'player_statistics')

        # 4. Add weather data to schedule
        if 'schedule' in results:
            logger.info("Adding weather data to schedule...")
            schedule_with_weather = self.fetch_weather_data(results['schedule'])
            results['schedule_with_weather'] = schedule_with_weather
            self.save_to_database(schedule_with_weather, 'games_with_weather')

        logger.info("NFL data collection completed successfully")
        return results


def run_nfl_data_collection(
    season: int = datetime.now().year,
    week: Optional[int] = None,
    save_to_files: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Main function to run NFL data collection

    Args:
        season: NFL season year (default: current year)
        week: Specific week to collect (optional)
        save_to_files: Whether to save data to files

    Returns:
        Dictionary of collected DataFrames
    """
    logger.info("Starting NFL data collection process...")

    collector = NFLDataCollector()
    results = collector.collect_all_data(season, week)

    if save_to_files:
        # Save to parquet files
        for data_type, df in results.items():
            if not df.empty:
                file_path = os.path.join(DATA_DIR, f"{data_type}_{season}.parquet")
                df.to_parquet(file_path, compression='snappy')
                logger.info(f"Saved {data_type} data to {file_path}")

    logger.info("NFL data collection process completed")
    return results


if __name__ == "__main__":
    # Example usage
    current_season = 2024  # Adjust for current NFL season

    # Collect all data for current season
    results = run_nfl_data_collection(
        season=current_season,
        week=None,  # Collect all weeks
        save_to_files=True
    )

    # Print summary
    for data_type, df in results.items():
        print(f"{data_type}: {len(df)} records")