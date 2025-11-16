"""
Unified Game Fetcher for NFL and NCAA
Handles API calls to The Odds API and game parsing
"""
import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.betting_config import API_CONFIG, SPORTS, get_paths


class GameFetcherError(Exception):
    """Custom exception for game fetching errors"""
    pass


class GameFetcher:
    """Unified game fetcher for NFL and NCAA"""

    def __init__(self, sport='nfl', api_key=None):
        """
        Initialize game fetcher

        Args:
            sport: 'nfl' or 'ncaa'
            api_key: The Odds API key (or reads from env)
        """
        if sport not in ['nfl', 'ncaa']:
            raise ValueError(f"Sport must be 'nfl' or 'ncaa', got: {sport}")

        self.sport = sport
        self.api_key = api_key or os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY')
        self.sport_config = SPORTS[sport]
        self.paths = get_paths(sport)

        if not self.api_key:
            raise GameFetcherError(
                "No API key found. Set ODDS_API_KEY environment variable"
            )

    def fetch_games(self) -> List[Dict]:
        """
        Fetch games from The Odds API

        Returns:
            List of game dictionaries

        Raises:
            GameFetcherError: If API call fails
        """
        sport_key = self.sport_config['api_key']
        url = f"{API_CONFIG['odds_api_base_url']}/{sport_key}/odds"

        params = {
            'apiKey': self.api_key,
            'regions': API_CONFIG['regions'],
            'markets': API_CONFIG['markets'],
            'oddsFormat': API_CONFIG['odds_format']
        }

        try:
            print(f"🏈 Fetching {self.sport.upper()} games from The Odds API...")
            response = requests.get(url, params=params, timeout=API_CONFIG['timeout'])

            if response.status_code == 403:
                raise GameFetcherError(
                    "API Key Error (403 Forbidden)\n"
                    "Get new key from: https://the-odds-api.com/"
                )

            if response.status_code != 200:
                raise GameFetcherError(
                    f"API returned status code {response.status_code}"
                )

            games = response.json()
            print(f"✅ Fetched {len(games)} {self.sport.upper()} games")

            return games

        except requests.exceptions.Timeout:
            raise GameFetcherError("API request timed out")
        except requests.exceptions.RequestException as e:
            raise GameFetcherError(f"Network error: {e}")
        except json.JSONDecodeError:
            raise GameFetcherError("Invalid JSON response from API")

    def parse_game(self, game: Dict) -> Optional[Dict]:
        """
        Parse game data into standardized format

        Args:
            game: Raw game data from API

        Returns:
            Parsed game dict or None if parsing fails
        """
        try:
            home_team = game['home_team']
            away_team = game['away_team']
            commence_time = datetime.fromisoformat(
                game['commence_time'].replace('Z', '+00:00')
            )

            # Initialize all odds fields
            spread_line = None
            spread_home_odds = None
            spread_away_odds = None
            total_line = None
            total_over_odds = None
            total_under_odds = None
            home_ml_odds = None
            away_ml_odds = None

            # Get first bookmaker
            bookmaker = game['bookmakers'][0] if game.get('bookmakers') else None

            if bookmaker:
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                spread_line = outcome['point']
                                spread_home_odds = outcome['price']
                            elif outcome['name'] == away_team:
                                spread_away_odds = outcome['price']

                    elif market['key'] == 'totals':
                        total_line = market['outcomes'][0]['point']
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':
                                total_over_odds = outcome['price']
                            elif outcome['name'] == 'Under':
                                total_under_odds = outcome['price']

                    elif market['key'] == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == home_team:
                                home_ml_odds = outcome['price']
                            elif outcome['name'] == away_team:
                                away_ml_odds = outcome['price']

            return {
                'game_id': f"{away_team}_at_{home_team}_{commence_time.strftime('%Y%m%d')}",
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': game['commence_time'],
                'kickoff_time': commence_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'spread': spread_line,
                'spread_home_odds': spread_home_odds,
                'spread_away_odds': spread_away_odds,
                'total': total_line,
                'total_over_odds': total_over_odds,
                'total_under_odds': total_under_odds,
                'home_ml_odds': home_ml_odds,
                'away_ml_odds': away_ml_odds,
                'bookmaker': bookmaker['title'] if bookmaker else None,
                'sport': self.sport
            }

        except (KeyError, IndexError, ValueError) as e:
            print(f"⚠️  Error parsing game: {e}")
            return None

    def save_games(self, games: List[Dict], output_file: Optional[str] = None):
        """
        Save games to JSON file

        Args:
            games: List of parsed game dicts
            output_file: Optional custom output path
        """
        if output_file is None:
            output_file = self.paths['live_games']

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(games, f, indent=2)

        print(f"💾 Saved {len(games)} games to {output_file}")

    def display_games(self, games: List[Dict]):
        """Display games in readable format"""
        if not games:
            print("\n⚠️  No games to display")
            return

        print("\n" + "="*100)
        print(f"🏈 {self.sport.upper()} GAMES ({len(games)} total)")
        print("="*100)

        # Group by day
        games_by_day = {}
        for game in games:
            kickoff = datetime.fromisoformat(
                game['commence_time'].replace('Z', '+00:00')
            )
            day_str = kickoff.strftime('%A, %B %d')

            if day_str not in games_by_day:
                games_by_day[day_str] = []
            games_by_day[day_str].append(game)

        for day, day_games in sorted(games_by_day.items()):
            print(f"\n📅 {day} ({len(day_games)} games)")
            print("-" * 100)

            for game in day_games:
                kickoff = datetime.fromisoformat(
                    game['commence_time'].replace('Z', '+00:00')
                )
                time_str = kickoff.strftime('%I:%M %p')

                spread_str = f"{game['spread']:+.1f}" if game['spread'] else "N/A"
                total_str = f"{game['total']:.1f}" if game['total'] else "N/A"

                print(f"  {time_str:<10} {game['away_team']:<30} @ {game['home_team']:<30}")
                print(f"             Spread: {spread_str:<8} Total: {total_str:<8} Book: {game['bookmaker']}")

    def run(self, save=True, display=True) -> List[Dict]:
        """
        Complete workflow: fetch, parse, save, display

        Args:
            save: Whether to save games to file
            display: Whether to display games

        Returns:
            List of parsed game dicts
        """
        # Fetch raw games
        raw_games = self.fetch_games()

        # Parse games
        parsed_games = []
        for game in raw_games:
            parsed = self.parse_game(game)
            if parsed:
                parsed_games.append(parsed)

        print(f"✅ Successfully parsed {len(parsed_games)}/{len(raw_games)} games")

        # Display
        if display:
            self.display_games(parsed_games)

        # Save
        if save:
            self.save_games(parsed_games)

        return parsed_games


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Fetch NFL or NCAA games')
    parser.add_argument('sport', choices=['nfl', 'ncaa'], help='Sport to fetch')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save to file')
    parser.add_argument('--no-display', action='store_true', help='Don\'t display games')

    args = parser.parse_args()

    try:
        fetcher = GameFetcher(sport=args.sport)
        games = fetcher.run(
            save=not args.no_save,
            display=not args.no_display
        )

        print(f"\n✅ {args.sport.upper()} games ready for prediction!")
        print(f"   Games file: {fetcher.paths['live_games']}")
        print("="*100)

    except GameFetcherError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
