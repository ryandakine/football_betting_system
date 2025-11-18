#!/usr/bin/env python3
"""
Historical Odds Scraper
Scrapes historical NFL betting lines from Pro Football Reference
"""
import sys
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import time


class HistoricalOddsScraper:
    """Scrape historical NFL odds from various sources"""

    def __init__(self):
        self.db_file = Path('data/historical_odds.json')
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.db_file.exists():
            self.db_file.write_text('{}')

    def load_database(self):
        """Load historical odds database"""
        with open(self.db_file) as f:
            return json.load(f)

    def save_database(self, db):
        """Save historical odds database"""
        with open(self.db_file, 'w') as f:
            json.dump(db, f, indent=2)

    def scrape_pro_football_reference(self, year, week):
        """
        Scrape Pro Football Reference for a specific week

        Example URL: https://www.pro-football-reference.com/years/2024/week_1.htm

        Note: This is a basic template. PFR may require headers/cookies.
        """
        url = f"https://www.pro-football-reference.com/years/{year}/week_{week}.htm"

        print(f"Fetching {url}...")

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"⚠️  HTTP {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find games table (adjust selector based on actual HTML)
            games_table = soup.find('table', {'id': 'games'})

            if not games_table:
                print("⚠️  No games table found")
                return []

            games = []

            # Parse rows (adjust based on actual HTML structure)
            for row in games_table.find_all('tr'):
                cols = row.find_all('td')

                if len(cols) < 8:
                    continue

                try:
                    # Example parsing (adjust to actual columns)
                    away_team = cols[3].text.strip()
                    home_team = cols[5].text.strip()

                    # Look for odds columns (varies by site)
                    # This is a placeholder - actual implementation depends on HTML structure
                    spread = None
                    total = None

                    if spread or total:
                        game_key = f"{year}-{week:02d}_{away_team}_at_{home_team}"
                        games.append({
                            'key': game_key,
                            'spread': spread,
                            'total': total,
                            'source': 'pro_football_reference',
                            'year': year,
                            'week': week
                        })

                except Exception as e:
                    print(f"Error parsing row: {e}")
                    continue

            return games

        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return []

    def add_manual_odds(self, game_key, spread=None, total=None, home_ml=None, away_ml=None, source='manual'):
        """
        Manually add historical odds to database

        Args:
            game_key: "2024-01-14_KC_at_BUF" format
            spread: Home spread (e.g., -2.5)
            total: Over/Under (e.g., 48.5)
            home_ml: Home moneyline (e.g., -135)
            away_ml: Away moneyline (e.g., +115)
            source: Data source
        """
        db = self.load_database()

        db[game_key] = {
            'spread': spread,
            'total': total,
            'home_ml': home_ml,
            'away_ml': away_ml,
            'source': source,
            'added_at': datetime.now().isoformat()
        }

        self.save_database(db)
        print(f"✅ Added {game_key}")

    def bulk_import_from_csv(self, csv_file):
        """
        Import historical odds from CSV file

        CSV format:
        date,away_team,home_team,spread,total,home_ml,away_ml,source
        2024-01-14,KC,BUF,-2.5,48.5,-135,+115,closing_line
        """
        import csv

        db = self.load_database()
        added = 0

        with open(csv_file) as f:
            reader = csv.DictReader(f)

            for row in reader:
                game_key = f"{row['date']}_{row['away_team']}_at_{row['home_team']}"

                db[game_key] = {
                    'spread': float(row['spread']) if row['spread'] else None,
                    'total': float(row['total']) if row['total'] else None,
                    'home_ml': int(row['home_ml']) if row['home_ml'] else None,
                    'away_ml': int(row['away_ml']) if row['away_ml'] else None,
                    'source': row.get('source', 'csv_import'),
                    'added_at': datetime.now().isoformat()
                }

                added += 1

        self.save_database(db)
        print(f"✅ Imported {added} games from {csv_file}")


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Scrape historical NFL odds')
    parser.add_argument('--year', type=int, help='NFL season year')
    parser.add_argument('--week', type=int, help='NFL week number')
    parser.add_argument('--manual', action='store_true', help='Manually add a game')
    parser.add_argument('--csv', type=str, help='Import from CSV file')

    args = parser.parse_args()

    scraper = HistoricalOddsScraper()

    if args.manual:
        print("Manual odds entry:")
        print("Example: 2024-01-14_KC_at_BUF")
        game_key = input("Game key: ")
        spread = input("Spread (home): ")
        total = input("Total: ")
        home_ml = input("Home ML: ")
        away_ml = input("Away ML: ")

        scraper.add_manual_odds(
            game_key,
            float(spread) if spread else None,
            float(total) if total else None,
            int(home_ml) if home_ml else None,
            int(away_ml) if away_ml else None
        )

    elif args.csv:
        scraper.bulk_import_from_csv(args.csv)

    elif args.year and args.week:
        games = scraper.scrape_pro_football_reference(args.year, args.week)
        print(f"\n✅ Found {len(games)} games")

        # Add to database
        db = scraper.load_database()
        for game in games:
            db[game['key']] = game
        scraper.save_database(db)

    else:
        print("Usage:")
        print("  python scrape_historical_odds.py --year 2024 --week 1")
        print("  python scrape_historical_odds.py --manual")
        print("  python scrape_historical_odds.py --csv historical_odds.csv")


if __name__ == '__main__':
    main()
