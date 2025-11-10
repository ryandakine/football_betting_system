#!/usr/bin/env python3
"""
Covers.com Historical Closing Lines Scraper
============================================

Covers.com has extensive historical closing lines archive.
This gets ACTUAL MARKET SPREADS that books were offering.

USAGE:
    python scrape_covers_historical.py 2023
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta


class CoversHistoricalSpreads:
    """Scrape historical closing spreads from Covers.com"""

    def __init__(self):
        self.base_url = "https://www.covers.com/sports/ncaaf/matchups"

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.covers.com/',
        }

        self.output_dir = Path("data/market_spreads")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scrape_season_by_weeks(self, year: int, start_week: int = 1, end_week: int = 15) -> pd.DataFrame:
        """
        Scrape season week by week

        Covers URL format:
        https://www.covers.com/sports/ncaaf/matchups?selectedDate=2023-09-02
        """

        print(f"\n{'='*80}")
        print(f"📊 SCRAPING {year} SEASON FROM COVERS.COM")
        print(f"{'='*80}\n")

        all_games = []

        # NCAA season typically starts late August / early September
        # Week 1 usually starts around Sep 1
        season_start = datetime(year, 9, 1)

        for week in range(start_week, end_week + 1):
            print(f"\n📅 Week {week}...")

            # Calculate approximate date for this week
            week_date = season_start + timedelta(weeks=week-1)
            date_str = week_date.strftime('%Y-%m-%d')

            url = f"{self.base_url}?selectedDate={date_str}"

            print(f"   URL: {url}")

            try:
                response = requests.get(url, headers=self.headers, timeout=20)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Covers uses various structures - look for game cards
                    game_cards = soup.find_all('div', class_=['cmg_matchup_game_box', 'covers-CoversConsensus'])

                    if not game_cards:
                        # Try alternate structure
                        game_cards = soup.find_all('div', {'data-game-id': True})

                    if not game_cards:
                        print(f"   ⚠️  No games found")
                    else:
                        print(f"   Found {len(game_cards)} potential games")

                        for card in game_cards:
                            try:
                                game_data = self._parse_game_card(card, year, week)
                                if game_data:
                                    all_games.append(game_data)
                            except Exception as e:
                                continue

                        print(f"   Extracted {len([g for g in all_games if g['week'] == week])} games")

                elif response.status_code == 403:
                    print(f"   ❌ 403 Forbidden")
                    break

                else:
                    print(f"   ❌ Error {response.status_code}")

            except Exception as e:
                print(f"   ❌ Error: {e}")

            # Be nice to server
            time.sleep(3)

        if all_games:
            df = pd.DataFrame(all_games)
            print(f"\n✅ Total scraped: {len(df)} games")
            return df
        else:
            print(f"\n⚠️  No games scraped")
            return pd.DataFrame()

    def validate_game(self, game: dict) -> tuple[bool, str]:
        """
        Validate scraped game data - fail fast on bad data

        WHY: Bad scrapes corrupt training data silently.
        This catches issues immediately at scrape time.

        Returns: (is_valid, reason_if_invalid)
        """
        # Required fields
        required = ['year', 'home_team', 'away_team', 'market_spread']

        for field in required:
            if field not in game:
                return False, f"Missing required field: {field}"

            if game[field] is None or (isinstance(game[field], str) and not game[field]):
                return False, f"Empty required field: {field}"

        # Year sanity check
        if not isinstance(game['year'], int) or not (2000 <= game['year'] <= 2030):
            return False, f"Invalid year: {game['year']}"

        # Spread validation
        spread = game['market_spread']
        if not isinstance(spread, (int, float)):
            return False, f"Invalid spread type: {type(spread)} - must be numeric"

        # Sanity check: NCAA spreads rarely exceed ±50
        if abs(spread) > 50:
            return False, f"Unrealistic spread: {spread} (exceeds ±50)"

        # Team name validation
        for team_field in ['home_team', 'away_team']:
            team = game[team_field]
            if len(team) < 2:
                return False, f"Team name too short: '{team}'"
            if len(team) > 100:
                return False, f"Team name too long: '{team}'"

        # Teams can't be identical
        if game['home_team'] == game['away_team']:
            return False, f"Home and away teams identical: '{game['home_team']}'"

        return True, ""

    def _parse_game_card(self, card, year: int, week: int) -> dict:
        """Parse game card to extract spread"""

        # Look for team names
        teams = card.find_all(class_=['covers-CoversConsensusDetailsTable-teamName', 'team-name'])

        if len(teams) < 2:
            return None

        away_team = teams[0].text.strip()
        home_team = teams[1].text.strip()

        # Look for spread
        spread_elements = card.find_all(class_=['spread', 'covers-CoversConsensusDetailsTable-consensus'])

        market_spread = None

        for elem in spread_elements:
            text = elem.text.strip()
            parsed = self._parse_spread_text(text)
            if parsed is not None:
                market_spread = parsed
                break

        if market_spread is None:
            return None

        game = {
            'year': year,
            'week': week,
            'away_team': away_team,
            'home_team': home_team,
            'market_spread': market_spread,
            'source': 'covers_closing'
        }

        # VALIDATION: Fail fast on bad data
        is_valid, reason = self.validate_game(game)
        if not is_valid:
            print(f"   ⚠️  REJECTED: {reason} - {away_team} @ {home_team}")
            return None

        return game

    def _parse_spread_text(self, text: str) -> float:
        """Parse spread from text"""
        import re

        if not text or 'pk' in text.lower() or 'pick' in text.lower():
            return 0.0

        # Look for pattern like "-14.5" or "+7"
        match = re.search(r'([+-]?\d+\.?\d*)', text)

        if match:
            try:
                return float(match.group(1))
            except:
                return None

        return None

    def save_data(self, df: pd.DataFrame, year: int):
        """Save scraped data"""

        if df.empty:
            print(f"\n⚠️  No data to save")
            return

        csv_file = self.output_dir / f"market_spreads_{year}_covers.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Saved to {csv_file}")

        print(f"\n📊 Summary:")
        print(f"   Total games: {len(df)}")
        print(f"   Weeks covered: {df['week'].nunique()}")
        print(f"   Games per week: {len(df) / df['week'].nunique():.1f}")


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python scrape_covers_historical.py YEAR")
        print()
        print("Example: python scrape_covers_historical.py 2023")
        print()
        return

    year = int(sys.argv[1])

    scraper = CoversHistoricalSpreads()

    df = scraper.scrape_season_by_weeks(year)

    scraper.save_data(df, year)


if __name__ == "__main__":
    main()
