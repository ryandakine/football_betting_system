#!/usr/bin/env python3
"""
TeamRankings Historical Closing Lines Scraper
==============================================

TeamRankings.com has historical closing spreads for NCAA games.
This scraper gets the ACTUAL MARKET SPREADS (not just scores).

USAGE:
    python scrape_teamrankings_historical.py 2023 5
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys
from pathlib import Path
import json


class TeamRankingsClosingSpreads:
    """Scrape historical closing spreads from TeamRankings"""

    def __init__(self):
        self.base_url = "https://www.teamrankings.com/ncf/odds-history/results/"

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        }

        self.output_dir = Path("data/market_spreads")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def scrape_season(self, year: int) -> pd.DataFrame:
        """
        Scrape full season of closing spreads

        TeamRankings format:
        https://www.teamrankings.com/ncf/odds-history/results/?year=2023
        """

        print(f"\n{'='*80}")
        print(f"📊 SCRAPING {year} SEASON CLOSING SPREADS")
        print(f"{'='*80}\n")

        url = f"{self.base_url}?year={year}"

        print(f"🌐 URL: {url}")

        try:
            response = requests.get(url, headers=self.headers, timeout=30)

            if response.status_code == 200:
                print(f"✅ Connected successfully")

                soup = BeautifulSoup(response.content, 'html.parser')

                # TeamRankings uses tables with class "tr-table datatable"
                tables = soup.find_all('table', {'class': 'tr-table'})

                if not tables:
                    print(f"⚠️  No tables found - page structure may have changed")
                    # Save HTML for inspection
                    debug_file = self.output_dir / f"teamrankings_{year}_debug.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    print(f"   Saved HTML to {debug_file} for inspection")
                    return pd.DataFrame()

                print(f"📋 Found {len(tables)} table(s)")

                all_games = []

                for table_idx, table in enumerate(tables):
                    print(f"\n   Processing table {table_idx + 1}...")

                    # Get headers
                    headers = []
                    header_row = table.find('thead')
                    if header_row:
                        headers = [th.text.strip() for th in header_row.find_all('th')]

                    print(f"   Headers: {headers}")

                    # Get rows
                    rows = table.find('tbody').find_all('tr') if table.find('tbody') else []

                    print(f"   Found {len(rows)} rows")

                    for row in rows:
                        try:
                            cells = row.find_all('td')

                            if len(cells) < 5:
                                continue

                            # Extract game data
                            # Typical format: Date | Away @ Home | Score | Spread | Result

                            date = cells[0].text.strip() if len(cells) > 0 else ''
                            matchup = cells[1].text.strip() if len(cells) > 1 else ''
                            score = cells[2].text.strip() if len(cells) > 2 else ''
                            spread = cells[3].text.strip() if len(cells) > 3 else ''
                            result = cells[4].text.strip() if len(cells) > 4 else ''

                            # Parse matchup (usually "Away @ Home")
                            if '@' in matchup:
                                parts = matchup.split('@')
                                away_team = parts[0].strip()
                                home_team = parts[1].strip()
                            else:
                                away_team = ''
                                home_team = matchup

                            # Parse spread (e.g., "Alabama -14.5" or "Pick")
                            market_spread = self._parse_spread(spread, home_team)

                            if market_spread is not None and away_team and home_team:
                                all_games.append({
                                    'year': year,
                                    'date': date,
                                    'away_team': away_team,
                                    'home_team': home_team,
                                    'market_spread': market_spread,
                                    'score': score,
                                    'result': result,
                                    'source': 'teamrankings_closing'
                                })

                        except Exception as e:
                            continue

                    print(f"   Extracted {len(all_games)} games so far")

                if all_games:
                    df = pd.DataFrame(all_games)
                    print(f"\n✅ Successfully scraped {len(df)} games with closing spreads")
                    return df
                else:
                    print(f"\n⚠️  No games extracted - check page structure")
                    return pd.DataFrame()

            elif response.status_code == 403:
                print(f"❌ 403 Forbidden - Need to adjust user agent or use VPN")
                return pd.DataFrame()

            else:
                print(f"❌ Error {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            print(f"❌ Error: {e}")
            return pd.DataFrame()

    def _parse_spread(self, spread_text: str, home_team: str) -> float:
        """
        Parse spread from text

        Examples:
        - "Alabama -14.5" → -14.5 (from home team perspective)
        - "Pick" → 0.0
        - "+7.5" → +7.5
        """

        if not spread_text or spread_text.lower() in ['pick', 'pk']:
            return 0.0

        # Remove team name
        spread_text = spread_text.replace(home_team, '').strip()

        # Extract number
        import re
        match = re.search(r'([+-]?\d+\.?\d*)', spread_text)

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

        # CSV
        csv_file = self.output_dir / f"market_spreads_{year}_teamrankings.csv"
        df.to_csv(csv_file, index=False)
        print(f"\n💾 Saved to {csv_file}")

        # JSON (for easier inspection)
        json_file = self.output_dir / f"market_spreads_{year}_teamrankings.json"
        df.to_json(json_file, orient='records', indent=2)
        print(f"💾 Saved to {json_file}")

        # Summary
        print(f"\n📊 Summary:")
        print(f"   Total games: {len(df)}")
        print(f"   Unique teams: {len(set(df['home_team']) | set(df['away_team']))}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python scrape_teamrankings_historical.py YEAR")
        print()
        print("Example: python scrape_teamrankings_historical.py 2023")
        print()
        print("This will scrape CLOSING SPREADS (actual market data) for the season")
        return

    year = int(sys.argv[1])

    scraper = TeamRankingsClosingSpreads()

    df = scraper.scrape_season(year)

    scraper.save_data(df, year)

    if not df.empty:
        print("\n" + "="*80)
        print("✅ SUCCESS")
        print("="*80)
        print()
        print(f"Got {len(df)} games with ACTUAL MARKET SPREADS")
        print()
        print("Sample data:")
        print(df.head(10).to_string(index=False))
        print()
    else:
        print("\n" + "="*80)
        print("⚠️  SCRAPING FAILED")
        print("="*80)
        print()
        print("Possible issues:")
        print("1. Site blocking automated requests (try VPN)")
        print("2. Page structure changed (check debug HTML file)")
        print("3. Network restrictions (run on different network)")
        print()
        print("Alternative: Pay $99 for Sports Insights historical data")
        print("URL: https://www.sportsinsights.com/")
        print()


if __name__ == "__main__":
    main()
