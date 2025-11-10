#!/usr/bin/env python3
"""
Archive.org Wayback Machine - NCAA Spreads Scraper
===================================================

GENIUS APPROACH: Scrape ARCHIVED snapshots of betting sites
- Archive.org saves historical snapshots of websites
- Less likely to block (public archive service)
- Can access old betting lines from 2015-2024
- 100% FREE and legal
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import re


class WaybackSpreadsScraper:
    """Scrape historical spreads from Archive.org Wayback Machine"""

    def __init__(self):
        self.wayback_api = "https://web.archive.org/web"
        self.wayback_cdx = "https://web.archive.org/cdx/search/cdx"

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        self.cache_dir = Path("data/wayback_spreads")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def find_snapshots(self, url: str, year: int, month: int = None) -> List[Dict]:
        """
        Find all Archive.org snapshots for a URL in a given timeframe

        Args:
            url: Original URL to find snapshots of
            year: Year to search
            month: Optional specific month

        Returns:
            List of snapshot timestamps
        """
        print(f"🔍 Finding Archive.org snapshots of {url}...")

        # CDX API query
        params = {
            'url': url,
            'matchType': 'prefix',
            'output': 'json',
            'fl': 'timestamp,original,statuscode',
            'filter': 'statuscode:200',
            'collapse': 'timestamp:8',  # One per day
        }

        # Add date filter
        if month:
            from_date = f"{year}{month:02d}01"
            to_date = f"{year}{month:02d}31"
        else:
            from_date = f"{year}0101"
            to_date = f"{year}1231"

        params['from'] = from_date
        params['to'] = to_date

        try:
            response = requests.get(self.wayback_cdx, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Skip header row
                snapshots = []
                for row in data[1:]:
                    timestamp, original_url, status = row
                    snapshots.append({
                        'timestamp': timestamp,
                        'url': original_url,
                        'archive_url': f"{self.wayback_api}/{timestamp}/{original_url}"
                    })

                print(f"   ✅ Found {len(snapshots)} snapshots")
                return snapshots

        except Exception as e:
            print(f"   ❌ Error: {e}")

        return []

    def scrape_oddsshark_snapshot(self, snapshot_url: str) -> List[Dict]:
        """Scrape OddsShark from an Archive.org snapshot"""
        print(f"   📡 Scraping snapshot...")

        try:
            response = requests.get(snapshot_url, headers=self.headers, timeout=20)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                games = []

                # Try multiple OddsShark structures (they've changed over years)

                # Structure 1: Game cards
                game_cards = soup.find_all('div', class_=['game-card', 'matchup-card', 'game'])

                for card in game_cards:
                    try:
                        # Find teams
                        teams = card.find_all(class_=['team-name', 'team', 'name'])
                        if len(teams) >= 2:
                            away_team = teams[0].get_text(strip=True)
                            home_team = teams[1].get_text(strip=True)

                            # Find spread
                            spread_elem = card.find(class_=['spread', 'line', 'closing-line'])
                            if spread_elem:
                                spread_text = spread_elem.get_text(strip=True)
                                spread = self._parse_spread(spread_text)

                                if spread is not None:
                                    games.append({
                                        'away_team': away_team,
                                        'home_team': home_team,
                                        'spread': spread,
                                        'source': 'oddsshark_archive'
                                    })

                    except Exception as e:
                        continue

                # Structure 2: Tables
                if not games:
                    tables = soup.find_all('table', class_=['odds-table', 'matchups'])

                    for table in tables:
                        rows = table.find_all('tr')[1:]  # Skip header

                        for row in rows:
                            try:
                                cells = row.find_all('td')
                                if len(cells) >= 3:
                                    # Parse teams and spread
                                    teams_text = cells[0].get_text(strip=True)

                                    # Look for "Team A @ Team B" format
                                    if '@' in teams_text:
                                        teams = teams_text.split('@')
                                        away_team = teams[0].strip()
                                        home_team = teams[1].strip()

                                        # Find spread in other cells
                                        for cell in cells[1:]:
                                            text = cell.get_text(strip=True)
                                            if '-' in text or '+' in text:
                                                spread = self._parse_spread(text)
                                                if spread is not None:
                                                    games.append({
                                                        'away_team': away_team,
                                                        'home_team': home_team,
                                                        'spread': spread,
                                                        'source': 'oddsshark_archive'
                                                    })
                                                    break

                            except Exception as e:
                                continue

                print(f"      Found {len(games)} games")
                return games

        except Exception as e:
            print(f"      ❌ Error: {e}")

        return []

    def scrape_teamrankings_snapshot(self, snapshot_url: str) -> List[Dict]:
        """Scrape TeamRankings from an Archive.org snapshot"""
        print(f"   📡 Scraping snapshot...")

        try:
            response = requests.get(snapshot_url, headers=self.headers, timeout=20)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                games = []

                # TeamRankings uses tables
                tables = soup.find_all('table', class_=['tr-table', 'datatable'])

                for table in tables:
                    rows = table.find_all('tr')[1:]  # Skip header

                    for row in rows:
                        try:
                            cells = row.find_all('td')
                            if len(cells) >= 4:
                                # TeamRankings format varies
                                away_team = cells[0].get_text(strip=True)
                                home_team = cells[1].get_text(strip=True)

                                # Look for spread
                                for cell in cells[2:]:
                                    text = cell.get_text(strip=True)
                                    if '-' in text or '+' in text or 'PK' in text:
                                        spread = self._parse_spread(text)
                                        if spread is not None:
                                            games.append({
                                                'away_team': away_team,
                                                'home_team': home_team,
                                                'spread': spread,
                                                'source': 'teamrankings_archive'
                                            })
                                            break

                        except Exception as e:
                            continue

                print(f"      Found {len(games)} games")
                return games

        except Exception as e:
            print(f"      ❌ Error: {e}")

        return []

    def _parse_spread(self, text: str) -> Optional[float]:
        """Parse spread from text"""
        if not text:
            return None

        # Clean
        text = text.strip().replace('½', '.5').replace('PK', '0')

        # Extract number
        match = re.search(r'([+-]?\d+\.?\d*)', text)
        if match:
            try:
                return float(match.group(1))
            except:
                return None

        return None

    def scrape_week_from_archive(self, year: int, week: int) -> List[Dict]:
        """
        Scrape a specific week using Wayback Machine

        Strategy:
        1. Find snapshots from that week's date range
        2. Scrape archived pages
        3. Combine results
        """
        print(f"\n{'='*80}")
        print(f"📅 Scraping {year} Week {week} via Archive.org")
        print(f"{'='*80}\n")

        # Calculate approximate dates for this week
        # NCAA season typically starts late August/early September
        season_start = datetime(year, 9, 1)  # Approximate
        week_start = season_start + timedelta(weeks=week-1)

        month = week_start.month

        all_games = []

        # Try multiple sources
        sources = [
            {
                'name': 'OddsShark',
                'url': f'https://www.oddsshark.com/ncaaf/scores',
                'scraper': self.scrape_oddsshark_snapshot
            },
            {
                'name': 'TeamRankings',
                'url': f'https://www.teamrankings.com/ncf/schedules/?season={year}&week={week}',
                'scraper': self.scrape_teamrankings_snapshot
            },
        ]

        for source in sources:
            print(f"🎯 {source['name']}...")

            # Find snapshots
            snapshots = self.find_snapshots(source['url'], year, month)

            if not snapshots:
                print(f"   ⚠️  No snapshots found")
                continue

            # Try first few snapshots (most likely to have data)
            for snapshot in snapshots[:3]:
                games = source['scraper'](snapshot['archive_url'])

                if games:
                    # Add year/week info
                    for game in games:
                        game['year'] = year
                        game['week'] = week
                        game['snapshot_date'] = snapshot['timestamp']

                    all_games.extend(games)
                    break  # Got data, move to next source

                time.sleep(2)  # Be nice to Archive.org

            time.sleep(3)  # Between sources

        print(f"\n✅ Total: {len(all_games)} games from Archive.org")

        # Save to cache
        if all_games:
            cache_file = self.cache_dir / f"ncaa_{year}_week{week}_wayback.json"
            with open(cache_file, 'w') as f:
                json.dump({
                    'year': year,
                    'week': week,
                    'games': all_games,
                    'scraped_at': datetime.now().isoformat(),
                    'count': len(all_games)
                }, f, indent=2)
            print(f"💾 Cached to {cache_file}")

        return all_games

    def match_with_our_data(self, year: int, week: int) -> pd.DataFrame:
        """Match archived spreads with our game data"""
        print(f"\n🔗 Matching archived spreads with our data...")

        # Load cached data
        cache_file = self.cache_dir / f"ncaa_{year}_week{week}_wayback.json"

        if not cache_file.exists():
            print(f"   ❌ No cached data for {year} Week {week}")
            return pd.DataFrame()

        with open(cache_file) as f:
            data = json.load(f)

        # Load our games
        from ncaa_models.feature_engineering import NCAAFeatureEngineer
        engineer = NCAAFeatureEngineer()
        our_games = engineer.load_season_data(year)

        # Filter to this week
        our_games = [g for g in our_games if g.get('week') == week]

        matched_games = []

        for archived_game in data['games']:
            away = archived_game['away_team']
            home = archived_game['home_team']

            for our_game in our_games:
                our_away = our_game.get('awayTeam', '')
                our_home = our_game.get('homeTeam', '')

                # Fuzzy match
                if (self._team_match(away, our_away) and
                    self._team_match(home, our_home)):

                    matched_games.append({
                        'game_id': our_game.get('id'),
                        'year': year,
                        'week': week,
                        'away_team': our_away,
                        'home_team': our_home,
                        'market_spread': archived_game['spread'],
                        'source': archived_game['source'],
                    })
                    break

        df = pd.DataFrame(matched_games)
        print(f"✅ Matched {len(df)} games")

        return df

    def _team_match(self, name1: str, name2: str) -> bool:
        """Fuzzy team name matching"""
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()

        if name1 == name2:
            return True

        if name1 in name2 or name2 in name1:
            return True

        # Remove common words
        for word in ['university', 'state', 'college', 'of']:
            name1 = name1.replace(word, '').strip()
            name2 = name2.replace(word, '').strip()

        if name1 == name2:
            return True

        return False


def test_wayback_scraper():
    """Test if Archive.org approach works"""
    scraper = WaybackSpreadsScraper()

    print("="*80)
    print("🎯 TESTING ARCHIVE.ORG WAYBACK MACHINE APPROACH")
    print("="*80)
    print()
    print("This is a FREE, legal way to get historical market spreads!")
    print("We're accessing public archives instead of live sites.")
    print()

    # Test with a recent week that should have good snapshots
    test_year = 2023
    test_week = 5  # Mid-season, should have lots of games

    print(f"Testing with {test_year} Week {test_week}...")
    print()

    games = scraper.scrape_week_from_archive(test_year, test_week)

    if games:
        print(f"\n🎉 SUCCESS! Found {len(games)} games via Archive.org")
        print("\nSample games:")
        for game in games[:5]:
            print(f"   {game['away_team']} @ {game['home_team']}: {game['spread']}")

        # Try matching
        matched_df = scraper.match_with_our_data(test_year, test_week)

        if len(matched_df) > 0:
            print(f"\n✅ MATCHED {len(matched_df)} games with our data!")
            print("\nThis approach WORKS! We can use this to get historical spreads.")
            return True
        else:
            print("\n⚠️  Found games but couldn't match with our data")
            print("   May need better team name matching")
            return False
    else:
        print("\n❌ No games found via Archive.org")
        print("   Archive.org may not have snapshots with odds data")
        return False


if __name__ == "__main__":
    success = test_wayback_scraper()

    if success:
        print("\n" + "="*80)
        print("✅ WAYBACK MACHINE APPROACH WORKS!")
        print("="*80)
        print()
        print("Next steps:")
        print("1. Run full scrape for all weeks 2015-2024")
        print("2. Match with our game data")
        print("3. Save to market_spreads_YEAR.csv")
        print("4. Run realistic backtest")
    else:
        print("\n" + "="*80)
        print("❌ WAYBACK MACHINE APPROACH FAILED")
        print("="*80)
        print()
        print("Fallback plan:")
        print("1. Prepare live NCAA system for 2025 season")
        print("2. Forward test with real-time odds")
        print("3. Save $99 for historical data purchase")
