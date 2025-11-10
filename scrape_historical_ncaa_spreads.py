#!/usr/bin/env python3
"""
Historical NCAA Spread Scraper
==============================

Multi-source scraper to collect historical NCAA football spreads.
Sources:
1. SportsBookReviewsOnline (covers-like archive)
2. TeamRankings.com (has historical odds)
3. OddsShark archive
4. ESPN BPI historical data
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class NCAAHistoricalSpreadScraper:
    """Scrapes historical NCAA spreads from multiple sources"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache_dir = Path("data/historical_spreads")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def scrape_teamrankings(self, year: int, week: int) -> List[Dict]:
        """
        Scrape TeamRankings.com for historical spreads
        Example URL: https://www.teamrankings.com/ncf/odds-history/results/
        """
        print(f"📡 Scraping TeamRankings.com for {year} Week {week}...")

        url = f"https://www.teamrankings.com/ncf/schedules/?season={year}&week={week}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            games = []

            # Find game tables
            tables = soup.find_all('table', class_='tr-table')

            for table in tables:
                rows = table.find_all('tr')[1:]  # Skip header

                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 5:
                        try:
                            away_team = cells[0].get_text(strip=True)
                            home_team = cells[1].get_text(strip=True)

                            # Look for spread in various cells
                            spread_text = None
                            for cell in cells:
                                text = cell.get_text(strip=True)
                                if '-' in text or '+' in text:
                                    try:
                                        spread_text = text
                                        break
                                    except:
                                        continue

                            if spread_text and away_team and home_team:
                                # Parse spread
                                spread = self._parse_spread(spread_text)

                                games.append({
                                    'source': 'teamrankings',
                                    'year': year,
                                    'week': week,
                                    'away_team': away_team,
                                    'home_team': home_team,
                                    'spread': spread,
                                    'scraped_at': datetime.now().isoformat()
                                })

                        except Exception as e:
                            continue

            print(f"   ✅ Found {len(games)} games")
            return games

        except Exception as e:
            print(f"   ❌ Error scraping TeamRankings: {e}")
            return []

    def scrape_oddsshark_archive(self, year: int, date: str) -> List[Dict]:
        """
        Scrape OddsShark archives
        Example: https://www.oddsshark.com/ncaaf/scores/{date}
        """
        print(f"📡 Scraping OddsShark for {date}...")

        url = f"https://www.oddsshark.com/ncaaf/scores/{date}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            games = []

            # Find game containers
            game_cards = soup.find_all('div', class_=['game-card', 'matchup-card'])

            for card in game_cards:
                try:
                    teams = card.find_all(class_=['team-name', 'team'])
                    if len(teams) >= 2:
                        away_team = teams[0].get_text(strip=True)
                        home_team = teams[1].get_text(strip=True)

                        # Find spread
                        spread_elem = card.find(class_=['spread', 'line'])
                        if spread_elem:
                            spread_text = spread_elem.get_text(strip=True)
                            spread = self._parse_spread(spread_text)

                            games.append({
                                'source': 'oddsshark',
                                'date': date,
                                'away_team': away_team,
                                'home_team': home_team,
                                'spread': spread,
                                'scraped_at': datetime.now().isoformat()
                            })

                except Exception as e:
                    continue

            print(f"   ✅ Found {len(games)} games")
            return games

        except Exception as e:
            print(f"   ❌ Error scraping OddsShark: {e}")
            return []

    def scrape_espn_bpi_archive(self, year: int) -> List[Dict]:
        """
        Scrape ESPN's historical BPI/FPI data
        This often includes historical spreads
        """
        print(f"📡 Scraping ESPN BPI data for {year}...")

        url = f"https://www.espn.com/college-football/schedule/_/season/{year}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            games = []

            # ESPN structure varies, look for game elements
            game_divs = soup.find_all('div', class_=['event', 'game'])

            for div in game_divs:
                try:
                    teams = div.find_all(class_='team')
                    if len(teams) >= 2:
                        away_team = teams[0].get_text(strip=True)
                        home_team = teams[1].get_text(strip=True)

                        # Look for odds/spread
                        odds = div.find(class_=['odds', 'line', 'spread'])
                        if odds:
                            spread = self._parse_spread(odds.get_text(strip=True))

                            games.append({
                                'source': 'espn',
                                'year': year,
                                'away_team': away_team,
                                'home_team': home_team,
                                'spread': spread,
                                'scraped_at': datetime.now().isoformat()
                            })

                except Exception as e:
                    continue

            print(f"   ✅ Found {len(games)} games")
            return games

        except Exception as e:
            print(f"   ❌ Error scraping ESPN: {e}")
            return []

    def scrape_covers_archive(self, year: int, week: int) -> List[Dict]:
        """
        Scrape Covers.com historical data
        Covers has one of the best historical databases
        """
        print(f"📡 Scraping Covers.com for {year} Week {week}...")

        # Covers URL format
        url = f"https://www.covers.com/sports/ncaaf/matchups?week={week}&year={year}"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            games = []

            # Find matchup cards
            matchups = soup.find_all('div', class_=['cmg_matchup', 'matchup'])

            for matchup in matchups:
                try:
                    # Parse team names
                    teams = matchup.find_all(class_=['cmg_team_name', 'team-name'])
                    if len(teams) >= 2:
                        away_team = teams[0].get_text(strip=True)
                        home_team = teams[1].get_text(strip=True)

                        # Find closing line
                        spread_elem = matchup.find(class_=['cmg_spread', 'closing-line', 'spread'])
                        if spread_elem:
                            spread_text = spread_elem.get_text(strip=True)
                            spread = self._parse_spread(spread_text)

                            games.append({
                                'source': 'covers',
                                'year': year,
                                'week': week,
                                'away_team': away_team,
                                'home_team': home_team,
                                'spread': spread,
                                'scraped_at': datetime.now().isoformat()
                            })

                except Exception as e:
                    continue

            print(f"   ✅ Found {len(games)} games")
            return games

        except Exception as e:
            print(f"   ❌ Error scraping Covers: {e}")
            return []

    def _parse_spread(self, spread_text: str) -> Optional[float]:
        """Parse spread from various text formats"""
        if not spread_text:
            return None

        # Clean up text
        spread_text = spread_text.strip().replace('½', '.5')

        # Try to extract number
        import re
        match = re.search(r'([+-]?\d+\.?\d*)', spread_text)
        if match:
            try:
                return float(match.group(1))
            except:
                return None

        return None

    def scrape_all_sources(self, year: int, week: int) -> List[Dict]:
        """Scrape all sources and combine results"""
        print(f"\n{'='*80}")
        print(f"🔍 Scraping Historical Spreads: {year} Week {week}")
        print(f"{'='*80}\n")

        all_games = []

        # Try each source
        sources = [
            ('teamrankings', lambda: self.scrape_teamrankings(year, week)),
            ('covers', lambda: self.scrape_covers_archive(year, week)),
        ]

        for source_name, scraper_func in sources:
            try:
                games = scraper_func()
                all_games.extend(games)
                time.sleep(2)  # Be respectful, wait between requests
            except Exception as e:
                print(f"⚠️  {source_name} failed: {e}")

        # Save to cache
        self._save_to_cache(year, week, all_games)

        print(f"\n✅ Total: {len(all_games)} games scraped")
        return all_games

    def _save_to_cache(self, year: int, week: int, games: List[Dict]):
        """Save scraped data to cache"""
        cache_file = self.cache_dir / f"ncaa_{year}_week{week}.json"

        with open(cache_file, 'w') as f:
            json.dump({
                'year': year,
                'week': week,
                'games': games,
                'scraped_at': datetime.now().isoformat(),
                'count': len(games)
            }, f, indent=2)

        print(f"💾 Cached to {cache_file}")

    def match_with_our_data(self, year: int) -> pd.DataFrame:
        """
        Match scraped spreads with our game data
        Returns DataFrame with: game_id, our_prediction, market_spread
        """
        print(f"\n🔗 Matching scraped spreads with our game data...")

        # Load our games
        from ncaa_models.feature_engineering import NCAAFeatureEngineer
        engineer = NCAAFeatureEngineer()
        our_games = engineer.load_season_data(year)

        # Load all cached spreads for this year
        cached_files = list(self.cache_dir.glob(f"ncaa_{year}_*.json"))

        matched_games = []

        for cache_file in cached_files:
            with open(cache_file) as f:
                data = json.load(f)

            for scraped_game in data['games']:
                # Try to match with our games
                away = scraped_game['away_team']
                home = scraped_game['home_team']

                for our_game in our_games:
                    our_away = our_game.get('awayTeam', '')
                    our_home = our_game.get('homeTeam', '')

                    # Fuzzy match team names
                    if (self._team_match(away, our_away) and
                        self._team_match(home, our_home)):

                        matched_games.append({
                            'game_id': our_game.get('id'),
                            'year': year,
                            'week': our_game.get('week'),
                            'away_team': our_away,
                            'home_team': our_home,
                            'market_spread': scraped_game['spread'],
                            'source': scraped_game['source'],
                        })
                        break

        df = pd.DataFrame(matched_games)
        print(f"✅ Matched {len(df)} games")

        return df

    def _team_match(self, name1: str, name2: str) -> bool:
        """Fuzzy match team names"""
        name1 = name1.lower().strip()
        name2 = name2.lower().strip()

        # Direct match
        if name1 == name2:
            return True

        # Check if one contains the other
        if name1 in name2 or name2 in name1:
            return True

        # Check nickname match (e.g., "Alabama" and "Crimson Tide")
        # This would need a lookup table in production

        return False


def scrape_full_season(year: int):
    """Scrape an entire season (weeks 1-15)"""
    scraper = NCAAHistoricalSpreadScraper()

    print(f"\n🏈 SCRAPING FULL {year} SEASON")
    print("="*80)

    all_scraped = []

    for week in range(1, 16):  # Weeks 1-15
        games = scraper.scrape_all_sources(year, week)
        all_scraped.extend(games)
        time.sleep(3)  # Be nice to servers

    print(f"\n✅ COMPLETE: {len(all_scraped)} total games scraped for {year}")

    # Match with our data
    matched_df = scraper.match_with_our_data(year)

    # Save matched data
    output_file = f"data/market_spreads_{year}.csv"
    matched_df.to_csv(output_file, index=False)
    print(f"💾 Saved matched spreads to {output_file}")

    return matched_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    else:
        year = 2024

    print("="*80)
    print("🏈 NCAA HISTORICAL SPREAD SCRAPER")
    print("="*80)
    print()
    print("⚠️  NOTE: Web scraping may violate some sites' Terms of Service")
    print("    Consider using official APIs or purchasing data instead:")
    print("    - Sports Insights API")
    print("    - The Odds API (historical)")
    print("    - SportsDataIO")
    print()
    print(f"Proceeding to scrape {year} season...")
    print()

    # Scrape full season
    df = scrape_full_season(year)

    print("\n📊 Sample of scraped data:")
    print(df.head(10))
