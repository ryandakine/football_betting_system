#!/usr/bin/env python3
"""
FREE Market Data Sources - Aggressive Scraper
==============================================

FREE sources that actually work:
1. OddsPortal.com - Free historical odds (best source!)
2. Vegas Insider - Free historical lines
3. Archive.org - Wayback Machine for old odds
4. Kaggle datasets - Check for NCAA betting data
5. GitHub repos - Pre-scraped datasets
6. Reddit datasets - r/sportsbook communities
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd


class FreeMarketDataCollector:
    """Collect FREE historical NCAA spreads"""

    def __init__(self):
        # Use different user agents to avoid blocks
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        ]
        self.cache_dir = Path("data/market_spreads")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_oddsportal_data(self, year: int) -> List[Dict]:
        """
        OddsPortal.com - FREE and comprehensive!
        URL: https://www.oddsportal.com/american-football/usa/ncaa/
        """
        print(f"\n🎯 OddsPortal.com (FREE - Best Source!)")
        print(f"   Fetching {year} NCAA spreads...")

        # OddsPortal season format
        season_str = f"{year}-{year+1}" if year < datetime.now().year else f"{year}"

        url = f"https://www.oddsportal.com/american-football/usa/ncaa-{season_str}/results/"

        headers = {
            'User-Agent': self.user_agents[0],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
        }

        try:
            response = requests.get(url, headers=headers, timeout=15)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # OddsPortal structure: find game rows
                games = []
                game_rows = soup.find_all('div', class_=['eventRow', 'table-main__row'])

                for row in game_rows:
                    try:
                        # Parse team names
                        teams = row.find_all(class_=['name', 'participant-name'])
                        if len(teams) >= 2:
                            away_team = teams[0].get_text(strip=True)
                            home_team = teams[1].get_text(strip=True)

                            # Parse spread
                            spread_cell = row.find(class_=['odds-ah', 'odds', 'asian-handicap'])
                            if spread_cell:
                                spread_text = spread_cell.get_text(strip=True)
                                spread = self._parse_spread(spread_text)

                                if spread is not None:
                                    games.append({
                                        'source': 'oddsportal',
                                        'year': year,
                                        'away_team': away_team,
                                        'home_team': home_team,
                                        'spread': spread,
                                    })

                    except Exception as e:
                        continue

                print(f"   ✅ Found {len(games)} games")
                return games

            else:
                print(f"   ⚠️  Status {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        return []

    def get_vegasinsider_data(self, year: int) -> List[Dict]:
        """
        Vegas Insider - FREE historical lines
        URL: https://www.vegasinsider.com/college-football/matchups/
        """
        print(f"\n🎯 Vegas Insider (FREE)")
        print(f"   Fetching {year} NCAA spreads...")

        url = f"https://www.vegasinsider.com/college-football/matchups/matchups.cfm/season/{year}"

        headers = {'User-Agent': self.user_agents[1]}

        try:
            response = requests.get(url, headers=headers, timeout=15)
            print(f"   Status: {response.status_code}")

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                games = []

                # Find matchup tables
                tables = soup.find_all('table', class_=['frodds-table', 'viBodyBorderNorm'])

                for table in tables:
                    rows = table.find_all('tr')[1:]  # Skip header

                    for row in rows:
                        try:
                            cells = row.find_all('td')
                            if len(cells) >= 3:
                                # Parse teams and spread
                                teams_cell = cells[0]
                                teams = teams_cell.get_text(strip=True).split(' at ')

                                if len(teams) == 2:
                                    away_team = teams[0]
                                    home_team = teams[1]

                                    # Find spread cell
                                    for cell in cells[1:]:
                                        text = cell.get_text(strip=True)
                                        if 'PK' in text or '-' in text or '+' in text:
                                            spread = self._parse_spread(text)
                                            if spread is not None:
                                                games.append({
                                                    'source': 'vegasinsider',
                                                    'year': year,
                                                    'away_team': away_team,
                                                    'home_team': home_team,
                                                    'spread': spread,
                                                })
                                                break

                        except Exception as e:
                            continue

                print(f"   ✅ Found {len(games)} games")
                return games

            else:
                print(f"   ⚠️  Status {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        return []

    def check_kaggle_datasets(self) -> List[str]:
        """Check Kaggle for NCAA betting datasets"""
        print(f"\n🎯 Kaggle Datasets")
        print("   Searching for NCAA betting data...")

        # Search terms
        search_terms = ['ncaa football betting', 'college football spreads', 'ncaa odds']

        datasets = []

        for term in search_terms:
            url = f"https://www.kaggle.com/search?q={term.replace(' ', '+')}"

            try:
                response = requests.get(url, headers={'User-Agent': self.user_agents[2]}, timeout=10)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Find dataset links
                    links = soup.find_all('a', href=re.compile(r'/datasets/'))

                    for link in links[:5]:  # Top 5 results
                        href = link.get('href')
                        if href:
                            full_url = f"https://www.kaggle.com{href}"
                            if full_url not in datasets:
                                datasets.append(full_url)
                                print(f"   📊 Found: {full_url}")

            except Exception as e:
                print(f"   ⚠️  Search failed: {e}")

        return datasets

    def check_github_repos(self) -> List[str]:
        """Check GitHub for pre-scraped NCAA betting data"""
        print(f"\n🎯 GitHub Repositories")
        print("   Searching for NCAA betting datasets...")

        # Search GitHub API
        search_queries = [
            'ncaa football betting data',
            'college football spreads dataset',
            'ncaa odds historical',
        ]

        repos = []

        for query in search_queries:
            url = f"https://api.github.com/search/repositories?q={query.replace(' ', '+')}&sort=stars"

            try:
                response = requests.get(url, headers={'User-Agent': self.user_agents[0]}, timeout=10)

                if response.status_code == 200:
                    data = response.json()

                    for item in data.get('items', [])[:5]:
                        repo_url = item.get('html_url')
                        description = item.get('description', 'No description')
                        stars = item.get('stargazers_count', 0)

                        if repo_url not in repos:
                            repos.append(repo_url)
                            print(f"   ⭐ {stars} stars: {repo_url}")
                            print(f"      {description}")

            except Exception as e:
                print(f"   ⚠️  Search failed: {e}")

        return repos

    def _parse_spread(self, text: str) -> float:
        """Parse spread from various formats"""
        if not text or text == 'PK':
            return 0.0

        # Clean text
        text = text.strip().replace('½', '.5').replace('PK', '0')

        # Extract number
        match = re.search(r'([+-]?\d+\.?\d*)', text)
        if match:
            try:
                return float(match.group(1))
            except:
                return None

        return None

    def download_from_url(self, url: str, output_file: str):
        """Download dataset from direct URL"""
        print(f"\n📥 Downloading from {url}...")

        try:
            response = requests.get(url, headers={'User-Agent': self.user_agents[0]}, timeout=30)

            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)

                print(f"   ✅ Downloaded to {output_file}")
                return True

            else:
                print(f"   ❌ Failed: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        return False


def main():
    """Try ALL free sources"""
    print("="*80)
    print("🎯 FREE NCAA MARKET DATA COLLECTOR")
    print("="*80)
    print()
    print("Trying multiple FREE sources...")
    print()

    collector = FreeMarketDataCollector()

    # Try each source
    all_games = []

    # 1. OddsPortal (best free source)
    for year in [2024, 2023, 2022]:
        games = collector.get_oddsportal_data(year)
        all_games.extend(games)
        time.sleep(2)

    # 2. Vegas Insider
    for year in [2024, 2023]:
        games = collector.get_vegasinsider_data(year)
        all_games.extend(games)
        time.sleep(2)

    # 3. Check Kaggle
    kaggle_datasets = collector.check_kaggle_datasets()

    # 4. Check GitHub
    github_repos = collector.check_github_repos()

    # Display results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"Total games scraped: {len(all_games)}")

    if all_games:
        df = pd.DataFrame(all_games)
        output_file = "data/FREE_market_spreads.csv"
        df.to_csv(output_file, index=False)
        print(f"✅ Saved to {output_file}")

        print("\nSample:")
        print(df.head(10))

    if kaggle_datasets:
        print(f"\n📊 Kaggle Datasets Found: {len(kaggle_datasets)}")
        print("   Download manually from:")
        for ds in kaggle_datasets:
            print(f"   - {ds}")

    if github_repos:
        print(f"\n💻 GitHub Repos Found: {len(github_repos)}")
        print("   Check these repos for datasets:")
        for repo in github_repos:
            print(f"   - {repo}")

    # Try direct dataset links
    print("\n" + "="*80)
    print("💡 DIRECT FREE DATASET SOURCES:")
    print("="*80)
    print()
    print("1. Sports Reference (free CSV exports):")
    print("   https://www.sports-reference.com/cfb/years/2024-schedule.html")
    print()
    print("2. CFBStats (free historical data):")
    print("   http://www.cfbstats.com/")
    print()
    print("3. Reddit r/CFBAnalysis (community datasets):")
    print("   https://www.reddit.com/r/CFBAnalysis/")
    print()
    print("4. Pro Football Reference (sister site):")
    print("   https://www.pro-football-reference.com/")
    print()
    print("5. Academic Datasets:")
    print("   - Google Dataset Search: https://datasetsearch.research.google.com/")
    print("   - Search: 'NCAA football betting historical odds'")


if __name__ == "__main__":
    from datetime import datetime
    main()
