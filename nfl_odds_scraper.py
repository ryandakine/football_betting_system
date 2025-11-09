#!/usr/bin/env python3
"""
NFL Odds Web Scraper
====================
Scrapes live NFL odds from public websites when API access is blocked.
Supports multiple sources for reliability.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLOddsScraper:
    """Scrapes NFL odds from multiple public sources"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def scrape_espn(self) -> List[Dict]:
        """Scrape NFL odds from ESPN"""
        logger.info("🏈 Scraping ESPN for NFL odds...")

        try:
            url = "https://www.espn.com/nfl/lines"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            games = []

            # ESPN has odds in table format
            # This is a simplified parser - may need adjustment based on actual HTML
            tables = soup.find_all('table', class_=re.compile('odds|lines', re.I))

            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        # Extract team names and odds
                        game_data = {
                            'source': 'ESPN',
                            'scraped_at': datetime.now().isoformat(),
                            'raw_data': [cell.get_text(strip=True) for cell in cells]
                        }
                        games.append(game_data)

            logger.info(f"✅ ESPN: Found {len(games)} entries")
            return games

        except Exception as e:
            logger.error(f"❌ ESPN scraping failed: {e}")
            return []

    def scrape_odds_shark(self) -> List[Dict]:
        """Scrape NFL odds from OddsShark"""
        logger.info("🦈 Scraping OddsShark for NFL odds...")

        try:
            url = "https://www.oddsshark.com/nfl/odds"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            games = []

            # Look for game containers
            game_rows = soup.find_all('div', class_=re.compile('game|matchup|odds', re.I))

            for game in game_rows:
                game_text = game.get_text(strip=True)
                if any(keyword in game_text.lower() for keyword in ['vs', '@', 'at']):
                    game_data = {
                        'source': 'OddsShark',
                        'scraped_at': datetime.now().isoformat(),
                        'raw_text': game_text
                    }
                    games.append(game_data)

            logger.info(f"✅ OddsShark: Found {len(games)} entries")
            return games

        except Exception as e:
            logger.error(f"❌ OddsShark scraping failed: {e}")
            return []

    def scrape_covers(self) -> List[Dict]:
        """Scrape NFL odds from Covers.com"""
        logger.info("📰 Scraping Covers for NFL odds...")

        try:
            url = "https://www.covers.com/sport/football/nfl/odds"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            games = []

            # Covers has a JSON data embedded in script tags sometimes
            scripts = soup.find_all('script', type='application/json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    games.append({
                        'source': 'Covers',
                        'scraped_at': datetime.now().isoformat(),
                        'json_data': data
                    })
                except:
                    pass

            logger.info(f"✅ Covers: Found {len(games)} entries")
            return games

        except Exception as e:
            logger.error(f"❌ Covers scraping failed: {e}")
            return []

    def scrape_action_network(self) -> List[Dict]:
        """Scrape NFL odds from Action Network"""
        logger.info("⚡ Scraping Action Network for NFL odds...")

        try:
            url = "https://www.actionnetwork.com/nfl/odds"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            games = []

            # Action Network often has data in data attributes
            game_elements = soup.find_all(attrs={'data-game-id': True})

            for elem in game_elements:
                game_data = {
                    'source': 'ActionNetwork',
                    'scraped_at': datetime.now().isoformat(),
                    'game_id': elem.get('data-game-id'),
                    'raw_html': str(elem)[:500]  # First 500 chars
                }
                games.append(game_data)

            logger.info(f"✅ Action Network: Found {len(games)} entries")
            return games

        except Exception as e:
            logger.error(f"❌ Action Network scraping failed: {e}")
            return []

    def scrape_bovada_style(self) -> List[Dict]:
        """Scrape from Bovada-style public odds pages"""
        logger.info("🎰 Scraping Bovada-style odds...")

        try:
            # Bovada has a public odds page that doesn't require login
            url = "https://www.bovada.lv/sports/football/nfl"
            response = self.session.get(url, timeout=10)

            soup = BeautifulSoup(response.content, 'html.parser')
            games = []

            # Look for odds elements
            odds_elements = soup.find_all(['span', 'div'], class_=re.compile('odd|line|price', re.I))

            current_game = {}
            for elem in odds_elements:
                text = elem.get_text(strip=True)
                # Look for patterns like +150, -110, etc.
                if re.search(r'[+-]\d{2,}', text):
                    if 'odds' not in current_game:
                        current_game = {
                            'source': 'Bovada',
                            'scraped_at': datetime.now().isoformat(),
                            'odds': []
                        }
                    current_game['odds'].append(text)

                    if len(current_game['odds']) >= 6:  # ML, Spread, Total for both teams
                        games.append(current_game)
                        current_game = {}

            logger.info(f"✅ Bovada-style: Found {len(games)} entries")
            return games

        except Exception as e:
            logger.error(f"❌ Bovada scraping failed: {e}")
            return []

    def parse_odds_string(self, text: str) -> Dict[str, any]:
        """Parse odds from text string"""
        result = {
            'moneyline': None,
            'spread': None,
            'spread_odds': None,
            'total': None,
            'total_odds': None
        }

        # Extract moneyline (e.g., +150, -200)
        ml_match = re.search(r'ML?\s*([+-]\d{2,})', text, re.I)
        if ml_match:
            result['moneyline'] = int(ml_match.group(1))

        # Extract spread (e.g., -7.5 -110)
        spread_match = re.search(r'([+-]?\d+\.?\d*)\s*([+-]\d{2,})', text)
        if spread_match:
            result['spread'] = float(spread_match.group(1))
            result['spread_odds'] = int(spread_match.group(2))

        # Extract total (e.g., O 47.5 -110, U 47.5 -110)
        total_match = re.search(r'[OU]\s*(\d+\.?\d*)\s*([+-]\d{2,})', text, re.I)
        if total_match:
            result['total'] = float(total_match.group(1))
            result['total_odds'] = int(total_match.group(2))

        return result

    def scrape_all_sources(self) -> Dict[str, List[Dict]]:
        """Scrape all available sources and aggregate results"""
        logger.info("=" * 70)
        logger.info("🏈 STARTING NFL ODDS SCRAPING")
        logger.info("=" * 70)

        all_odds = {
            'timestamp': datetime.now().isoformat(),
            'espn': self.scrape_espn(),
            'oddsshark': self.scrape_odds_shark(),
            'covers': self.scrape_covers(),
            'action_network': self.scrape_action_network(),
            'bovada': self.scrape_bovada_style()
        }

        # Count total results
        total_results = sum(len(v) for v in all_odds.values() if isinstance(v, list))

        logger.info("=" * 70)
        logger.info(f"✅ SCRAPING COMPLETE: {total_results} total entries")
        logger.info("=" * 70)

        return all_odds

    def save_results(self, odds_data: Dict, filename: str = 'scraped_nfl_odds.json'):
        """Save scraped odds to JSON file"""
        filepath = f"data/{filename}"

        with open(filepath, 'w') as f:
            json.dump(odds_data, f, indent=2)

        logger.info(f"💾 Results saved to: {filepath}")
        return filepath


def main():
    """Main execution function"""
    print("""
    ═══════════════════════════════════════════════════════════════
    🏈 NFL ODDS WEB SCRAPER
    ═══════════════════════════════════════════════════════════════
    This scraper fetches NFL odds from multiple public sources
    when API access is blocked.
    ═══════════════════════════════════════════════════════════════
    """)

    scraper = NFLOddsScraper()

    # Scrape all sources
    odds_data = scraper.scrape_all_sources()

    # Save results
    filepath = scraper.save_results(odds_data)

    # Display summary
    print("\n" + "=" * 70)
    print("📊 SCRAPING SUMMARY")
    print("=" * 70)

    for source, data in odds_data.items():
        if isinstance(data, list):
            print(f"  {source.upper()}: {len(data)} entries")

    print("\n" + "=" * 70)
    print(f"✅ Data saved to: {filepath}")
    print("=" * 70)

    # Try to extract actual game info
    print("\n🎯 EXTRACTING GAME INFORMATION...\n")

    games_found = []
    for source, entries in odds_data.items():
        if isinstance(entries, list):
            for entry in entries:
                # Try to find team names and odds
                raw_text = str(entry)

                # Common NFL team patterns
                teams_pattern = r'(Chiefs|Bills|Bengals|Ravens|Browns|Steelers|Texans|Colts|Jaguars|Titans|Broncos|Raiders|Chargers|Patriots|Dolphins|Jets|Cowboys|Eagles|Giants|Commanders|Bears|Packers|Lions|Vikings|Falcons|Panthers|Saints|Buccaneers|Cardinals|Rams|49ers|Seahawks)'

                teams = re.findall(teams_pattern, raw_text, re.I)
                odds = re.findall(r'([+-]\d{2,})', raw_text)

                if len(teams) >= 2:
                    game_info = f"{teams[0]} vs {teams[1]}"
                    if odds:
                        game_info += f" | Odds: {', '.join(odds[:4])}"

                    if game_info not in games_found:
                        games_found.append(game_info)
                        print(f"  📍 {game_info}")

    if not games_found:
        print("  ⚠️  No clear games extracted - check raw data in JSON file")

    print("\n" + "=" * 70)
    print("🏁 SCRAPING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review scraped_nfl_odds.json for raw data")
    print("  2. Run the analysis script with this data")
    print("  3. Get AI predictions for tonight's game")
    print("=" * 70)


if __name__ == "__main__":
    main()
