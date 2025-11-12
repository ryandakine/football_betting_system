#!/usr/bin/env python3
"""
Action Network Selenium Scraper - Production Ready

WHY SELENIUM:
- Action Network uses JavaScript to load content
- BeautifulSoup can't see dynamically loaded data
- Selenium renders the page like a real browser

REQUIREMENTS:
    pip install selenium webdriver-manager

USAGE:
    python action_network_selenium_scraper.py --week 11
    python action_network_selenium_scraper.py --save
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("❌ Missing dependencies. Install with:")
    print("   pip install selenium webdriver-manager")
    sys.exit(1)


class ActionNetworkSeleniumScraper:
    """Scrapes Action Network using Selenium for JavaScript rendering"""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def setup_driver(self):
        """Setup Chrome driver with options"""
        print("🔧 Setting up Chrome driver...")

        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument('--headless')  # Run without GUI
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

        # Use webdriver-manager to auto-download correct ChromeDriver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

        print("✅ Chrome driver ready")

    def fetch_nfl_games(self, week: Optional[int] = None) -> List[Dict]:
        """
        Fetch NFL games from Action Network with betting percentages.

        Args:
            week: NFL week number (None = current week)

        Returns:
            List of games with handle data
        """
        if not self.driver:
            self.setup_driver()

        print(f"🌐 Loading Action Network NFL page...")

        try:
            # Navigate to Action Network NFL page
            url = "https://www.actionnetwork.com/nfl/odds"
            self.driver.get(url)

            # Wait for page to load
            print("⏳ Waiting for page to render...")
            wait = WebDriverWait(self.driver, 15)

            # Wait for game elements to appear
            # NOTE: These selectors may need updating based on current site structure
            try:
                wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='game-card']"))
                )
                print("✅ Page loaded successfully")
            except TimeoutException:
                # Try alternative selector
                try:
                    wait.until(
                        EC.presence_of_element_located((By.CLASS_NAME, "game-box"))
                    )
                    print("✅ Page loaded (alternative selector)")
                except TimeoutException:
                    print("❌ Timeout waiting for games to load")
                    print("   TIP: Check if Action Network changed their HTML structure")
                    return []

            # Small delay to ensure all content loaded
            time.sleep(2)

            # Extract game data
            games = self._extract_games_from_page()

            return games

        except Exception as e:
            print(f"❌ Error fetching games: {e}")
            return []

        finally:
            # Don't close driver yet - may want to reuse

            pass

    def _extract_games_from_page(self) -> List[Dict]:
        """
        Extract game data from loaded page.

        NOTE: This is the part that needs updating based on current site structure.
        Use Chrome DevTools to find correct selectors.
        """
        games = []

        print("📊 Extracting game data...")

        # Find all game containers
        # TRY MULTIPLE SELECTORS (site may use different ones)
        game_selectors = [
            "[data-testid='game-card']",
            ".game-box",
            ".game-card",
            "[class*='GameCard']",
            "[class*='game']"
        ]

        game_elements = []
        for selector in game_selectors:
            try:
                game_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if game_elements:
                    print(f"✅ Found {len(game_elements)} games using selector: {selector}")
                    break
            except:
                continue

        if not game_elements:
            print("❌ Could not find game elements")
            print("   ACTION REQUIRED: Inspect Action Network page and update selectors")
            print("   1. Open https://www.actionnetwork.com/nfl/odds in browser")
            print("   2. Right-click on a game → Inspect")
            print("   3. Find the container class/attribute")
            print("   4. Update selectors in this code")
            return []

        # Extract data from each game
        for idx, game_elem in enumerate(game_elements):
            try:
                game_data = self._parse_game_element(game_elem, idx)
                if game_data:
                    games.append(game_data)
            except Exception as e:
                print(f"   ⚠️  Error parsing game {idx + 1}: {e}")
                continue

        print(f"✅ Extracted {len(games)} games")
        return games

    def _parse_game_element(self, element, idx: int) -> Optional[Dict]:
        """
        Parse a single game element to extract betting data.

        This is the key function that needs customization based on current HTML.
        """
        try:
            # EXAMPLE PARSING - UPDATE THESE SELECTORS
            # Use Chrome DevTools to find actual selectors

            # Extract team names
            teams = element.find_elements(By.CSS_SELECTOR, "[data-testid='team-name']")
            if len(teams) < 2:
                # Try alternative
                teams = element.find_elements(By.CLASS_NAME, "team-name")

            away_team = teams[0].text if len(teams) > 0 else "AWAY"
            home_team = teams[1].text if len(teams) > 1 else "HOME"

            # Extract betting percentages
            # Look for elements containing "Public" and "Money" percentages
            percentages = element.find_elements(By.CSS_SELECTOR, "[class*='percent']")

            # Default values
            public_pct = 50
            money_pct = 50

            # Try to extract actual values
            for pct_elem in percentages:
                text = pct_elem.text
                if 'public' in text.lower() or 'tickets' in text.lower():
                    # Extract number
                    public_pct = self._extract_percentage(text)
                elif 'money' in text.lower() or 'handle' in text.lower():
                    money_pct = self._extract_percentage(text)

            # Extract moneyline odds
            odds_elements = element.find_elements(By.CSS_SELECTOR, "[class*='odds']")
            home_ml = -150  # Default
            away_ml = +130  # Default

            # Build game data structure
            game_data = {
                'game': f"{away_team} @ {home_team}",
                'home_team': home_team,
                'away_team': away_team,
                'home_ml': home_ml,
                'away_ml': away_ml,
                'home_handle': money_pct / 100 if money_pct else 0.5,
                'away_handle': (100 - money_pct) / 100 if money_pct else 0.5,
                'public_percent': public_pct,
                'money_percent': money_pct,
                'divergence': abs(money_pct - public_pct),
                'timestamp': datetime.now().isoformat()
            }

            print(f"   ✅ {game_data['game']}: Public {public_pct}%, Money {money_pct}%")
            return game_data

        except Exception as e:
            print(f"   ⚠️  Could not parse game {idx + 1}: {e}")
            return None

    def _extract_percentage(self, text: str) -> int:
        """Extract percentage number from text like '72%' or 'Public: 72%'"""
        import re
        match = re.search(r'(\d+)%', text)
        if match:
            return int(match.group(1))
        return 50  # Default

    def save_games(self, games: List[Dict], filename: Optional[str] = None):
        """Save scraped games to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"action_network_scraped_{timestamp}.json"

        output_file = self.data_dir / filename

        output = {
            'scraped_at': datetime.now().isoformat(),
            'total_games': len(games),
            'games': games
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"💾 Saved to {output_file}")
        return output_file

    def close(self):
        """Close the browser"""
        if self.driver:
            self.driver.quit()
            print("🔒 Browser closed")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Action Network using Selenium"
    )
    parser.add_argument(
        "--week",
        type=int,
        help="NFL week number"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show browser window (not headless)"
    )

    args = parser.parse_args()

    scraper = ActionNetworkSeleniumScraper(headless=not args.show_browser)

    try:
        # Fetch games
        games = scraper.fetch_nfl_games(week=args.week)

        if games:
            # Display results
            print("\n" + "=" * 70)
            print("📊 SCRAPED HANDLE DATA")
            print("=" * 70)
            print()

            for game in games:
                print(f"🏈 {game['game']}")
                print(f"   Public: {game['public_percent']}%")
                print(f"   Money:  {game['money_percent']}%")
                print(f"   Gap:    {game['divergence']}%")
                print()

            # Save if requested
            if args.save:
                scraper.save_games(games)

        else:
            print("\n⚠️  No games found!")
            print()
            print("🔧 TROUBLESHOOTING:")
            print("   1. Run with --show-browser to see what's happening")
            print("   2. Check if Action Network changed their HTML")
            print("   3. Update selectors in _parse_game_element()")

    finally:
        scraper.close()

    print("\n💡 NEXT STEPS:")
    print("   1. Verify data looks correct")
    print("   2. If no data, inspect page and update selectors")
    print("   3. Use this data with trap_detector.py")


if __name__ == "__main__":
    main()
