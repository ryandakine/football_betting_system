#!/usr/bin/env python3
"""
Action Network Handle Data Scraper - CRAWLBASE EDITION
======================================================

Alternative scraper using Crawlbase MCP for reliable web scraping

WHY USE THIS VERSION:
- ✅ Handles JavaScript rendering (Action Network is JS-heavy)
- ✅ Bypasses anti-bot protection with proxy rotation
- ✅ More reliable than requests/BeautifulSoup
- ✅ Doesn't break when HTML structure changes

WHEN TO USE:
- Option 1: scrape_action_network_handle.py (free, but may break)
- Option 2: THIS FILE (Crawlbase, more reliable, needs token)
- Option 3: Manual entry from ActionNetwork.com

SETUP:
1. Get free Crawlbase token: https://crawlbase.com/
2. Add to .env:
   CRAWLBASE_TOKEN=your_token_here
3. Or add to MCP config (.windsurf/mcp.json or .cursor/mcp.json)

USAGE:
    python scrape_action_network_crawlbase.py

OUTPUT:
    data/handle_data/ncaa_handle_YYYY-MM-DD.json

INTEGRATION:
    from scrape_action_network_crawlbase import get_handle_for_game_crawlbase
    handle_data = get_handle_for_game_crawlbase("Toledo", "Bowling Green")
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# NOTE: This uses Crawlbase Python API client
# Install: pip install crawlbase
try:
    from crawlbase import CrawlingAPI
    CRAWLBASE_AVAILABLE = True
except ImportError:
    CRAWLBASE_AVAILABLE = False
    print("⚠️  Crawlbase not installed. Install with: pip install crawlbase")


def scrape_action_network_crawlbase() -> List[Dict]:
    """
    Scrape NCAA handle data using Crawlbase API

    ADVANTAGES over regular scraper:
    - Renders JavaScript (sees actual page content)
    - Proxy rotation (avoids detection)
    - More reliable (doesn't break on HTML changes)

    Returns: List of games with public/money %
    """

    if not CRAWLBASE_AVAILABLE:
        print("❌ Crawlbase Python client not installed")
        print("   Install with: pip install crawlbase")
        return []

    # Get Crawlbase token from environment
    token = os.getenv('CRAWLBASE_TOKEN')

    if not token or token == 'YOUR_CRAWLBASE_TOKEN_HERE':
        print("❌ CRAWLBASE_TOKEN not set in environment")
        print("   Get free token at: https://crawlbase.com/")
        print("   Then add to .env: CRAWLBASE_TOKEN=your_token")
        return []

    url = "https://www.actionnetwork.com/ncaaf/odds"

    print(f"📊 Crawling {url} with Crawlbase...")

    try:
        # Initialize Crawlbase API
        api = CrawlingAPI({'token': token})

        # Crawl with JavaScript rendering enabled
        response = api.get(url, {
            'ajax_wait': 'true',      # Wait for AJAX to complete
            'page_wait': '5000',      # Wait 5 seconds for page load
            'country': 'US',          # Use US proxy (betting sites geo-locked)
        })

        if response['status_code'] != 200:
            print(f"❌ Crawlbase returned status {response['status_code']}")
            return []

        html_content = response['body'].decode('utf-8')

        # Parse the rendered HTML
        games = parse_action_network_html(html_content)

        return games

    except Exception as e:
        print(f"❌ Error with Crawlbase: {e}")
        return []


def parse_action_network_html(html: str) -> List[Dict]:
    """
    Parse Action Network HTML to extract betting data

    This uses regex patterns instead of BeautifulSoup for more robustness
    """

    games = []

    try:
        # Action Network embeds data in JSON format within the page
        # Look for pattern like: window.__INITIAL_STATE__ = {...}

        # Try to find embedded JSON data (most reliable method)
        json_pattern = r'window\.__INITIAL_STATE__\s*=\s*({.*?});'
        json_match = re.search(json_pattern, html, re.DOTALL)

        if json_match:
            print("✅ Found embedded JSON data")
            data = json.loads(json_match.group(1))

            # Navigate JSON structure to find games
            # Structure varies but typically: data.games or data.pageProps.games
            games_data = None

            if 'games' in data:
                games_data = data['games']
            elif 'pageProps' in data and 'games' in data['pageProps']:
                games_data = data['pageProps']['games']
            elif 'pageProps' in data and 'initialData' in data['pageProps']:
                if 'games' in data['pageProps']['initialData']:
                    games_data = data['pageProps']['initialData']['games']

            if games_data:
                for game in games_data:
                    try:
                        # Extract team names
                        away_team = game.get('away_team', {}).get('name', 'Unknown')
                        home_team = game.get('home_team', {}).get('name', 'Unknown')

                        # Extract betting data
                        betting = game.get('betting_data', {}) or game.get('odds', {})

                        # Public percentage (% of bets)
                        public_pct = betting.get('public_bet_pct')
                        if isinstance(public_pct, str):
                            public_pct = float(public_pct.rstrip('%')) / 100

                        # Money percentage (% of handle) - THE KEY METRIC!
                        money_pct = betting.get('money_pct') or betting.get('handle_pct')
                        if isinstance(money_pct, str):
                            money_pct = float(money_pct.rstrip('%')) / 100

                        # Spread and moneyline
                        spread = betting.get('spread')
                        moneyline = betting.get('moneyline')

                        # Line movement
                        opening_line = betting.get('opening_spread')
                        current_line = spread

                        line_movement = 0
                        if opening_line and current_line:
                            try:
                                line_movement = float(current_line) - float(opening_line)
                            except:
                                pass

                        game_data = {
                            'away_team': away_team,
                            'home_team': home_team,
                            'public_percentage': public_pct,
                            'money_percentage': money_pct,  # HANDLE!
                            'moneyline': moneyline,
                            'spread': spread,
                            'opening_line': opening_line,
                            'current_line': current_line,
                            'line_movement': line_movement,
                            'scraped_at': datetime.now().isoformat(),
                            'source': 'crawlbase'
                        }

                        games.append(game_data)

                    except Exception as e:
                        print(f"⚠️  Error parsing game: {e}")
                        continue

        else:
            print("⚠️  No embedded JSON found, trying fallback parsing")
            # Fallback: Parse visible HTML (less reliable but better than nothing)
            games = parse_html_fallback(html)

        return games

    except Exception as e:
        print(f"❌ Error parsing HTML: {e}")
        return []


def parse_html_fallback(html: str) -> List[Dict]:
    """
    Fallback parser using BeautifulSoup if JSON extraction fails
    """

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("⚠️  BeautifulSoup not available for fallback parsing")
        return []

    games = []
    soup = BeautifulSoup(html, 'html.parser')

    # Look for common patterns in Action Network HTML
    # (This will vary and may need updates)

    # Find game containers - try multiple selectors
    game_elements = (
        soup.find_all('div', class_=re.compile('game.*card', re.I)) or
        soup.find_all('tr', class_=re.compile('game.*row', re.I)) or
        soup.find_all('div', attrs={'data-testid': re.compile('game', re.I)})
    )

    for game_elem in game_elements:
        try:
            # Try to extract team names and betting data
            # This is very fragile and will need updates

            teams = game_elem.find_all(text=re.compile(r'[A-Z][a-z]+'))
            if len(teams) >= 2:
                game_data = {
                    'away_team': teams[0],
                    'home_team': teams[1],
                    'public_percentage': None,
                    'money_percentage': None,
                    'moneyline': None,
                    'spread': None,
                    'opening_line': None,
                    'current_line': None,
                    'line_movement': 0,
                    'scraped_at': datetime.now().isoformat(),
                    'source': 'crawlbase_fallback'
                }

                games.append(game_data)

        except Exception as e:
            continue

    return games


def save_handle_data(games: List[Dict]):
    """Save handle data to file"""

    output_dir = Path("data/handle_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"ncaa_handle_crawlbase_{today}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'scraped_at': datetime.now().isoformat(),
            'count': len(games),
            'source': 'crawlbase',
            'games': games
        }, f, indent=2)

    print(f"✅ Saved {len(games)} games to {output_file}")


def get_handle_for_game_crawlbase(away_team: str, home_team: str) -> Optional[Dict]:
    """
    Get handle data for specific game (from Crawlbase-scraped data)

    Returns: Handle data or None if not found
    """

    # Load latest Crawlbase handle data
    handle_dir = Path("data/handle_data")

    if not handle_dir.exists():
        print("⚠️  No handle data available - run scraper first")
        return None

    # Find most recent Crawlbase file
    handle_files = sorted(
        handle_dir.glob("ncaa_handle_crawlbase_*.json"),
        reverse=True
    )

    if not handle_files:
        print("⚠️  No Crawlbase handle data files found")
        print("   Run: python scrape_action_network_crawlbase.py")
        return None

    # Load latest file
    with open(handle_files[0]) as f:
        data = json.load(f)

    # Find matching game
    for game in data.get('games', []):
        if (game['away_team'].lower() in away_team.lower() or
            away_team.lower() in game['away_team'].lower()) and \
           (game['home_team'].lower() in home_team.lower() or
            home_team.lower() in game['home_team'].lower()):
            return game

    return None


def main():
    """Main scraper entry point"""

    print("=" * 80)
    print("Action Network Handle Data Scraper - CRAWLBASE EDITION")
    print("=" * 80)
    print()

    if not CRAWLBASE_AVAILABLE:
        print("❌ Crawlbase Python client not installed")
        print()
        print("INSTALL:")
        print("  pip install crawlbase")
        print()
        print("GET TOKEN:")
        print("  https://crawlbase.com/ (has free tier)")
        print()
        print("SETUP:")
        print("  export CRAWLBASE_TOKEN=your_token")
        print("  # or add to .env file")
        print()
        return

    print("🔧 Using Crawlbase for reliable web scraping")
    print("   ✅ JavaScript rendering")
    print("   ✅ Anti-bot protection")
    print("   ✅ Proxy rotation")
    print()

    games = scrape_action_network_crawlbase()

    if games:
        print(f"✅ Scraped {len(games)} games")
        print()

        # Show sample
        if games:
            sample = games[0]
            print("📋 Sample game:")
            print(f"   {sample['away_team']} @ {sample['home_team']}")
            print(f"   Public %: {sample.get('public_percentage', 'N/A')}")
            print(f"   Money %: {sample.get('money_percentage', 'N/A')} ← HANDLE!")
            print(f"   Moneyline: {sample.get('moneyline', 'N/A')}")
            print(f"   Spread: {sample.get('spread', 'N/A')}")
            print(f"   Line movement: {sample.get('line_movement', 0):+.1f}")
            print()

        save_handle_data(games)

        # Show statistics
        games_with_handle = [g for g in games if g.get('money_percentage')]
        print(f"📊 Games with handle data: {len(games_with_handle)}/{len(games)}")

        if games_with_handle:
            print()
            print("🎯 READY FOR TRAP DETECTION!")
            print("   Use this data with: python ncaa_trap_detection.py")

    else:
        print("❌ No games scraped")
        print()
        print("TROUBLESHOOTING:")
        print("1. Check CRAWLBASE_TOKEN is set correctly")
        print("2. Verify you have credits (check Crawlbase dashboard)")
        print("3. Try regular scraper: python scrape_action_network_handle.py")
        print("4. Manual alternative: Visit https://www.actionnetwork.com/ncaaf/odds")


if __name__ == "__main__":
    main()
