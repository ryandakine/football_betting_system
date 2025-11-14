#!/usr/bin/env python3
"""
Handle Data Fetcher - Unified Interface for Multiple Scraping Methods
=====================================================================

Provides a single interface to fetch handle data using multiple methods:

METHOD 1: Crawlbase (Most Reliable)
- ✅ JavaScript rendering
- ✅ Anti-bot protection
- ✅ Proxy rotation
- ⚠️ Requires Crawlbase token (free tier available)

METHOD 2: Direct Scraping (Free)
- ✅ No external dependencies
- ✅ Free
- ⚠️ May break when HTML changes
- ⚠️ No JavaScript support

METHOD 3: Manual Entry
- ✅ Always works
- ⚠️ Manual effort required

USAGE:
    from handle_data_fetcher import get_handle_data

    # Automatically tries all methods in order
    games = get_handle_data()

    # Force specific method
    games = get_handle_data(method='crawlbase')
    games = get_handle_data(method='direct')

    # Get data for specific game
    handle = get_handle_for_game("Toledo", "Bowling Green")
"""

import os
from typing import Dict, List, Optional
from pathlib import Path


def get_handle_data(method: str = 'auto') -> List[Dict]:
    """
    Get handle data using specified method

    Args:
        method: 'auto', 'crawlbase', 'direct', or 'manual'
               'auto' tries methods in order until one works

    Returns:
        List of games with handle data
    """

    if method == 'auto':
        # Try methods in order of reliability
        print("🔄 Auto-detecting best scraping method...")

        # Try Crawlbase first (most reliable)
        if _check_crawlbase_available():
            print("✅ Crawlbase available - using for scraping")
            return get_handle_data_crawlbase()

        # Fallback to direct scraping
        print("⚠️  Crawlbase not available - using direct scraping")
        return get_handle_data_direct()

    elif method == 'crawlbase':
        return get_handle_data_crawlbase()

    elif method == 'direct':
        return get_handle_data_direct()

    elif method == 'manual':
        print("📝 Manual entry mode")
        print("   Visit: https://www.actionnetwork.com/ncaaf/odds")
        print("   Note money % for each game")
        print("   Enter data in: data/handle_data/manual_entry.json")
        return []

    else:
        raise ValueError(f"Unknown method: {method}")


def get_handle_data_crawlbase() -> List[Dict]:
    """
    Get handle data using Crawlbase scraper

    Returns: List of games or empty list if failed
    """

    try:
        from scrape_action_network_crawlbase import scrape_action_network_crawlbase

        print("📊 Scraping with Crawlbase...")
        games = scrape_action_network_crawlbase()

        if games:
            print(f"✅ Crawlbase: Got {len(games)} games")
            return games
        else:
            print("❌ Crawlbase: No games returned")
            return []

    except ImportError as e:
        print(f"❌ Crawlbase scraper not available: {e}")
        return []
    except Exception as e:
        print(f"❌ Crawlbase error: {e}")
        return []


def get_handle_data_direct() -> List[Dict]:
    """
    Get handle data using direct scraper (requests/BeautifulSoup)

    Returns: List of games or empty list if failed
    """

    try:
        from scrape_action_network_handle import scrape_action_network_ncaa

        print("📊 Scraping with direct method...")
        games = scrape_action_network_ncaa()

        if games:
            print(f"✅ Direct scraper: Got {len(games)} games")
            return games
        else:
            print("❌ Direct scraper: No games returned")
            return []

    except ImportError as e:
        print(f"❌ Direct scraper not available: {e}")
        return []
    except Exception as e:
        print(f"❌ Direct scraper error: {e}")
        return []


def get_handle_for_game(away_team: str, home_team: str,
                        method: str = 'auto') -> Optional[Dict]:
    """
    Get handle data for specific game

    Tries to load from already-scraped data first,
    then scrapes if necessary

    Args:
        away_team: Away team name
        home_team: Home team name
        method: Scraping method to use

    Returns:
        Handle data dict or None
    """

    # First try to load from existing data (either source)
    handle = _load_from_cache(away_team, home_team)

    if handle:
        return handle

    # If not cached, scrape fresh data
    print(f"🔍 No cached data for {away_team} @ {home_team}")
    print("   Scraping fresh data...")

    games = get_handle_data(method=method)

    if not games:
        return None

    # Find the game in scraped data
    for game in games:
        if (_team_match(game.get('away_team', ''), away_team) and
            _team_match(game.get('home_team', ''), home_team)):
            return game

    return None


def _check_crawlbase_available() -> bool:
    """Check if Crawlbase is configured and available"""

    try:
        from crawlbase import CrawlingAPI

        token = os.getenv('CRAWLBASE_TOKEN')

        if token and token != 'YOUR_CRAWLBASE_TOKEN_HERE':
            return True

        return False

    except ImportError:
        return False


def _load_from_cache(away_team: str, home_team: str) -> Optional[Dict]:
    """
    Load handle data from cache files

    Checks both Crawlbase and direct scraper cache files
    """

    handle_dir = Path("data/handle_data")

    if not handle_dir.exists():
        return None

    # Check all cache files (both sources)
    cache_files = sorted(
        handle_dir.glob("ncaa_handle_*.json"),
        reverse=True
    )

    for cache_file in cache_files:
        try:
            import json

            with open(cache_file) as f:
                data = json.load(f)

            for game in data.get('games', []):
                if (_team_match(game.get('away_team', ''), away_team) and
                    _team_match(game.get('home_team', ''), home_team)):
                    return game

        except Exception as e:
            continue

    return None


def _team_match(team1: str, team2: str) -> bool:
    """Check if two team names match (fuzzy matching)"""

    t1 = team1.lower().strip()
    t2 = team2.lower().strip()

    # Exact match
    if t1 == t2:
        return True

    # Partial match (handles "Toledo" vs "Toledo Rockets")
    if t1 in t2 or t2 in t1:
        return True

    return False


def get_scraping_status() -> Dict:
    """
    Get status of all scraping methods

    Returns dict with availability of each method
    """

    status = {
        'crawlbase': {
            'available': _check_crawlbase_available(),
            'priority': 1,
            'reliability': 'HIGH',
            'cost': 'Free tier available',
        },
        'direct': {
            'available': True,  # Always available
            'priority': 2,
            'reliability': 'MEDIUM',
            'cost': 'Free',
        },
        'manual': {
            'available': True,  # Always available
            'priority': 3,
            'reliability': 'HIGH (if done correctly)',
            'cost': 'Time/effort',
        }
    }

    return status


def print_scraping_guide():
    """Print guide for setting up scraping methods"""

    print("=" * 80)
    print("HANDLE DATA SCRAPING - SETUP GUIDE")
    print("=" * 80)
    print()

    status = get_scraping_status()

    print("📊 AVAILABLE METHODS:")
    print()

    for method, info in status.items():
        available_icon = "✅" if info['available'] else "❌"
        print(f"{available_icon} {method.upper()}")
        print(f"   Priority: #{info['priority']}")
        print(f"   Reliability: {info['reliability']}")
        print(f"   Cost: {info['cost']}")
        print()

    if not status['crawlbase']['available']:
        print("🔧 TO ENABLE CRAWLBASE (RECOMMENDED):")
        print("   1. Get free token: https://crawlbase.com/")
        print("   2. Install: pip install crawlbase")
        print("   3. Set env: export CRAWLBASE_TOKEN=your_token")
        print("      Or add to .env file")
        print()

    print("📖 USAGE:")
    print("   from handle_data_fetcher import get_handle_data")
    print("   games = get_handle_data()  # Auto-selects best method")
    print()

    print("🎯 FOR TRAP DETECTION:")
    print("   from handle_data_fetcher import get_handle_for_game")
    print("   handle = get_handle_for_game('Toledo', 'Bowling Green')")
    print()


if __name__ == "__main__":
    print_scraping_guide()
