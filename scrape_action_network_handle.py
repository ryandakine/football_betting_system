#!/usr/bin/env python3
"""
Action Network Handle Data Scraper
===================================

Scrapes public betting data from Action Network for trap detection

WHAT THIS SCRAPES:
- Public betting percentage (% of bets)
- Money percentage (% of handle) ← THIS IS KEY!
- Line movement (opening vs current)
- Sharp money signals

WHY:
- Trap detection requires knowing expected vs actual handle
- Action Network has best free public data
- Enables sharp money vs public money analysis

USAGE:
    python scrape_action_network_handle.py

OUTPUT:
    data/handle_data/ncaa_handle_YYYY-MM-DD.json

INTEGRATION:
    from scrape_action_network_handle import get_handle_for_game
    handle_data = get_handle_for_game("Toledo", "Bowling Green")
"""

import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def scrape_action_network_ncaa() -> List[Dict]:
    """
    Scrape NCAA handle data from Action Network

    Returns: List of games with public/money %
    """

    url = "https://www.actionnetwork.com/ncaaf/odds"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        games = []

        # Find all game containers
        # Note: Action Network structure changes frequently
        # This is a template - may need adjustment based on current HTML structure

        game_elements = soup.find_all('div', class_='game-card')

        for game_elem in game_elements:
            try:
                # Extract teams
                away_team = game_elem.find('span', class_='away-team')
                home_team = game_elem.find('span', class_='home_team')

                if not away_team or not home_team:
                    continue

                away = away_team.text.strip()
                home = home_team.text.strip()

                # Extract betting percentages
                public_pct_elem = game_elem.find('span', class_='public-pct')
                money_pct_elem = game_elem.find('span', class_='money-pct')

                public_pct = float(public_pct_elem.text.strip().rstrip('%')) / 100 if public_pct_elem else None
                money_pct = float(money_pct_elem.text.strip().rstrip('%')) / 100 if money_pct_elem else None

                # Extract moneyline
                ml_elem = game_elem.find('span', class_='moneyline')
                moneyline = float(ml_elem.text.strip()) if ml_elem else None

                # Extract spread
                spread_elem = game_elem.find('span', class_='spread')
                spread = float(spread_elem.text.strip()) if spread_elem else None

                # Extract line movement
                opening_line = game_elem.find('span', class_='opening-line')
                current_line = game_elem.find('span', class_='current-line')

                opening = float(opening_line.text.strip()) if opening_line else None
                current = float(current_line.text.strip()) if current_line else None

                game_data = {
                    'away_team': away,
                    'home_team': home,
                    'public_percentage': public_pct,
                    'money_percentage': money_pct,  # THIS IS HANDLE!
                    'moneyline': moneyline,
                    'spread': spread,
                    'opening_line': opening,
                    'current_line': current,
                    'line_movement': current - opening if opening and current else 0,
                    'scraped_at': datetime.now().isoformat()
                }

                games.append(game_data)

            except Exception as e:
                print(f"⚠️  Error parsing game: {e}")
                continue

        return games

    except Exception as e:
        print(f"❌ Error scraping Action Network: {e}")
        return []


def scrape_action_network_api() -> List[Dict]:
    """
    Alternative: Use Action Network API (requires API key)

    BETTER: More reliable than web scraping
    """

    # Action Network offers API access
    # https://www.actionnetwork.com/api/
    # Requires API key (paid)

    api_key = "YOUR_ACTION_NETWORK_API_KEY"

    if api_key == "YOUR_ACTION_NETWORK_API_KEY":
        print("⚠️  Action Network API key not configured")
        print("   Get API key at: https://www.actionnetwork.com/api/")
        return []

    url = "https://api.actionnetwork.com/web/v1/leagues/ncaaf/games"

    headers = {
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'NCAA Betting System/1.0'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()

        games = []

        for game in data.get('games', []):
            # Extract betting data from API response
            betting_data = game.get('betting', {})

            game_data = {
                'away_team': game.get('away_team', {}).get('name'),
                'home_team': game.get('home_team', {}).get('name'),
                'public_percentage': betting_data.get('public_bet_percentage'),
                'money_percentage': betting_data.get('money_percentage'),
                'moneyline': betting_data.get('moneyline'),
                'spread': betting_data.get('spread'),
                'opening_line': betting_data.get('opening_line'),
                'current_line': betting_data.get('current_line'),
                'line_movement': betting_data.get('line_movement'),
                'scraped_at': datetime.now().isoformat()
            }

            games.append(game_data)

        return games

    except Exception as e:
        print(f"❌ Error with Action Network API: {e}")
        return []


def save_handle_data(games: List[Dict]):
    """Save handle data to file"""

    output_dir = Path("data/handle_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"ncaa_handle_{today}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'scraped_at': datetime.now().isoformat(),
            'count': len(games),
            'games': games
        }, f, indent=2)

    print(f"✅ Saved {len(games)} games to {output_file}")


def get_handle_for_game(away_team: str, home_team: str) -> Optional[Dict]:
    """
    Get handle data for specific game

    Returns: Handle data or None if not found
    """

    # Load latest handle data
    handle_dir = Path("data/handle_data")

    if not handle_dir.exists():
        print("⚠️  No handle data available - run scraper first")
        return None

    # Find most recent file
    handle_files = sorted(handle_dir.glob("ncaa_handle_*.json"), reverse=True)

    if not handle_files:
        print("⚠️  No handle data files found")
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

    print("Action Network Handle Data Scraper")
    print("=" * 80)
    print()

    print("📊 Scraping NCAA handle data from Action Network...")
    print()

    # Try API first (more reliable)
    games = scrape_action_network_api()

    if not games:
        print("⚠️  API scraping failed, trying web scraping...")
        games = scrape_action_network_ncaa()

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
            print(f"   Line movement: {sample.get('line_movement', 0):+.1f}")
            print()

        save_handle_data(games)

    else:
        print("❌ No games scraped")
        print()
        print("TROUBLESHOOTING:")
        print("1. Action Network may have changed HTML structure")
        print("2. Consider getting API key: https://www.actionnetwork.com/api/")
        print("3. Alternative sources: BetOnline, VegasInsider")
        print()
        print("MANUAL ALTERNATIVE:")
        print("Visit https://www.actionnetwork.com/ncaaf/odds")
        print("Note down money % for each game manually")


if __name__ == "__main__":
    main()
