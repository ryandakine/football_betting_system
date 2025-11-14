#!/usr/bin/env python3
"""
Auto Handle Fetcher - Scrapes Action Network for money % (handle)
Uses Crawlbase MCP for reliable scraping
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime

def fetch_action_network_handle(away_team, home_team):
    """
    Scrape Action Network for handle data using Crawlbase MCP.
    
    Args:
        away_team: "Eagles", "Packers", etc.
        home_team: "Packers", "Eagles", etc.
    
    Returns:
        dict: {
            'home_handle': 0.72,
            'away_handle': 0.28,
            'home_bets': 0.65,
            'away_bets': 0.35,
            'source': 'action_network',
            'timestamp': '2025-01-14T14:05:00Z'
        }
    """
    
    # Construct URL - Action Network game page
    game_slug = f"{away_team.lower()}-{home_team.lower()}"
    url = f"https://www.actionnetwork.com/nfl/{game_slug}"
    
    print(f"🔍 Fetching handle data for {away_team} @ {home_team}...")
    print(f"📍 URL: {url}")
    
    try:
        # Call Crawlbase MCP via subprocess
        # This assumes you're running inside Claude Desktop with MCP configured
        result = subprocess.run(
            ['npx', '@crawlbase/mcp@latest', 'crawl_markdown', url],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"❌ Crawlbase error: {result.stderr}")
            return None
        
        markdown = result.stdout
        
        # Parse markdown for handle data
        handle_data = parse_handle_from_markdown(markdown, away_team, home_team)
        
        if handle_data:
            print(f"✅ Handle data fetched successfully!")
            print(f"   Home ({home_team}): {handle_data['home_handle']*100:.1f}% handle")
            print(f"   Away ({away_team}): {handle_data['away_handle']*100:.1f}% handle")
            return handle_data
        else:
            print(f"⚠️  Could not parse handle data from page")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout fetching data from Action Network")
        return None
    except FileNotFoundError:
        print(f"❌ Crawlbase MCP not found. Make sure it's configured in Claude Desktop.")
        print(f"   See CRAWLBASE_INTEGRATION.md for setup instructions")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def parse_handle_from_markdown(markdown, away_team, home_team):
    """
    Parse Action Network markdown for handle percentages.
    
    Action Network shows:
    - Money % (handle) - what % of money is on each side
    - Bets % (tickets) - what % of bets are on each side
    """
    
    # Look for patterns like:
    # "Money: 72% Packers, 28% Eagles"
    # "Handle: Packers 72%, Eagles 28%"
    # "Money%: 72-28"
    
    patterns = [
        # Pattern 1: "Money: 72% Packers"
        r'money[:\s]+(\d+)%\s*' + re.escape(home_team),
        r'money[:\s]+(\d+)%\s*' + re.escape(away_team),
        
        # Pattern 2: "Packers 72%"
        re.escape(home_team) + r'\s+(\d+)%\s+money',
        re.escape(away_team) + r'\s+(\d+)%\s+money',
        
        # Pattern 3: "Handle: 72-28"
        r'handle[:\s]+(\d+)\s*-\s*(\d+)',
    ]
    
    home_handle = None
    away_handle = None
    
    # Try to find home handle
    for pattern in patterns:
        match = re.search(pattern, markdown, re.IGNORECASE)
        if match:
            if home_team.lower() in pattern.lower():
                home_handle = float(match.group(1)) / 100
            elif away_team.lower() in pattern.lower():
                away_handle = float(match.group(1)) / 100
            break
    
    # If found one, calculate the other
    if home_handle and not away_handle:
        away_handle = 1.0 - home_handle
    elif away_handle and not home_handle:
        home_handle = 1.0 - away_handle
    
    if home_handle and away_handle:
        return {
            'home_handle': home_handle,
            'away_handle': away_handle,
            'home_bets': None,  # Can parse this too if available
            'away_bets': None,
            'source': 'action_network',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
    
    return None


def save_handle_data(game_id, handle_data):
    """Save handle data to file for later use."""
    
    data_dir = 'data/handle_data'
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = f"{data_dir}/{game_id}.json"
    
    with open(filepath, 'w') as f:
        json.dump(handle_data, f, indent=2)
    
    print(f"💾 Saved handle data to {filepath}")


def get_cached_handle(game_id, max_age_minutes=60):
    """
    Get cached handle data if it's recent enough.
    
    Args:
        game_id: Unique game identifier
        max_age_minutes: How old cached data can be (default 60 min)
    
    Returns:
        dict or None
    """
    filepath = f"data/handle_data/{game_id}.json"
    
    if not os.path.exists(filepath):
        return None
    
    # Check if file is recent enough
    file_age = datetime.now().timestamp() - os.path.getmtime(filepath)
    if file_age > max_age_minutes * 60:
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 auto_fetch_handle.py <away_team> <home_team>")
        print("Example: python3 auto_fetch_handle.py Eagles Packers")
        sys.exit(1)
    
    away_team = sys.argv[1]
    home_team = sys.argv[2]
    
    # Try to fetch handle data
    handle_data = fetch_action_network_handle(away_team, home_team)
    
    if handle_data:
        game_id = f"{away_team}_{home_team}_{datetime.now().strftime('%Y%m%d')}"
        save_handle_data(game_id, handle_data)
        
        print("\n📊 HANDLE DATA:")
        print(json.dumps(handle_data, indent=2))
    else:
        print("\n❌ Failed to fetch handle data")
        sys.exit(1)
