#!/usr/bin/env python3
"""
Auto Line Shopping - Scrapes DraftKings, FanDuel, BetMGM for best lines
Uses Crawlbase MCP for reliable scraping
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime

SPORTSBOOKS = {
    'DraftKings': 'https://sportsbook.draftkings.com/leagues/football/nfl',
    'FanDuel': 'https://sportsbook.fanduel.com/navigation/nfl',
    'BetMGM': 'https://sports.betmgm.com/en/sports/football-11/betting/usa-9/nfl-35'
}


def fetch_best_lines(away_team, home_team):
    """
    Scrape all major sportsbooks for best lines.
    
    Args:
        away_team: "Eagles"
        home_team: "Packers"
    
    Returns:
        dict: {
            'spread': {
                'best_line': -1.0,
                'best_odds': -110,
                'best_book': 'FanDuel',
                'all_books': {
                    'DraftKings': {'line': -1.5, 'odds': -110},
                    'FanDuel': {'line': -1.0, 'odds': -110},
                    'BetMGM': {'line': -2.0, 'odds': -105}
                }
            },
            'total': {
                'best_over': 45.5,
                'best_over_odds': -110,
                'best_under': 45.5,
                'best_under_odds': -105,
                'best_book': 'BetMGM'
            },
            'timestamp': '2025-01-14T14:05:00Z'
        }
    """
    
    print(f"🔍 Fetching lines for {away_team} @ {home_team}...")
    
    all_lines = {}
    
    for book_name, book_url in SPORTSBOOKS.items():
        print(f"\n📍 Checking {book_name}...")
        
        lines = scrape_sportsbook(book_name, book_url, away_team, home_team)
        
        if lines:
            all_lines[book_name] = lines
            print(f"   ✅ {book_name}: {lines.get('spread_line', 'N/A')} @ {lines.get('spread_odds', 'N/A')}")
        else:
            print(f"   ⚠️  Could not fetch {book_name} lines")
    
    if not all_lines:
        print("\n❌ No lines found from any sportsbook")
        return None
    
    # Find best lines across all books
    best_lines = find_best_lines(all_lines, home_team)
    
    print(f"\n🎯 BEST LINES:")
    print(f"   Spread: {best_lines['spread']['best_book']} {best_lines['spread']['best_line']} @ {best_lines['spread']['best_odds']}")
    print(f"   Total: {best_lines['total']['best_book']} O/U {best_lines['total']['best_over']} @ {best_lines['total']['best_over_odds']}")
    
    return best_lines


def scrape_sportsbook(book_name, url, away_team, home_team):
    """
    Scrape a single sportsbook using Crawlbase MCP.
    
    Returns:
        dict: {
            'spread_line': -1.5,
            'spread_odds': -110,
            'total_line': 45.5,
            'over_odds': -110,
            'under_odds': -110
        }
    """
    
    try:
        # Use JavaScript token for dynamic sportsbook sites
        result = subprocess.run(
            ['npx', '@crawlbase/mcp@latest', 'crawl_markdown', url],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, 'USE_JS_TOKEN': 'true'}  # Use JS token for dynamic sites
        )
        
        if result.returncode != 0:
            return None
        
        markdown = result.stdout
        
        # Parse markdown for game lines
        lines = parse_lines_from_markdown(markdown, away_team, home_team, book_name)
        
        return lines
        
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"   Error scraping {book_name}: {e}")
        return None


def parse_lines_from_markdown(markdown, away_team, home_team, book_name):
    """
    Parse sportsbook markdown for spread and total lines.
    
    Each sportsbook has different HTML structure, so we try multiple patterns.
    """
    
    # Look for patterns like:
    # "Packers -1.5 (-110)"
    # "Eagles +1.5 (-105)"
    # "Over 45.5 (-110)"
    # "Under 45.5 (-110)"
    
    lines = {}
    
    # Try to find spread
    spread_patterns = [
        # Pattern 1: "Packers -1.5 (-110)"
        re.escape(home_team) + r'\s+([-+]?\d+\.?\d*)\s+\(([-+]\d+)\)',
        
        # Pattern 2: "-1.5 | -110 | Packers"
        r'([-+]?\d+\.?\d*)\s+\|\s+([-+]\d+)\s+\|.*?' + re.escape(home_team),
        
        # Pattern 3: Spread: -1.5, Odds: -110
        r'spread[:\s]+([-+]?\d+\.?\d*).*?odds[:\s]+([-+]\d+)',
    ]
    
    for pattern in spread_patterns:
        match = re.search(pattern, markdown, re.IGNORECASE)
        if match:
            lines['spread_line'] = float(match.group(1))
            lines['spread_odds'] = int(match.group(2))
            break
    
    # Try to find total
    total_patterns = [
        # Pattern 1: "Over 45.5 (-110)"
        r'over\s+(\d+\.?\d*)\s+\(([-+]\d+)\)',
        
        # Pattern 2: "Total: 45.5, Over: -110"
        r'total[:\s]+(\d+\.?\d*).*?over[:\s]+([-+]\d+)',
    ]
    
    for pattern in total_patterns:
        match = re.search(pattern, markdown, re.IGNORECASE)
        if match:
            lines['total_line'] = float(match.group(1))
            lines['over_odds'] = int(match.group(2))
            # Assume under odds are similar if not found
            lines['under_odds'] = int(match.group(2))
            break
    
    return lines if lines else None


def find_best_lines(all_lines, home_team):
    """
    Compare all sportsbooks and find best lines for each bet type.
    
    Best spread = most favorable number + best odds
    Best total = best number for over/under + best odds
    """
    
    best = {
        'spread': {'best_line': None, 'best_odds': -999, 'best_book': None, 'all_books': {}},
        'total': {'best_over': None, 'best_over_odds': -999, 'best_under_odds': -999, 'best_book': None},
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    # Find best spread
    for book, lines in all_lines.items():
        if 'spread_line' in lines:
            best['spread']['all_books'][book] = {
                'line': lines['spread_line'],
                'odds': lines['spread_odds']
            }
            
            # Better spread = lower absolute value (less points to cover)
            if best['spread']['best_line'] is None or abs(lines['spread_line']) < abs(best['spread']['best_line']):
                best['spread']['best_line'] = lines['spread_line']
                best['spread']['best_odds'] = lines['spread_odds']
                best['spread']['best_book'] = book
    
    # Find best total
    for book, lines in all_lines.items():
        if 'total_line' in lines:
            if best['total']['best_over'] is None:
                best['total']['best_over'] = lines['total_line']
                best['total']['best_over_odds'] = lines['over_odds']
                best['total']['best_under_odds'] = lines.get('under_odds', lines['over_odds'])
                best['total']['best_book'] = book
            
            # Better over = higher total (more margin)
            # Better under = lower total (more margin)
            if lines['over_odds'] > best['total']['best_over_odds']:
                best['total']['best_over_odds'] = lines['over_odds']
                best['total']['best_book'] = book
    
    return best


def save_line_data(game_id, line_data):
    """Save line shopping results to file."""
    
    data_dir = 'data/line_shopping'
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = f"{data_dir}/{game_id}.json"
    
    with open(filepath, 'w') as f:
        json.dump(line_data, f, indent=2)
    
    print(f"💾 Saved line data to {filepath}")


def calculate_clv_improvement(all_books):
    """
    Calculate Closing Line Value (CLV) improvement from line shopping.
    
    Returns:
        float: Expected ROI boost from getting best line
    """
    
    if len(all_books) < 2:
        return 0.0
    
    # Best line vs worst line
    lines = [book['line'] for book in all_books.values()]
    best_line = min(lines, key=abs)
    worst_line = max(lines, key=abs)
    
    # Each 0.5 point = ~2% ROI improvement
    line_diff = abs(worst_line - best_line)
    clv_improvement = line_diff * 0.04  # 4% per point
    
    return clv_improvement


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 auto_line_shopping.py <away_team> <home_team>")
        print("Example: python3 auto_line_shopping.py Eagles Packers")
        sys.exit(1)
    
    away_team = sys.argv[1]
    home_team = sys.argv[2]
    
    # Fetch best lines
    best_lines = fetch_best_lines(away_team, home_team)
    
    if best_lines:
        game_id = f"{away_team}_{home_team}_{datetime.now().strftime('%Y%m%d')}"
        save_line_data(game_id, best_lines)
        
        # Calculate CLV improvement
        if best_lines['spread']['all_books']:
            clv = calculate_clv_improvement(best_lines['spread']['all_books'])
            print(f"\n💰 CLV Improvement: +{clv*100:.1f}% ROI from line shopping")
        
        print("\n📊 LINE SHOPPING RESULTS:")
        print(json.dumps(best_lines, indent=2))
    else:
        print("\n❌ Failed to fetch lines")
        sys.exit(1)
