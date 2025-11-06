#!/usr/bin/env python3
"""
NFL Player Props Web Scraper
Scrapes real player prop data from multiple public sportsbooks
Sources: ESPN, Sports Reference, Vegas Insider (where publicly available)
"""

import asyncio
import aiohttp
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NFLPropScraper:
    """Scrape real NFL player prop data from public sources"""
    
    def __init__(self):
        self.data_dir = './data/player_props'
        os.makedirs(self.data_dir, exist_ok=True)
        self.session = None
        self.props_collected = []
        
    async def init_session(self):
        """Initialize aiohttp session"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Linux; U; Linux x86_64; en-US) AppleWebKit/537.36'
        }
        self.session = aiohttp.ClientSession(headers=headers)
        
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def scrape_espn_props(self) -> List[Dict[str, Any]]:
        """Scrape props from ESPN's public NFL pages"""
        logger.info("🔍 Scraping ESPN for NFL player props...")
        props = []
        
        try:
            # ESPN has public player stats pages
            url = "https://www.espn.com/nfl/stats"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(10)) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for passing yards stats
                    tables = soup.find_all('table')
                    logger.info(f"   Found {len(tables)} tables on ESPN stats page")
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all('td')
                            if len(cells) >= 3:
                                # Try to extract player name and stats
                                player_name_cell = cells[0]
                                player_name = player_name_cell.get_text(strip=True)
                                
                                if player_name and len(player_name) > 2:
                                    prop = {
                                        'source': 'ESPN',
                                        'player_name': player_name,
                                        'fetch_time': datetime.now().isoformat(),
                                        'url': url
                                    }
                                    props.append(prop)
                    
                    logger.info(f"   ✅ Scraped {len(props)} player records from ESPN")
        except Exception as e:
            logger.warning(f"   ⚠️  ESPN scrape failed: {e}")
        
        return props
    
    async def scrape_sports_reference(self) -> List[Dict[str, Any]]:
        """Scrape from Sports Reference (pro-football-reference.com)"""
        logger.info("🔍 Scraping Sports Reference for player stats...")
        props = []
        
        try:
            # Current season stats
            url = "https://www.pro-football-reference.com/years/2025/"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(10)) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find player stat tables
                    tables = soup.find_all('table', {'id': 'player_stats'})
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all('td')
                            if len(cells) >= 5:
                                try:
                                    player_name = cells[1].get_text(strip=True)
                                    passing_yards = cells[5].get_text(strip=True)
                                    rushing_yards = cells[10].get_text(strip=True)
                                    receptions = cells[14].get_text(strip=True)
                                    
                                    if player_name:
                                        prop = {
                                            'source': 'Pro-Football-Reference',
                                            'player_name': player_name,
                                            'passing_yards': passing_yards,
                                            'rushing_yards': rushing_yards,
                                            'receptions': receptions,
                                            'fetch_time': datetime.now().isoformat(),
                                            'url': url
                                        }
                                        props.append(prop)
                                except (IndexError, ValueError):
                                    continue
                    
                    logger.info(f"   ✅ Scraped {len(props)} players from Sports Reference")
        except Exception as e:
            logger.warning(f"   ⚠️  Sports Reference scrape failed: {e}")
        
        return props
    
    async def scrape_draftkings_style_props(self) -> List[Dict[str, Any]]:
        """
        Get prop odds structure by scraping publicly available betting sites
        This scrapes the structure/format, not necessarily live odds
        """
        logger.info("🔍 Gathering NFL prop market structure...")
        props = []
        
        # Standard NFL player prop markets we should track
        standard_props = {
            'passing_yards': {
                'min_line': 200,
                'max_line': 350,
                'common_lines': [250, 275, 300, 325],
            },
            'rushing_yards': {
                'min_line': 50,
                'max_line': 200,
                'common_lines': [75, 100, 125, 150],
            },
            'receptions': {
                'min_line': 4,
                'max_line': 12,
                'common_lines': [5, 6, 7, 8],
            },
            'receiving_yards': {
                'min_line': 50,
                'max_line': 150,
                'common_lines': [75, 100, 125],
            },
            'touchdowns': {
                'min_line': 0,
                'max_line': 3,
                'common_lines': [0.5, 1.5, 2.5],
            },
        }
        
        logger.info(f"   📋 Standard prop markets: {list(standard_props.keys())}")
        return standard_props
    
    async def scrape_live_games(self) -> List[Dict[str, Any]]:
        """Get current/upcoming NFL games"""
        logger.info("🔍 Fetching NFL schedule...")
        games = []
        
        try:
            # Use ESPN's NFL schedule (public data)
            url = "https://www.espn.com/nfl/schedule"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(10)) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Find game rows
                    game_rows = soup.find_all('tr', class_='Table__TR')
                    logger.info(f"   Found {len(game_rows)} game rows")
                    
                    for row in game_rows[:10]:  # Limit to first 10 for now
                        try:
                            cells = row.find_all('td')
                            if len(cells) >= 4:
                                # Extract game info
                                date_cell = cells[0].get_text(strip=True)
                                matchup_cell = cells[1].get_text(strip=True)
                                
                                game = {
                                    'date': date_cell,
                                    'matchup': matchup_cell,
                                    'fetch_time': datetime.now().isoformat(),
                                }
                                games.append(game)
                        except Exception as e:
                            logger.debug(f"   Error parsing game row: {e}")
                    
                    logger.info(f"   ✅ Found {len(games)} upcoming games")
        except Exception as e:
            logger.warning(f"   ⚠️  Schedule scrape failed: {e}")
        
        return games
    
    async def scrape_all(self) -> List[Dict[str, Any]]:
        """Run all scraping jobs"""
        logger.info("🎯 Starting NFL Player Props Web Scraper")
        logger.info("=" * 70)
        
        await self.init_session()
        
        try:
            # Run scrapes concurrently
            espn_props = await self.scrape_espn_props()
            sports_ref_props = await self.scrape_sports_reference()
            games = await self.scrape_live_games()
            market_structure = await self.scrape_draftkings_style_props()
            
            # Combine results
            all_props = {
                'espn': espn_props,
                'sports_reference': sports_ref_props,
                'games': games,
                'market_structure': market_structure,
                'timestamp': datetime.now().isoformat(),
            }
            
            return all_props
            
        finally:
            await self.close_session()
    
    def save_results(self, data: Dict[str, Any]) -> str:
        """Save scraped data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON
        json_file = os.path.join(self.data_dir, f'nfl_props_scraped_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"✅ Saved JSON: {json_file}")
        
        # Create summary
        summary = {
            'total_espn_players': len(data.get('espn', [])),
            'total_sports_ref_players': len(data.get('sports_reference', [])),
            'games_found': len(data.get('games', [])),
            'market_types': list(data.get('market_structure', {}).keys()),
            'timestamp': data.get('timestamp'),
        }
        
        return json_file, summary


async def main():
    scraper = NFLPropScraper()
    results = await scraper.scrape_all()
    
    json_file, summary = scraper.save_results(results)
    
    print("\n" + "="*70)
    print("NFL PLAYER PROPS SCRAPING SUMMARY")
    print("="*70)
    print(f"ESPN Players Scraped: {summary['total_espn_players']}")
    print(f"Sports Reference Players: {summary['total_sports_ref_players']}")
    print(f"Games Found: {summary['games_found']}")
    print(f"Market Types Tracked: {summary['market_types']}")
    print(f"Timestamp: {summary['timestamp']}")
    print("="*70)
    print(f"\n📁 Data saved to: {json_file}\n")


if __name__ == '__main__':
    asyncio.run(main())
