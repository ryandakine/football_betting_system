#!/usr/bin/env python3
"""
Fetch real NFL player prop data from The Odds API
Stores data for backtesting the Prop Vet Exploit Engine
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import pandas as pd
from dotenv import load_dotenv

# Load .env file
load_dotenv('.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealPropDataFetcher:
    """Fetch real NFL prop data from The Odds API"""
    
    # NFL player prop markets
    PROP_MARKETS = {
        'player_pass_yards': 'Passing Yards',
        'player_pass_tds': 'Passing TDs',
        'player_pass_interceptions': 'Pass Interceptions',
        'player_rush_yards': 'Rushing Yards',
        'player_rush_tds': 'Rushing TDs',
        'player_receptions': 'Receptions',
        'player_reception_yards': 'Reception Yards',
        'player_reception_tds': 'Reception TDs',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        # Try both API keys from .env
        self.api_key = api_key or os.getenv('THE_ODDS_API_KEY') or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("No ODDS API key found - set THE_ODDS_API_KEY or ODDS_API_KEY")
        self.base_url = "https://api.the-odds-api.com/v4"
        self.data_dir = './data/player_props'
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"🔑 Using API key: {self.api_key[:10]}...")
    
    async def fetch_nfl_games(self) -> List[Dict[str, Any]]:
        """Fetch upcoming NFL games"""
        url = f"{self.base_url}/sports/americanfootball_nfl/events"
        params = {
            'apiKey': self.api_key,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"✅ Fetched {len(data)} NFL games")
                        return data
                    else:
                        logger.error(f"❌ Failed to fetch games: {resp.status}")
                        return []
        except Exception as e:
            logger.error(f"❌ Error fetching games: {e}")
            return []
    
    async def fetch_prop_odds(self, game_id: str, markets: List[str] = None) -> Dict[str, Any]:
        """Fetch prop odds for a specific game"""
        url = f"{self.base_url}/sports/americanfootball_nfl/events/{game_id}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'oddsFormat': 'american',
        }
        if markets:
            params['markets'] = ','.join(markets)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(30)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data
                    else:
                        logger.warning(f"⚠️  Game {game_id}: {resp.status}")
                        return {}
        except Exception as e:
            logger.warning(f"⚠️  Error fetching props for {game_id}: {e}")
            return {}
    
    def parse_prop_data(self, game: Dict, prop_data: Dict) -> List[Dict[str, Any]]:
        """Parse and structure prop data from API response"""
        props = []
        
        if not prop_data.get('bookmakers'):
            return props
        
        game_id = game.get('id')
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        commence_time = game.get('commence_time')
        
        # Use first bookmaker with most comprehensive data
        bookmakers = sorted(
            prop_data.get('bookmakers', []),
            key=lambda b: len(b.get('markets', [])),
            reverse=True
        )
        
        if not bookmakers:
            return props
        
        bookmaker = bookmakers[0]
        bookmaker_name = bookmaker.get('title', 'Unknown')
        
        for market in bookmaker.get('markets', []):
            market_key = market.get('key', '')
            
            # Skip if not a recognized prop market
            if market_key not in self.PROP_MARKETS:
                continue
            
            market_name = self.PROP_MARKETS[market_key]
            
            for outcome in market.get('outcomes', []):
                # Parse player name and over/under
                description = outcome.get('description', '')
                name = outcome.get('name', '')
                
                # Determine if over or under
                price = outcome.get('price')
                is_over = 'Over' in description or 'Over' in name
                is_under = 'Under' in description or 'Under' in name
                
                if not (is_over or is_under):
                    continue
                
                # Extract line from description (format: "PlayerName Over/Under X.Y")
                parts = description.split()
                line = None
                player_name = description
                
                for i, part in enumerate(parts):
                    try:
                        line = float(part)
                        player_name = ' '.join(parts[:i])
                        break
                    except ValueError:
                        continue
                
                if not line:
                    continue
                
                prop_entry = {
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'commence_time': commence_time,
                    'bookmaker': bookmaker_name,
                    'player_name': player_name.strip(),
                    'market_type': market_key,
                    'market_name': market_name,
                    'line': line,
                    'over_under': 'Over' if is_over else 'Under',
                    'odds': price,
                    'fetch_time': datetime.now().isoformat(),
                }
                props.append(prop_entry)
        
        return props
    
    async def fetch_all_props(self) -> List[Dict[str, Any]]:
        """Fetch all available prop data"""
        logger.info("🎯 Starting real prop data fetch from The Odds API")
        
        games = await self.fetch_nfl_games()
        if not games:
            logger.error("❌ No games fetched")
            return []
        
        all_props = []
        market_list = list(self.PROP_MARKETS.keys())
        logger.info(f"📊 Requesting markets: {market_list}")
        
        # Fetch first game to debug available markets (try without market filter)
        if games:
            first_game = games[0]
            logger.info(f"🔍 Checking first game {first_game.get('id')} for available markets...")
            first_props = await self.fetch_prop_odds(first_game.get('id'), markets=None)
            if first_props.get('bookmakers'):
                for bm in first_props['bookmakers']:
                    available_markets = [m.get('key') for m in bm.get('markets', [])]
                    logger.info(f"   Available markets from {bm.get('title')}: {available_markets[:10]}...")  # Show first 10
        
        # Fetch props concurrently
        tasks = [
            self.fetch_prop_odds(game.get('id'), market_list)
            for game in games
        ]
        
        results = await asyncio.gather(*tasks)
        
        for i, (game, prop_data) in enumerate(zip(games, results)):
            props = self.parse_prop_data(game, prop_data)
            all_props.extend(props)
            
            if (i + 1) % 10 == 0:
                logger.info(f"   Processed {i + 1}/{len(games)} games ({len(all_props)} props collected)")
        
        logger.info(f"✅ Collected {len(all_props)} total prop records")
        return all_props
    
    def save_props_to_parquet(self, props: List[Dict[str, Any]]) -> str:
        """Save props to parquet file"""
        if not props:
            logger.error("❌ No props to save")
            return ""
        
        df = pd.DataFrame(props)
        
        # Add metadata
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"nfl_player_props_{timestamp}.parquet"
        filepath = os.path.join(self.data_dir, filename)
        
        df.to_parquet(filepath, index=False)
        logger.info(f"✅ Saved {len(df)} props to {filepath}")
        
        # Also save CSV for inspection
        csv_filename = f"nfl_player_props_{timestamp}.csv"
        csv_filepath = os.path.join(self.data_dir, csv_filename)
        df.to_csv(csv_filepath, index=False)
        logger.info(f"✅ Also saved CSV: {csv_filepath}")
        
        return filepath
    
    def save_props_to_json(self, props: List[Dict[str, Any]]) -> str:
        """Save props to JSON file for review"""
        if not props:
            logger.error("❌ No props to save")
            return ""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"nfl_player_props_{timestamp}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(props, f, indent=2)
        
        logger.info(f"✅ Saved {len(props)} props to {filepath}")
        return filepath


def main():
    try:
        fetcher = RealPropDataFetcher()
        
        # Fetch all available prop data
        props = asyncio.run(fetcher.fetch_all_props())
        
        if not props:
            logger.error("❌ No prop data collected")
            return
        
        # Save to both parquet (for backtest) and JSON (for inspection)
        parquet_path = fetcher.save_props_to_parquet(props)
        json_path = fetcher.save_props_to_json(props)
        
        # Print summary
        df = pd.DataFrame(props)
        print("\n" + "="*80)
        print("REAL NFL PLAYER PROP DATA SUMMARY")
        print("="*80)
        print(f"Total Props Collected: {len(df)}")
        print(f"\nUnique Markets:")
        print(df['market_name'].value_counts())
        print(f"\nUnique Players: {df['player_name'].nunique()}")
        print(f"Unique Games: {df['game_id'].nunique()}")
        print(f"Bookmakers: {', '.join(df['bookmaker'].unique())}")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
