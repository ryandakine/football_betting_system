#!/usr/bin/env python3
"""
API Key Validator for NFL Betting System
=======================================

Comprehensive API key validation and setup for all data sources:
- ESPN API (no key required, but rate limited)
- NFL.com API (no key required, but rate limited)
- The Odds API (requires key)
- OpenWeatherMap API (requires key)
- Pro Football Reference (scraping, no key required)

Validates connectivity and provides setup instructions.
"""

import os
import json
import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIKeyValidator:
    """Validates and manages API keys for the NFL betting system"""
    
    def __init__(self, config_file: str = ".env"):
        self.config_file = config_file
        self.api_keys = {}
        self.validation_results = {}
        
        # API configurations
        self.api_configs = {
            'odds_api': {
                'name': 'The Odds API',
                'key_env': 'ODDS_API_KEY',
                'test_url': 'https://api.the-odds-api.com/v4/sports',
                'requires_key': True,
                'free_tier': True,
                'signup_url': 'https://the-odds-api.com/',
                'description': 'Provides betting odds and lines for NFL games'
            },
            'openweather': {
                'name': 'OpenWeatherMap API',
                'key_env': 'OPENWEATHER_API_KEY',
                'test_url': 'https://api.openweathermap.org/data/2.5/weather',
                'requires_key': True,
                'free_tier': True,
                'signup_url': 'https://openweathermap.org/api',
                'description': 'Weather data for game conditions'
            },
            'espn': {
                'name': 'ESPN API',
                'key_env': None,
                'test_url': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
                'requires_key': False,
                'free_tier': True,
                'signup_url': None,
                'description': 'NFL scores, schedules, and basic stats'
            },
            'nfl_com': {
                'name': 'NFL.com API',
                'key_env': None,
                'test_url': 'https://api.nfl.com/v1/games',
                'requires_key': False,
                'free_tier': True,
                'signup_url': None,
                'description': 'Official NFL data (limited public access)'
            },
            'pro_football_ref': {
                'name': 'Pro Football Reference',
                'key_env': None,
                'test_url': 'https://www.pro-football-reference.com/years/2024/',
                'requires_key': False,
                'free_tier': True,
                'signup_url': None,
                'description': 'Historical stats via web scraping'
            }
        }
        
        self.load_api_keys()
    
    def load_api_keys(self):
        """Load API keys from environment variables and config files"""
        logger.info("Loading API keys from environment and config files...")
        
        # Load from environment variables
        for api_id, config in self.api_configs.items():
            if config['key_env']:
                key = os.getenv(config['key_env'])
                if key:
                    self.api_keys[api_id] = key
                    logger.info(f"✅ Found {config['name']} key in environment")
                else:
                    logger.warning(f"⚠️ No {config['name']} key in environment")
        
        # Try to load from .env file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            
                            # Check if this matches any of our API keys
                            for api_id, config in self.api_configs.items():
                                if config['key_env'] == key:
                                    if api_id not in self.api_keys:
                                        self.api_keys[api_id] = value
                                        logger.info(f"✅ Found {config['name']} key in {self.config_file}")
            except Exception as e:
                logger.error(f"Error reading {self.config_file}: {e}")
        
        # Try to load from config.json
        config_json = "config/api_keys.json"
        if os.path.exists(config_json):
            try:
                with open(config_json, 'r') as f:
                    config_data = json.load(f)
                    for api_id, config in self.api_configs.items():
                        if config['key_env'] and config['key_env'] in config_data:
                            if api_id not in self.api_keys:
                                self.api_keys[api_id] = config_data[config['key_env']]
                                logger.info(f"✅ Found {config['name']} key in {config_json}")
            except Exception as e:
                logger.error(f"Error reading {config_json}: {e}")
    
    async def validate_odds_api(self) -> Tuple[bool, str]:
        """Validate The Odds API key"""
        if 'odds_api' not in self.api_keys:
            return False, "API key not found"
        
        try:
            params = {
                'apiKey': self.api_keys['odds_api'],
                'regions': 'us',
                'markets': 'h2h'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_configs['odds_api']['test_url'],
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list):
                            return True, f"✅ Valid - Found {len(data)} sports available"
                        else:
                            return True, "✅ Valid - API responding correctly"
                    elif response.status == 401:
                        return False, "❌ Invalid API key"
                    elif response.status == 429:
                        return False, "⚠️ Rate limit exceeded - key is valid but quota reached"
                    else:
                        return False, f"❌ API returned status {response.status}"
        
        except asyncio.TimeoutError:
            return False, "❌ Request timeout"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    async def validate_openweather_api(self) -> Tuple[bool, str]:
        """Validate OpenWeatherMap API key"""
        if 'openweather' not in self.api_keys:
            return False, "API key not found"
        
        try:
            params = {
                'appid': self.api_keys['openweather'],
                'q': 'Kansas City,US',
                'units': 'imperial'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_configs['openweather']['test_url'],
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'weather' in data and 'main' in data:
                            temp = data['main'].get('temp', 'N/A')
                            return True, f"✅ Valid - Current KC temp: {temp}°F"
                        else:
                            return True, "✅ Valid - API responding correctly"
                    elif response.status == 401:
                        return False, "❌ Invalid API key"
                    elif response.status == 429:
                        return False, "⚠️ Rate limit exceeded - key is valid but quota reached"
                    else:
                        return False, f"❌ API returned status {response.status}"
        
        except asyncio.TimeoutError:
            return False, "❌ Request timeout"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    async def validate_espn_api(self) -> Tuple[bool, str]:
        """Validate ESPN API (no key required)"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_configs['espn']['test_url'],
                    timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'events' in data:
                            num_games = len(data['events'])
                            return True, f"✅ Available - Found {num_games} NFL events"
                        else:
                            return True, "✅ Available - API responding correctly"
                    elif response.status == 429:
                        return False, "⚠️ Rate limited - try again later"
                    else:
                        return False, f"❌ API returned status {response.status}"
        
        except asyncio.TimeoutError:
            return False, "❌ Request timeout"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    async def validate_nfl_com_api(self) -> Tuple[bool, str]:
        """Validate NFL.com API (no key required, but often restricted)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_configs['nfl_com']['test_url'],
                    headers=headers,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        return True, "✅ Available - NFL.com API accessible"
                    elif response.status == 403:
                        return False, "❌ Access forbidden - NFL.com blocking requests"
                    elif response.status == 429:
                        return False, "⚠️ Rate limited - try again later"
                    else:
                        return False, f"❌ API returned status {response.status}"
        
        except asyncio.TimeoutError:
            return False, "❌ Request timeout"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    async def validate_pro_football_ref(self) -> Tuple[bool, str]:
        """Validate Pro Football Reference (web scraping)"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_configs['pro_football_ref']['test_url'],
                    headers=headers,
                    timeout=15
                ) as response:
                    if response.status == 200:
                        text = await response.text()
                        if 'team_stats' in text or 'NFL' in text:
                            return True, "✅ Available - Pro Football Reference accessible"
                        else:
                            return True, "✅ Available - Site responding"
                    elif response.status == 429:
                        return False, "⚠️ Rate limited - try again later"
                    else:
                        return False, f"❌ Site returned status {response.status}"
        
        except asyncio.TimeoutError:
            return False, "❌ Request timeout"
        except Exception as e:
            return False, f"❌ Error: {str(e)}"
    
    async def validate_all_apis(self) -> Dict[str, Tuple[bool, str]]:
        """Validate all APIs concurrently"""
        logger.info("🔍 Validating all API connections...")
        
        validation_tasks = {
            'odds_api': self.validate_odds_api(),
            'openweather': self.validate_openweather_api(),
            'espn': self.validate_espn_api(),
            'nfl_com': self.validate_nfl_com_api(),
            'pro_football_ref': self.validate_pro_football_ref()
        }
        
        results = {}
        for api_id, task in validation_tasks.items():
            try:
                results[api_id] = await task
            except Exception as e:
                results[api_id] = (False, f"❌ Validation error: {str(e)}")
        
        self.validation_results = results
        return results
    
    def print_validation_report(self):
        """Print a comprehensive validation report"""
        print("\n" + "="*70)
        print("🏈 NFL BETTING SYSTEM - API VALIDATION REPORT")
        print("="*70)
        
        working_apis = 0
        total_apis = len(self.api_configs)
        
        for api_id, config in self.api_configs.items():
            print(f"\n📡 {config['name']}")
            print("-" * 50)
            print(f"Description: {config['description']}")
            print(f"Requires Key: {'Yes' if config['requires_key'] else 'No'}")
            
            if config['requires_key']:
                if api_id in self.api_keys:
                    print(f"API Key: ✅ Found")
                else:
                    print(f"API Key: ❌ Missing")
                    print(f"Sign up at: {config['signup_url']}")
            
            if api_id in self.validation_results:
                is_valid, message = self.validation_results[api_id]
                print(f"Status: {message}")
                if is_valid:
                    working_apis += 1
            else:
                print("Status: ⏳ Not tested")
        
        print("\n" + "="*70)
        print("📊 SUMMARY")
        print("="*70)
        print(f"Working APIs: {working_apis}/{total_apis}")
        print(f"Success Rate: {(working_apis/total_apis)*100:.1f}%")
        
        # Recommendations
        print("\n🔧 RECOMMENDATIONS")
        print("-" * 30)
        
        missing_keys = []
        for api_id, config in self.api_configs.items():
            if config['requires_key'] and api_id not in self.api_keys:
                missing_keys.append((api_id, config))
        
        if missing_keys:
            print("Missing API Keys:")
            for api_id, config in missing_keys:
                print(f"  • {config['name']}: Get key from {config['signup_url']}")
                print(f"    Set environment variable: {config['key_env']}")
        
        failed_apis = []
        for api_id, (is_valid, message) in self.validation_results.items():
            if not is_valid and "Rate limit" not in message:
                failed_apis.append((api_id, self.api_configs[api_id], message))
        
        if failed_apis:
            print("\nFailed API Connections:")
            for api_id, config, error in failed_apis:
                print(f"  • {config['name']}: {error}")
        
        if working_apis == total_apis:
            print("\n🎉 ALL SYSTEMS GO! Your NFL betting system is ready.")
        elif working_apis >= 3:
            print("\n✅ GOOD TO GO! Core systems are working.")
        else:
            print("\n⚠️ SETUP NEEDED! Please configure missing API keys.")
    
    def create_sample_env_file(self):
        """Create a sample .env file with API key placeholders"""
        sample_content = """# NFL Betting System API Keys
# Copy this file to .env and add your actual API keys

# The Odds API - Get your key from https://the-odds-api.com/
ODDS_API_KEY=your_odds_api_key_here

# OpenWeatherMap API - Get your key from https://openweathermap.org/api
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Note: ESPN and NFL.com APIs don't require keys
# Pro Football Reference uses web scraping (no key required)
"""
        
        sample_file = ".env.example"
        with open(sample_file, 'w') as f:
            f.write(sample_content)
        
        print(f"📝 Created sample environment file: {sample_file}")
        print("Copy this to .env and add your actual API keys")
    
    def get_setup_instructions(self) -> Dict[str, str]:
        """Get detailed setup instructions for each API"""
        instructions = {}
        
        for api_id, config in self.api_configs.items():
            if config['requires_key']:
                instructions[api_id] = f"""
{config['name']} Setup:
1. Visit: {config['signup_url']}
2. Sign up for a free account
3. Get your API key
4. Set environment variable: {config['key_env']}=your_key_here
5. Or add to .env file: {config['key_env']}=your_key_here
"""
        
        return instructions


async def main():
    """Main validation and setup function"""
    print("🏈 NFL Betting System - API Key Validator")
    print("=" * 50)
    
    validator = APIKeyValidator()
    
    # Validate all APIs
    results = await validator.validate_all_apis()
    
    # Print comprehensive report
    validator.print_validation_report()
    
    # Create sample .env file if it doesn't exist
    if not os.path.exists(".env") and not os.path.exists(".env.example"):
        validator.create_sample_env_file()
    
    # Print setup instructions for missing keys
    instructions = validator.get_setup_instructions()
    if instructions:
        print("\n" + "="*70)
        print("📋 DETAILED SETUP INSTRUCTIONS")
        print("="*70)
        for api_id, instruction in instructions.items():
            if api_id not in validator.api_keys:
                print(instruction)


if __name__ == "__main__":
    asyncio.run(main())
