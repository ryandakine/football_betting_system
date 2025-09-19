#!/usr/bin/env python3
"""
Setup NFL System API Keys
========================

Configures API keys for the NFL betting system using existing keys
from the .env file and helps get missing ones.
"""

import os
import shutil
from pathlib import Path


def setup_nfl_api_keys():
    """Setup API keys for NFL betting system"""
    print("🔑 Setting Up NFL Betting System API Keys")
    print("=" * 50)
    
    # Read existing .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found!")
        return False
    
    # Read current content
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Check what we have
    existing_keys = {}
    for line in content.split('\n'):
        if '=' in line and not line.strip().startswith('#'):
            key, value = line.split('=', 1)
            existing_keys[key.strip()] = value.strip()
    
    print("📋 Current API Key Status:")
    print("-" * 30)
    
    # Check for NFL-specific keys we need
    nfl_keys = {
        'ODDS_API_KEY': 'The Odds API (betting lines)',
        'OPENWEATHER_API_KEY': 'OpenWeatherMap (weather data)',
    }
    
    updates_needed = []
    
    for key, description in nfl_keys.items():
        if key in existing_keys and existing_keys[key]:
            print(f"✅ {description}: Found")
        elif key == 'ODDS_API_KEY' and 'THE_ODDS_API_KEY' in existing_keys:
            print(f"✅ {description}: Found (as THE_ODDS_API_KEY)")
            updates_needed.append(f"ODDS_API_KEY={existing_keys['THE_ODDS_API_KEY']}")
        else:
            print(f"❌ {description}: Missing")
            if key == 'OPENWEATHER_API_KEY':
                updates_needed.append(f"# Get free key from https://openweathermap.org/api")
                updates_needed.append(f"OPENWEATHER_API_KEY=your_openweather_key_here")
    
    # Add updates if needed
    if updates_needed:
        print(f"\n📝 Adding {len(updates_needed)} updates to .env file...")
        
        # Backup original
        backup_file = Path(".env.backup")
        shutil.copy(env_file, backup_file)
        print(f"✅ Created backup: {backup_file}")
        
        # Add new keys
        with open(env_file, 'a') as f:
            f.write("\n# NFL Betting System Keys\n")
            for update in updates_needed:
                f.write(f"{update}\n")
        
        print("✅ Updated .env file with NFL system keys")
    
    # Show setup instructions for missing keys
    if 'OPENWEATHER_API_KEY' not in existing_keys or not existing_keys.get('OPENWEATHER_API_KEY', '').startswith('your_'):
        print("\n🌤️ OpenWeatherMap Setup Instructions:")
        print("1. Visit: https://openweathermap.org/api")
        print("2. Sign up for free account")
        print("3. Get your API key")
        print("4. Replace 'your_openweather_key_here' in .env file")
    
    return True


def test_api_keys():
    """Test the configured API keys"""
    print("\n🧪 Testing API Keys...")
    print("=" * 30)
    
    # Import our validator
    import asyncio
    import sys
    sys.path.append('.')
    
    try:
        from api_key_validator import APIKeyValidator
        
        async def run_test():
            validator = APIKeyValidator()
            results = await validator.validate_all_apis()
            validator.print_validation_report()
        
        asyncio.run(run_test())
        
    except ImportError as e:
        print(f"❌ Could not import API validator: {e}")
        print("Run: python3 api_key_validator.py")


if __name__ == "__main__":
    success = setup_nfl_api_keys()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 NFL API Key Setup Complete!")
        print("=" * 50)
        
        # Test the keys
        test_api_keys()
    else:
        print("\n❌ Setup failed. Please check your .env file.")
