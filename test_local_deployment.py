#!/usr/bin/env python3
"""
Test NCAA System on Local Machine
==================================

Run this on your LOCAL MACHINE (not in sandbox) to verify everything works.

USAGE:
    python test_local_deployment.py 0c405bc90c59a6a83d77bf1907da0299
"""

import sys
import requests
import json
from pathlib import Path


def test_api_key(api_key: str):
    """Test if Odds API key works"""
    print("="*80)
    print("🔑 Testing Odds API Key")
    print("="*80)
    print()

    url = "https://api.the-odds-api.com/v4/sports/"
    params = {'apiKey': api_key}

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Key VALID!")
            print(f"   Found {len(data)} sports available")

            # Check if NCAA football is available
            ncaaf = [s for s in data if s['key'] == 'americanfootball_ncaaf']
            if ncaaf:
                print(f"   ✅ NCAA Football available!")
                print(f"   Title: {ncaaf[0]['title']}")
                print(f"   Description: {ncaaf[0].get('description', 'N/A')}")
            else:
                print(f"   ⚠️  NCAA Football not in list (might be off-season)")

            # Check requests remaining
            remaining = response.headers.get('x-requests-remaining', 'Unknown')
            print(f"\n   API Requests Remaining: {remaining}")

            return True

        elif response.status_code == 401:
            print(f"❌ Invalid API Key")
            print(f"   Status: {response.status_code}")
            return False

        elif response.status_code == 403:
            print(f"❌ Access Denied")
            print(f"   Status: {response.status_code}")
            print(f"   Your account may not be activated")
            return False

        elif response.status_code == 429:
            print(f"❌ Rate Limit Exceeded")
            print(f"   Free tier: 500 requests/month")
            return False

        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print(f"   Make sure you have internet connection")
        return False


def test_ncaa_odds(api_key: str):
    """Test fetching NCAA odds"""
    print("\n" + "="*80)
    print("🏈 Testing NCAA Football Odds")
    print("="*80)
    print()

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds/"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'spreads',
        'oddsFormat': 'american',
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if len(data) > 0:
                print(f"✅ Found {len(data)} NCAA games with odds!")
                print()
                print("Sample games:")
                for i, game in enumerate(data[:3], 1):
                    home = game['home_team']
                    away = game['away_team']
                    commence = game.get('commence_time', 'N/A')

                    # Get spread
                    spread = "N/A"
                    if game.get('bookmakers'):
                        for bookmaker in game['bookmakers']:
                            for market in bookmaker.get('markets', []):
                                if market['key'] == 'spreads':
                                    for outcome in market['outcomes']:
                                        if outcome['name'] == home:
                                            spread = outcome.get('point', 'N/A')
                                            break

                    print(f"\n{i}. {away} @ {home}")
                    print(f"   Spread: {spread}")
                    print(f"   Commence: {commence}")

                remaining = response.headers.get('x-requests-remaining', 'Unknown')
                print(f"\n   API Requests Remaining: {remaining}")

                return True
            else:
                print(f"⚠️  No games available")
                print(f"   This is normal during off-season")
                print(f"   Check back when 2025 season starts")
                return True  # Still counts as success

        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_models():
    """Test if models are present and loadable"""
    print("\n" + "="*80)
    print("🤖 Testing Models")
    print("="*80)
    print()

    models_dir = Path("models/ncaa")

    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return False

    model_files = ['xgboost_super.pkl', 'neural_net_deep.pkl', 'alt_spread.pkl']

    all_exist = True
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {model_file} not found")
            all_exist = False

    if all_exist:
        print(f"\n✅ All models present!")
        return True
    else:
        print(f"\n⚠️  Some models missing - run quick_retrain_3_models.py")
        return False


def test_imports():
    """Test if required packages are installed"""
    print("\n" + "="*80)
    print("📦 Testing Package Imports")
    print("="*80)
    print()

    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'xgboost': 'XGBoost',
        'requests': 'Requests',
        'bs4': 'BeautifulSoup4',
    }

    all_imported = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - pip install {package}")
            all_imported = False

    if all_imported:
        print(f"\n✅ All packages installed!")
        return True
    else:
        print(f"\n⚠️  Some packages missing - run: pip install -r requirements.txt")
        return False


def main():
    """Main test suite"""
    print("="*80)
    print("🧪 NCAA BETTING SYSTEM - LOCAL DEPLOYMENT TEST")
    print("="*80)
    print()
    print("This script tests if your system is ready to deploy locally.")
    print("Run this on your LOCAL MACHINE, not in Claude Code sandbox.")
    print()

    if len(sys.argv) < 2:
        print("❌ No API key provided")
        print()
        print("Usage: python test_local_deployment.py YOUR_API_KEY")
        print()
        print("Example: python test_local_deployment.py 0c405bc90c59a6a83d77bf1907da0299")
        print()
        sys.exit(1)

    api_key = sys.argv[1]

    # Run tests
    results = []

    results.append(("Package Imports", test_imports()))
    results.append(("Model Files", test_models()))
    results.append(("API Key", test_api_key(api_key)))
    results.append(("NCAA Odds", test_ncaa_odds(api_key)))

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print()

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print()
        print("Your system is ready to use! Next steps:")
        print()
        print("1. Wait for 2025 NCAA season to start")
        print("2. Run weekly: python ncaa_live_predictions_2025.py " + api_key)
        print("3. Place bets on high-edge games")
        print("4. Track results over season")
        print()
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED")
        print("="*80)
        print()
        print("Fix the failing tests above, then run again.")
        print()


if __name__ == "__main__":
    main()
