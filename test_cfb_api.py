#!/usr/bin/env python3
"""Test College Football Data API access and determine tier level"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFB_DATA_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"

def test_api_access():
    """Test API key and determine access level"""

    if not API_KEY:
        print("❌ No API key found in .env file")
        return

    print("🔍 Testing College Football Data API access...\n")

    headers = {"Authorization": f"Bearer {API_KEY}"}

    # Test endpoints to determine tier
    tests = [
        {
            "name": "Basic Games Data",
            "endpoint": "/games",
            "params": {"year": 2024, "seasonType": "regular", "week": 1},
            "tier": "FREE"
        },
        {
            "name": "Team Stats",
            "endpoint": "/stats/season",
            "params": {"year": 2024},
            "tier": "FREE"
        },
        {
            "name": "Rankings",
            "endpoint": "/rankings",
            "params": {"year": 2024, "week": 1},
            "tier": "FREE"
        },
        {
            "name": "SP+ Ratings",
            "endpoint": "/ratings/sp",
            "params": {"year": 2024},
            "tier": "SILVER/GOLD"
        },
        {
            "name": "EPA Data",
            "endpoint": "/metrics/ppa/games",
            "params": {"year": 2024, "week": 1},
            "tier": "SILVER/GOLD"
        },
        {
            "name": "Win Probability",
            "endpoint": "/metrics/wp",
            "params": {"gameId": 401628467},
            "tier": "SILVER/GOLD"
        },
        {
            "name": "Recruiting",
            "endpoint": "/recruiting/teams",
            "params": {"year": 2024},
            "tier": "BRONZE+"
        }
    ]

    results = []

    for test in tests:
        try:
            url = f"{BASE_URL}{test['endpoint']}"
            response = requests.get(url, params=test['params'], headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                count = len(data) if isinstance(data, list) else 1
                status = f"✅ {count} records"
                results.append((test['name'], test['tier'], True))
            elif response.status_code == 401:
                status = "❌ Unauthorized (upgrade needed)"
                results.append((test['name'], test['tier'], False))
            elif response.status_code == 403:
                status = "❌ Forbidden (upgrade needed)"
                results.append((test['name'], test['tier'], False))
            else:
                status = f"❌ Error {response.status_code}"
                results.append((test['name'], test['tier'], False))

            print(f"{test['name']:25s} [{test['tier']:12s}] → {status}")

        except Exception as e:
            print(f"{test['name']:25s} [{test['tier']:12s}] → ❌ Error: {str(e)[:50]}")
            results.append((test['name'], test['tier'], False))

    # Determine tier
    print("\n" + "="*70)
    print("📊 API TIER ANALYSIS")
    print("="*70)

    has_free = any(r[2] for r in results if r[1] == "FREE")
    has_bronze = any(r[2] for r in results if r[1] == "BRONZE+")
    has_silver = any(r[2] for r in results if r[1] == "SILVER/GOLD")

    if has_silver:
        tier = "SILVER or GOLD"
        print(f"🎉 Your API key has {tier} tier access!")
        print("✅ You have access to advanced metrics (SP+, EPA, Win Probability)")
        print("✅ This is the BEST tier for betting systems!")
    elif has_bronze:
        tier = "BRONZE"
        print(f"✅ Your API key has {tier} tier access!")
        print("✅ You have recruiting data access")
        print("💡 Consider upgrading to Silver for SP+ and EPA data")
    elif has_free:
        tier = "FREE"
        print(f"✅ Your API key has {tier} tier access!")
        print("✅ You can collect game results, stats, and rankings")
        print("💡 This is enough to build a working system!")
        print("💡 Consider upgrading to Bronze/Silver for advanced metrics")
    else:
        tier = "UNKNOWN"
        print("❌ Unable to determine tier or API key invalid")

    print("\n📈 Next Steps:")
    if has_silver:
        print("1. Collect SP+ ratings (huge value for predictions)")
        print("2. Collect EPA data for play efficiency")
        print("3. Get win probability for live betting")
        print("4. Run full historical data collection (2015-2024)")
    elif has_bronze:
        print("1. Collect recruiting data")
        print("2. Collect game results and team stats")
        print("3. Consider upgrading to Silver for SP+ ($25/month)")
    else:
        print("1. Collect game results (2015-2024)")
        print("2. Collect team stats and rankings")
        print("3. Validate your backtester with real data")
        print("4. If profitable, upgrade to Bronze/Silver")

    return tier, results

if __name__ == "__main__":
    test_api_access()
