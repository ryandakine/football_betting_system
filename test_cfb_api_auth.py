#!/usr/bin/env python3
"""Test different authentication methods for CFB Data API"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CFB_DATA_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"

def test_auth_methods():
    """Test different authentication methods"""

    if not API_KEY:
        print("❌ No API key found")
        return

    print("Testing different authentication methods...\n")

    # Test endpoint
    endpoint = "/games"
    params = {"year": 2024, "seasonType": "regular", "week": 1}

    methods = [
        {
            "name": "Bearer Token (Header)",
            "headers": {"Authorization": f"Bearer {API_KEY}"},
            "params": params
        },
        {
            "name": "API Key Header",
            "headers": {"x-api-key": API_KEY},
            "params": params
        },
        {
            "name": "Authorization Header (plain)",
            "headers": {"Authorization": API_KEY},
            "params": params
        },
        {
            "name": "Query Parameter",
            "headers": {},
            "params": {**params, "api_key": API_KEY}
        },
        {
            "name": "No Auth (Free/Public)",
            "headers": {},
            "params": params
        }
    ]

    for method in methods:
        try:
            url = f"{BASE_URL}{endpoint}"
            response = requests.get(
                url,
                headers=method["headers"],
                params=method["params"],
                timeout=10
            )

            status = f"Status: {response.status_code}"

            if response.status_code == 200:
                data = response.json()
                count = len(data) if isinstance(data, list) else 0
                print(f"✅ {method['name']:30s} → SUCCESS! ({count} games)")
                print(f"   Sample game: {data[0].get('home_team', 'N/A') if data else 'N/A'} vs {data[0].get('away_team', 'N/A') if data else 'N/A'}")
                return method  # Return working method
            else:
                error_msg = response.text[:100] if response.text else "No error message"
                print(f"❌ {method['name']:30s} → {status}")
                print(f"   Error: {error_msg}")

        except Exception as e:
            print(f"❌ {method['name']:30s} → Error: {str(e)[:50]}")

    print("\n⚠️  None of the authentication methods worked.")
    print("\nPossible issues:")
    print("1. API key may be invalid or expired")
    print("2. College Football Data API may have changed authentication")
    print("3. Network/firewall issues")
    print("\nTry:")
    print("- Verify your API key at https://collegefootballdata.com")
    print("- Check if you need to activate/confirm your account")
    print("- Try the API key in a browser or Postman first")

if __name__ == "__main__":
    test_auth_methods()
