#!/usr/bin/env python3
"""
Configure API keys from existing aci.env file
"""

import os
import re
from pathlib import Path


def extract_api_keys_from_aci_env():
    """Extract API keys from aci.env file and configure them properly."""
    print("🔑 Configuring API Keys from aci.env")
    print("=" * 50)

    # Read aci.env file
    aci_env_path = Path("aci.env")
    if not aci_env_path.exists():
        print("❌ aci.env file not found!")
        return False

    print("✅ Found aci.env file")

    # Read and parse the file
    api_keys = {}
    with open(aci_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                api_keys[key] = value

    print(f"Found {len(api_keys)} environment variables in aci.env")

    # Map the keys to what the daily prediction system needs
    key_mapping = {
        "THE_ODDS_API_KEY": "ODDS_API_KEY",
        "YOUTUBE_API_KEY": "YOUTUBE_API_KEY",
        "SUPABASE_URL": "SUPABASE_URL",
        "SUPABASE_ANON_KEY": "SUPABASE_ANON_KEY",
        "SLACK_WEBHOOK_URL": "SLACK_WEBHOOK_URL",
        "OPENAI_API_KEY": "OPENAI_API_KEY",  # This might not be in aci.env yet
    }

    # Check which keys we have
    available_keys = {}
    missing_keys = []

    for aci_key, system_key in key_mapping.items():
        if aci_key in api_keys and api_keys[aci_key] != f"your_{aci_key.lower()}_here":
            available_keys[system_key] = api_keys[aci_key]
            print(f"✅ {system_key}: Found in aci.env")
        else:
            missing_keys.append(system_key)
            print(f"⚠️  {system_key}: Not found or not set in aci.env")

    # Set environment variables for current session
    for key, value in available_keys.items():
        os.environ[key] = value
        print(f"✅ Set {key} in environment")

    # Create a summary file
    summary_file = Path("api_keys_summary.txt")
    with open(summary_file, "w") as f:
        f.write("API Keys Configuration Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write("✅ Available Keys:\n")
        for key in available_keys:
            f.write(f"  - {key}\n")

        f.write("\n⚠️  Missing Keys:\n")
        for key in missing_keys:
            f.write(f"  - {key}\n")

        f.write("\n📝 Notes:\n")
        f.write("- Available keys are set in environment variables\n")
        f.write("- Missing keys can be added to aci.env file\n")
        f.write("- The system will work with sample data if keys are missing\n")

    print(f"\n📄 Summary saved to: {summary_file}")

    return len(available_keys) > 0


def test_api_keys():
    """Test the configured API keys."""
    print("\n🧪 Testing API Keys")
    print("=" * 20)

    test_results = {}

    # Test The Odds API
    odds_key = os.getenv("ODDS_API_KEY")
    if odds_key and odds_key != "your_odds_api_key_here":
        test_results["ODDS_API_KEY"] = test_odds_api(odds_key)
    else:
        test_results["ODDS_API_KEY"] = "⚠️  Not configured"

    # Test YouTube API
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if youtube_key and youtube_key != "your_youtube_api_key_here":
        test_results["YOUTUBE_API_KEY"] = test_youtube_api(youtube_key)
    else:
        test_results["YOUTUBE_API_KEY"] = "⚠️  Not configured"

    # Test Supabase
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    if supabase_url and supabase_key:
        test_results["SUPABASE"] = test_supabase(supabase_url, supabase_key)
    else:
        test_results["SUPABASE"] = "⚠️  Not configured"

    # Display results
    print("\nAPI Key Test Results:")
    for key, result in test_results.items():
        print(f"  {key}: {result}")


def test_odds_api(api_key):
    """Test The Odds API key."""
    try:
        import requests

        url = "https://api.the-odds-api.com/v4/sports"
        response = requests.get(url, params={"apiKey": api_key})

        if response.status_code == 200:
            return "✅ Valid API key"
        elif response.status_code == 401:
            return "❌ Invalid API key"
        else:
            return f"⚠️  API error: {response.status_code}"
    except Exception as e:
        return f"❌ Test failed: {e}"


def test_youtube_api(api_key):
    """Test YouTube API key."""
    try:
        import requests

        url = "https://www.googleapis.com/youtube/v3/search"
        params = {"part": "snippet", "q": "test", "maxResults": 1, "key": api_key}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return "✅ Valid API key"
        elif response.status_code == 400:
            return "❌ Invalid API key"
        else:
            return f"⚠️  API error: {response.status_code}"
    except Exception as e:
        return f"❌ Test failed: {e}"


def test_supabase(url, key):
    """Test Supabase connection."""
    try:
        import requests

        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        response = requests.get(f"{url}/rest/v1/", headers=headers)

        if response.status_code == 200:
            return "✅ Valid connection"
        else:
            return f"⚠️  Connection error: {response.status_code}"
    except Exception as e:
        return f"❌ Test failed: {e}"


def main():
    """Main function."""
    success = extract_api_keys_from_aci_env()

    if success:
        test_api_keys()

        print("\n" + "=" * 50)
        print("✅ API Keys configured successfully!")
        print("\nNext steps:")
        print("1. Run: .\\run_test.bat")
        print("2. Start the learning API server: python learning_api_server.py")
        print("3. Test daily predictions: python daily_prediction_system.py")
    else:
        print("\n⚠️  No API keys found. The system will work with sample data.")
        print("\nTo add API keys:")
        print("1. Edit aci.env file")
        print("2. Replace placeholder values with actual API keys")
        print("3. Run this script again")


if __name__ == "__main__":
    main()
