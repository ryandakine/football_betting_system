#!/usr/bin/env python3
"""
Setup Real API Keys for Live Data
"""

import os
from pathlib import Path


def setup_api_keys():
    """Interactive setup for real API keys."""
    print("🔑 Setting Up Real API Keys for Live Data")
    print("=" * 50)

    # Read existing aci.env
    aci_env_path = Path("aci.env")
    if not aci_env_path.exists():
        print("❌ aci.env file not found!")
        return

    print("✅ Found aci.env file")
    print("\n📝 Current API Keys Status:")

    # Check current keys
    with open(aci_env_path) as f:
        content = f.read()

    keys_to_check = [
        ("THE_ODDS_API_KEY", "The Odds API"),
        ("YOUTUBE_API_KEY", "YouTube Data API"),
        ("OPENAI_API_KEY", "OpenAI API"),
        ("SLACK_WEBHOOK_URL", "Slack Webhook"),
        ("REDDIT_CLIENT_ID", "Reddit API"),
        ("REDDIT_CLIENT_SECRET", "Reddit API Secret"),
        ("TWITTER_API_KEY", "Twitter API"),
    ]

    for key_name, description in keys_to_check:
        if key_name in content:
            if f"your_{key_name.lower()}_here" in content:
                print(f"  ⚠️  {description}: Not set")
            else:
                print(f"  ✅ {description}: Set")
        else:
            print(f"  ❌ {description}: Missing")

    print("\n🎯 To add real API keys:")
    print("1. Edit aci.env file")
    print("2. Replace placeholder values with actual API keys")
    print("3. Save the file")
    print("4. Run this script again to verify")

    print("\n🔗 API Key Sources:")
    print("  📊 The Odds API: https://the-odds-api.com/")
    print("  📺 YouTube Data API: https://console.cloud.google.com/")
    print("  🤖 OpenAI API: https://platform.openai.com/")
    print("  💬 Slack Webhook: https://api.slack.com/apps")
    print("  📱 Reddit API: https://www.reddit.com/prefs/apps")
    print("  🐦 Twitter API: https://developer.twitter.com/")

    print("\n💡 Tips:")
    print("  - Start with The Odds API for live odds")
    print("  - Add YouTube API for sentiment analysis")
    print("  - OpenAI API for advanced analysis")
    print("  - Slack for notifications")


def test_api_connections():
    """Test API connections if keys are available."""
    print("\n🧪 Testing API Connections")
    print("=" * 30)

    # Test The Odds API
    odds_key = os.getenv("THE_ODDS_API_KEY")
    if odds_key and odds_key != "your_odds_api_key_here":
        print("🔍 Testing The Odds API...")
        try:
            import requests

            url = "https://api.the-odds-api.com/v4/sports"
            response = requests.get(url, params={"apiKey": odds_key})
            if response.status_code == 200:
                print("  ✅ The Odds API: Working")
            else:
                print(f"  ❌ The Odds API: Error {response.status_code}")
        except Exception as e:
            print(f"  ❌ The Odds API: {e}")
    else:
        print("  ⚠️  The Odds API: Not configured")

    # Test YouTube API
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if youtube_key and youtube_key != "your_youtube_api_key_here":
        print("🔍 Testing YouTube API...")
        try:
            import requests

            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": "MLB daily picks",
                "maxResults": 1,
                "key": youtube_key,
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                print("  ✅ YouTube API: Working")
            else:
                print(f"  ❌ YouTube API: Error {response.status_code}")
        except Exception as e:
            print(f"  ❌ YouTube API: {e}")
    else:
        print("  ⚠️  YouTube API: Not configured")


def main():
    """Main function."""
    setup_api_keys()
    test_api_connections()

    print("\n" + "=" * 50)
    print("🎯 Next Steps:")
    print("1. Get API keys from the sources listed above")
    print("2. Update aci.env with real keys")
    print("3. Run this script again to test")
    print("4. Start using live data in your system!")


if __name__ == "__main__":
    main()
