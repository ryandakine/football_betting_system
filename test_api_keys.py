#!/usr/bin/env python3
"""
Quick API Key Test for MLB Opportunity Detector
"""

import os

import requests

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_api_keys():
    print("🔍 Testing your existing API keys...\n")

    # Check YouTube API
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    if youtube_key:
        print(f"✅ YouTube API Key: {youtube_key[:20]}...")
        try:
            response = requests.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "part": "snippet",
                    "q": "MLB betting picks today",
                    "type": "video",
                    "maxResults": 1,
                    "key": youtube_key,
                },
                timeout=10,
            )
            if response.status_code == 200:
                print("   ✅ YouTube API working!")
            else:
                print(f"   ❌ YouTube API error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ YouTube API test failed: {e}")
    else:
        print("❌ YouTube API Key not found")

    # Check Odds API
    odds_key = os.getenv("THE_ODDS_API_KEY")
    if odds_key:
        print(f"✅ Odds API Key: {odds_key[:20]}...")
        try:
            response = requests.get(
                "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
                params={"regions": "us", "markets": "h2h", "apiKey": odds_key},
                timeout=10,
            )
            if response.status_code == 200:
                print("   ✅ Odds API working!")
            else:
                print(f"   ❌ Odds API error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Odds API test failed: {e}")
    else:
        print("❌ Odds API Key not found")

    # Check Slack Webhook
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        print(f"✅ Slack Webhook: {slack_webhook[:30]}...")
        try:
            test_message = {"text": "🧪 MLB Opportunity Detector test message"}
            response = requests.post(slack_webhook, json=test_message, timeout=10)
            if response.status_code == 200:
                print("   ✅ Slack webhook working!")
            else:
                print(f"   ❌ Slack webhook error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Slack webhook test failed: {e}")
    else:
        print("❌ Slack Webhook not found")

    print("\n🎯 Your API keys are ready for the Enhanced MLB Opportunity Detector!")
    print("📋 Next step: Import the workflow into n8n")


if __name__ == "__main__":
    test_api_keys()
