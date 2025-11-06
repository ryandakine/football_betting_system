#!/usr/bin/env python3
"""
Simple YouTube API Test
"""

import requests

# Your YouTube API key from the .env file
YOUTUBE_API_KEY = "AIzaSyAirGlfovjzmg0xUvwA1VGBFDaFgwfQmYY"


def test_youtube_api():
    print("🎥 Testing YouTube API...")
    print(f"🔑 API Key: {YOUTUBE_API_KEY[:10]}...{YOUTUBE_API_KEY[-4:]}")

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": "MLB betting picks today",
        "type": "video",
        "order": "relevance",
        "maxResults": 3,
        "key": YOUTUBE_API_KEY,
    }

    try:
        print("📡 Making API request...")
        response = requests.get(url, params=params, timeout=10)

        print(f"📊 Response Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if "items" in data and len(data["items"]) > 0:
                print(f"✅ SUCCESS! Found {len(data['items'])} videos")

                for i, video in enumerate(data["items"][:3]):
                    print(f"🎥 {i+1}. {video['snippet']['title']}")
                    print(f"   📺 {video['snippet']['channelTitle']}")

                return True
            else:
                print("⚠️ No videos found")
                print(f"Response: {data}")
                return False
        else:
            print(f"❌ API Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data}")

                if response.status_code == 403:
                    print("\n🔧 This is likely a quota or API key issue!")
                    print("1. Check if YouTube Data API v3 is enabled")
                    print("2. Verify your API key permissions")
                    print("3. Check your quota usage")
                elif response.status_code == 400:
                    print("\n🔧 Bad request - check parameters")
                elif response.status_code == 429:
                    print("\n🔧 Quota exceeded - wait and retry")

            except:
                print(f"Raw response: {response.text}")

            return False

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


if __name__ == "__main__":
    test_youtube_api()
