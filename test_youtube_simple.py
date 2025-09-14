#!/usr/bin/env python3
"""
Simple YouTube API test
"""

import os

import requests


# Load environment variables from aci.env
def load_env():
    with open("aci.env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


load_env()

api_key = os.getenv("YOUTUBE_API_KEY")
print(f"YouTube API Key: {api_key[:10]}..." if api_key else "Not found")

if not api_key or api_key == "your_youtube_api_key_here":
    print("❌ YouTube API key not set!")
    exit(1)

# Test the API
url = "https://www.googleapis.com/youtube/v3/search"
params = {
    "part": "snippet",
    "q": "MLB daily picks today",
    "maxResults": 3,
    "key": api_key,
}

try:
    print("🔍 Testing YouTube API...")
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        videos = data.get("items", [])
        print(f"✅ Found {len(videos)} videos!")

        for i, video in enumerate(videos, 1):
            title = video["snippet"]["title"]
            print(f"  {i}. {title}")

    elif response.status_code == 403:
        print("❌ API quota exceeded or key invalid")
    else:
        print(f"❌ Error: {response.status_code}")

except Exception as e:
    print(f"❌ Error: {e}")
