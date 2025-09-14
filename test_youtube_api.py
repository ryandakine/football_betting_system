#!/usr/bin/env python3
"""
Test YouTube API functionality
"""

import os
from datetime import datetime, timedelta

import requests


def test_youtube_api():
    """Test YouTube API with sample search."""
    print("🔍 Testing YouTube API...")
    print("=" * 40)

    # Get API key from environment
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key or api_key == "your_youtube_api_key_here":
        print("❌ YouTube API key not set!")
        print("Please add your YouTube API key to aci.env")
        return False

    # Test search for MLB daily picks
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": "MLB daily picks today",
        "maxResults": 5,
        "order": "relevance",
        "type": "video",
        "publishedAfter": (datetime.now() - timedelta(days=1)).isoformat() + "Z",
        "key": api_key,
    }

    try:
        print("🔍 Searching for 'MLB daily picks today'...")
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            videos = data.get("items", [])

            print(f"✅ Found {len(videos)} videos")
            print("\n📺 Recent MLB Daily Picks Videos:")

            for i, video in enumerate(videos, 1):
                title = video["snippet"]["title"]
                channel = video["snippet"]["channelTitle"]
                published = video["snippet"]["publishedAt"][:10]

                print(f"  {i}. {title}")
                print(f"     Channel: {channel}")
                print(f"     Published: {published}")
                print()

            return True

        elif response.status_code == 403:
            print("❌ API quota exceeded or key invalid")
            return False
        else:
            print(f"❌ API Error: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def analyze_video_content():
    """Analyze video content for betting insights."""
    print("\n📊 Analyzing Video Content...")
    print("=" * 40)

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key or api_key == "your_youtube_api_key_here":
        return

    # Search for videos
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": "MLB picks today",
        "maxResults": 10,
        "order": "relevance",
        "type": "video",
        "key": api_key,
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return

        videos = response.json().get("items", [])

        # Analyze content
        teams_mentioned = {}
        confidence_indicators = []
        positive_words = []
        negative_words = []

        for video in videos:
            title = video["snippet"]["title"].lower()
            description = video["snippet"]["description"].lower()
            text = f"{title} {description}"

            # Count team mentions
            mlb_teams = [
                "yankees",
                "red sox",
                "blue jays",
                "rays",
                "orioles",
                "white sox",
                "guardians",
                "tigers",
                "royals",
                "twins",
                "astros",
                "angels",
                "athletics",
                "mariners",
                "rangers",
                "braves",
                "marlins",
                "mets",
                "phillies",
                "nationals",
                "cubs",
                "reds",
                "brewers",
                "pirates",
                "cardinals",
                "diamondbacks",
                "rockies",
                "dodgers",
                "padres",
                "giants",
            ]

            for team in mlb_teams:
                if team in text:
                    teams_mentioned[team] = teams_mentioned.get(team, 0) + 1

            # Look for confidence indicators
            confidence_words = [
                "lock",
                "guaranteed",
                "sure thing",
                "easy money",
                "best bet",
            ]
            for word in confidence_words:
                if word in text:
                    confidence_indicators.append(word)

            # Sentiment analysis
            if any(word in text for word in ["win", "victory", "dominate", "crush"]):
                positive_words.append(video["snippet"]["title"])
            elif any(word in text for word in ["lose", "struggle", "avoid", "bad bet"]):
                negative_words.append(video["snippet"]["title"])

        # Display results
        print("🏆 Most Mentioned Teams:")
        sorted_teams = sorted(teams_mentioned.items(), key=lambda x: x[1], reverse=True)
        for team, count in sorted_teams[:5]:
            print(f"  {team.title()}: {count} mentions")

        print(f"\n💪 Confidence Indicators: {len(confidence_indicators)}")
        if confidence_indicators:
            print(f"  Found: {', '.join(set(confidence_indicators))}")

        print(f"\n✅ Positive Videos: {len(positive_words)}")
        print(f"❌ Negative Videos: {len(negative_words)}")

    except Exception as e:
        print(f"❌ Analysis error: {e}")


def main():
    """Main function."""
    print("🎯 YouTube API Test for MLB Betting System")
    print("=" * 50)

    # Test basic API functionality
    success = test_youtube_api()

    if success:
        # Analyze content
        analyze_video_content()

        print("\n" + "=" * 50)
        print("✅ YouTube API is working!")
        print("\n🎯 Ready to integrate with n8n workflow")
        print("📋 Next steps:")
        print("1. Import mlb_youtube_analysis_workflow.json to n8n")
        print("2. Configure YouTube API credentials in n8n")
        print("3. Test the complete workflow")
    else:
        print("\n❌ YouTube API setup needed")
        print("📋 Please:")
        print("1. Get YouTube API key from Google Cloud Console")
        print("2. Add it to aci.env file")
        print("3. Run this test again")


if __name__ == "__main__":
    main()
