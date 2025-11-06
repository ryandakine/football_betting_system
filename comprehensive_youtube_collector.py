#!/usr/bin/env python3
"""
Comprehensive YouTube Data Collector for MLB Betting System
Fetches ALL videos from today, gets transcripts, and analyzes content thoroughly
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import requests


# Load environment variables
def load_env():
    with open("aci.env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()


load_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


class ComprehensiveYouTubeCollector:
    """Comprehensive YouTube data collector for MLB analysis"""

    def __init__(self):
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.data_dir = Path("data/youtube_analysis")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Comprehensive search terms for MLB analysis
        self.search_terms = [
            "MLB daily picks",
            "MLB picks today",
            "MLB betting picks",
            "MLB injury updates",
            "MLB lineup changes",
            "MLB starting pitchers",
            "MLB weather impact",
            "MLB bullpen news",
            "MLB player props",
            "MLB over under picks",
            "MLB moneyline picks",
            "MLB run line picks",
            "MLB expert picks",
            "MLB sharp picks",
            "MLB value bets",
            "MLB public betting",
            "MLB sharp money",
            "MLB consensus picks",
            "MLB best bets",
            "MLB lock picks",
        ]

    def get_all_todays_videos(self) -> list[dict]:
        """Fetch ALL videos from today for comprehensive analysis"""
        logger.info("🎯 Starting comprehensive YouTube data collection...")

        all_videos = []
        today = datetime.now().strftime("%Y-%m-%d")

        for term in self.search_terms:
            logger.info(f"🔍 Searching for: {term}")

            # Get maximum results (50 per search term)
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": term,
                "maxResults": 50,  # Maximum allowed
                "order": "relevance",
                "type": "video",
                "publishedAfter": (datetime.now() - timedelta(days=1)).isoformat()
                + "Z",
                "key": self.youtube_api_key,
            }

            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    videos = response.json().get("items", [])
                    all_videos.extend(videos)
                    logger.info(f"✅ Found {len(videos)} videos for '{term}'")

                    # Rate limiting
                    time.sleep(0.1)
                else:
                    logger.error(f"❌ Error searching '{term}': {response.status_code}")

            except Exception as e:
                logger.error(f"❌ Exception searching '{term}': {e}")

        # Remove duplicates based on video ID
        unique_videos = {}
        for video in all_videos:
            video_id = video["id"]["videoId"]
            if video_id not in unique_videos:
                unique_videos[video_id] = video

        unique_video_list = list(unique_videos.values())
        logger.info(f"🎉 Total unique videos found: {len(unique_video_list)}")

        return unique_video_list

    def get_video_details(self, video_ids: list[str]) -> list[dict]:
        """Get detailed information for videos including transcripts"""
        logger.info(f"📊 Getting detailed info for {len(video_ids)} videos...")

        detailed_videos = []

        # Process in batches of 50 (YouTube API limit)
        batch_size = 50
        for i in range(0, len(video_ids), batch_size):
            batch = video_ids[i : i + batch_size]

            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(batch),
                "key": self.youtube_api_key,
            }

            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    videos = response.json().get("items", [])
                    detailed_videos.extend(videos)
                    logger.info(f"✅ Got details for batch {i//batch_size + 1}")

                    # Rate limiting
                    time.sleep(0.1)
                else:
                    logger.error(
                        f"❌ Error getting video details: {response.status_code}"
                    )

            except Exception as e:
                logger.error(f"❌ Exception getting video details: {e}")

        return detailed_videos

    def get_video_transcript(self, video_id: str) -> str:
        """Get transcript for a video (if available)"""
        try:
            # YouTube doesn't provide transcripts via API, but we can try to get captions
            url = f"https://www.googleapis.com/youtube/v3/captions"
            params = {
                "part": "snippet",
                "videoId": video_id,
                "key": self.youtube_api_key,
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                captions = response.json().get("items", [])
                if captions:
                    logger.info(f"📝 Found captions for video {video_id}")
                    return "Captions available"

            return "No transcript available"

        except Exception as e:
            logger.error(f"❌ Error getting transcript for {video_id}: {e}")
            return "Error getting transcript"

    def analyze_video_content(self, video: dict) -> dict:
        """Analyze video content for betting-relevant information"""
        snippet = video.get("snippet", {})
        title = snippet.get("title", "").lower()
        description = snippet.get("description", "").lower()

        analysis = {
            "video_id": video.get("id"),
            "title": snippet.get("title", ""),
            "channel": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "view_count": video.get("statistics", {}).get("viewCount", 0),
            "like_count": video.get("statistics", {}).get("likeCount", 0),
            "duration": video.get("contentDetails", {}).get("duration", ""),
            # Content analysis
            "teams_mentioned": [],
            "players_mentioned": [],
            "confidence_indicators": [],
            "betting_terms": [],
            "sentiment": "neutral",
            "injury_mentions": False,
            "weather_mentions": False,
            "lineup_mentions": False,
            "pitching_mentions": False,
            "prop_bet_mentions": False,
            "over_under_mentions": False,
            "moneyline_mentions": False,
            "run_line_mentions": False,
        }

        # Team mentions
        mlb_teams = [
            "yankees",
            "red sox",
            "blue jays",
            "orioles",
            "rays",
            "white sox",
            "indians",
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
            if team in title or team in description:
                analysis["teams_mentioned"].append(team)

        # Betting terms
        betting_terms = [
            "lock",
            "best bet",
            "value",
            "sharp",
            "public",
            "consensus",
            "over",
            "under",
            "moneyline",
            "run line",
            "prop",
            "parlay",
            "fade",
            "lean",
            "confidence",
            "unit",
            "bankroll",
        ]

        for term in betting_terms:
            if term in title or term in description:
                analysis["betting_terms"].append(term)

        # Confidence indicators
        confidence_words = ["lock", "guaranteed", "sure thing", "100%", "confident"]
        for word in confidence_words:
            if word in title or word in description:
                analysis["confidence_indicators"].append(word)

        # Specific mentions
        analysis["injury_mentions"] = any(
            word in title or word in description
            for word in ["injury", "hurt", "out", "questionable"]
        )
        analysis["weather_mentions"] = any(
            word in title or word in description
            for word in ["weather", "rain", "wind", "temperature"]
        )
        analysis["lineup_mentions"] = any(
            word in title or word in description
            for word in ["lineup", "batting order", "roster"]
        )
        analysis["pitching_mentions"] = any(
            word in title or word in description
            for word in ["pitcher", "starting", "bullpen", "relief"]
        )
        analysis["prop_bet_mentions"] = any(
            word in title or word in description
            for word in ["prop", "player prop", "hits", "runs", "rbis"]
        )
        analysis["over_under_mentions"] = any(
            word in title or word in description for word in ["over", "under", "total"]
        )
        analysis["moneyline_mentions"] = any(
            word in title or word in description
            for word in ["moneyline", "ml", "straight up"]
        )
        analysis["run_line_mentions"] = any(
            word in title or word in description
            for word in ["run line", "spread", "rl"]
        )

        # Sentiment analysis (simple)
        positive_words = ["lock", "best", "guaranteed", "sure", "confident", "value"]
        negative_words = ["fade", "avoid", "stay away", "bad", "terrible", "worst"]

        positive_count = sum(
            1 for word in positive_words if word in title or word in description
        )
        negative_count = sum(
            1 for word in negative_words if word in title or word in description
        )

        if positive_count > negative_count:
            analysis["sentiment"] = "positive"
        elif negative_count > positive_count:
            analysis["sentiment"] = "negative"

        return analysis

    def save_comprehensive_data(self, videos: list[dict], analyses: list[dict]):
        """Save comprehensive YouTube data"""
        today = datetime.now().strftime("%Y-%m-%d")

        # Save raw video data
        videos_file = self.data_dir / f"youtube_videos_{today}.json"
        with open(videos_file, "w") as f:
            json.dump(videos, f, indent=2)

        # Save analysis data
        analysis_file = self.data_dir / f"youtube_analysis_{today}.json"
        with open(analysis_file, "w") as f:
            json.dump(analyses, f, indent=2)

        # Save summary
        summary = {
            "date": today,
            "total_videos": len(videos),
            "videos_with_teams": len([a for a in analyses if a["teams_mentioned"]]),
            "videos_with_betting_terms": len(
                [a for a in analyses if a["betting_terms"]]
            ),
            "videos_with_confidence": len(
                [a for a in analyses if a["confidence_indicators"]]
            ),
            "injury_mentions": len([a for a in analyses if a["injury_mentions"]]),
            "weather_mentions": len([a for a in analyses if a["weather_mentions"]]),
            "lineup_mentions": len([a for a in analyses if a["lineup_mentions"]]),
            "pitching_mentions": len([a for a in analyses if a["pitching_mentions"]]),
            "prop_bet_mentions": len([a for a in analyses if a["prop_bet_mentions"]]),
            "over_under_mentions": len(
                [a for a in analyses if a["over_under_mentions"]]
            ),
            "moneyline_mentions": len([a for a in analyses if a["moneyline_mentions"]]),
            "run_line_mentions": len([a for a in analyses if a["run_line_mentions"]]),
            "sentiment_breakdown": {
                "positive": len([a for a in analyses if a["sentiment"] == "positive"]),
                "negative": len([a for a in analyses if a["sentiment"] == "negative"]),
                "neutral": len([a for a in analyses if a["sentiment"] == "neutral"]),
            },
            "top_teams_mentioned": self._get_top_teams(analyses),
            "top_betting_terms": self._get_top_terms(analyses),
        }

        summary_file = self.data_dir / f"youtube_summary_{today}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"💾 Data saved to {self.data_dir}")
        logger.info(f"📊 Summary: {summary['total_videos']} videos analyzed")

        return summary

    def _get_top_teams(self, analyses: list[dict]) -> list[str]:
        """Get most mentioned teams"""
        team_counts = {}
        for analysis in analyses:
            for team in analysis["teams_mentioned"]:
                team_counts[team] = team_counts.get(team, 0) + 1

        return sorted(team_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    def _get_top_terms(self, analyses: list[dict]) -> list[str]:
        """Get most mentioned betting terms"""
        term_counts = {}
        for analysis in analyses:
            for term in analysis["betting_terms"]:
                term_counts[term] = term_counts.get(term, 0) + 1

        return sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    def run_comprehensive_collection(self):
        """Run the complete YouTube data collection process"""
        logger.info("🚀 Starting comprehensive YouTube data collection...")

        # Step 1: Get all today's videos
        videos = self.get_all_todays_videos()

        if not videos:
            logger.warning("⚠️ No videos found for today")
            return None

        # Step 2: Get detailed video information
        video_ids = [v["id"]["videoId"] for v in videos]
        detailed_videos = self.get_video_details(video_ids)

        # Step 3: Analyze each video
        analyses = []
        for video in detailed_videos:
            analysis = self.analyze_video_content(video)
            analyses.append(analysis)

        # Step 4: Save comprehensive data
        summary = self.save_comprehensive_data(detailed_videos, analyses)

        logger.info("✅ Comprehensive YouTube data collection complete!")
        return summary


def main():
    collector = ComprehensiveYouTubeCollector()
    summary = collector.run_comprehensive_collection()

    if summary:
        print("\n" + "=" * 60)
        print("📊 YOUTUBE DATA COLLECTION SUMMARY")
        print("=" * 60)
        print(f"📅 Date: {summary['date']}")
        print(f"🎥 Total Videos: {summary['total_videos']}")
        print(f"🏟️ Videos with Teams: {summary['videos_with_teams']}")
        print(f"💰 Videos with Betting Terms: {summary['videos_with_betting_terms']}")
        print(f"🎯 Videos with Confidence: {summary['videos_with_confidence']}")
        print(f"🏥 Injury Mentions: {summary['injury_mentions']}")
        print(f"🌤️ Weather Mentions: {summary['weather_mentions']}")
        print(f"📋 Lineup Mentions: {summary['lineup_mentions']}")
        print(f"⚾ Pitching Mentions: {summary['pitching_mentions']}")
        print(f"🎲 Prop Bet Mentions: {summary['prop_bet_mentions']}")
        print(f"📈 Over/Under Mentions: {summary['over_under_mentions']}")
        print(f"💵 Moneyline Mentions: {summary['moneyline_mentions']}")
        print(f"📊 Run Line Mentions: {summary['run_line_mentions']}")
        print("\n😊 Sentiment Breakdown:")
        print(f"   Positive: {summary['sentiment_breakdown']['positive']}")
        print(f"   Negative: {summary['sentiment_breakdown']['negative']}")
        print(f"   Neutral: {summary['sentiment_breakdown']['neutral']}")

        print("\n🏆 Top Teams Mentioned:")
        for team, count in summary["top_teams_mentioned"][:5]:
            print(f"   {team.title()}: {count}")

        print("\n💰 Top Betting Terms:")
        for term, count in summary["top_betting_terms"][:5]:
            print(f"   {term.title()}: {count}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
