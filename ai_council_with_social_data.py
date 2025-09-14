#!/usr/bin/env python3
"""
AI Council Integration with Social Data Analysis
Combines your existing multimodal AI council with YouTube and Reddit data
"""

import asyncio
import json
import logging
import os
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


class SocialDataCollector:
    """Collects data from YouTube and Reddit for AI analysis"""

    def __init__(self):
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

    async def collect_youtube_data(self) -> dict:
        """Collect YouTube data for MLB analysis"""
        logger.info("📺 Collecting YouTube data...")

        search_terms = [
            "MLB daily picks today",
            "MLB injury updates today",
            "MLB lineup changes today",
            "MLB betting picks today",
        ]

        all_videos = []
        for term in search_terms:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": term,
                "maxResults": 5,
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
                    logger.info(f"Found {len(videos)} videos for '{term}'")
            except Exception as e:
                logger.error(f"Error collecting YouTube data for '{term}': {e}")

        return {
            "videos": all_videos,
            "total_videos": len(all_videos),
            "search_terms": search_terms,
        }

    async def collect_reddit_data(self) -> dict:
        """Collect Reddit data for MLB analysis"""
        logger.info("📱 Collecting Reddit data...")

        subreddits = ["baseball", "sportsbook", "fantasybaseball"]
        all_posts = []

        for subreddit in subreddits:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            headers = {"User-Agent": "MLB-Betting-System/1.0"}

            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    posts = response.json().get("data", {}).get("children", [])
                    all_posts.extend(posts[:10])  # Top 10 posts from each subreddit
                    logger.info(f"Found {len(posts[:10])} posts from r/{subreddit}")
            except Exception as e:
                logger.error(f"Error collecting Reddit data from r/{subreddit}: {e}")

        return {
            "posts": all_posts,
            "total_posts": len(all_posts),
            "subreddits": subreddits,
        }


class AICouncilAnalyzer:
    """Your existing AI council with enhanced social data analysis"""

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

    async def analyze_with_openai(self, social_data: dict, odds_data: dict) -> dict:
        """Analyze with OpenAI GPT-4o"""
        logger.info("🤖 OpenAI GPT-4o analyzing...")

        prompt = self._create_analysis_prompt(social_data, odds_data)

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert MLB betting analyst. Analyze the provided data and give comprehensive betting recommendations with reasoning.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1500,
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return {
                    "source": "OpenAI GPT-4o",
                    "analysis": result["choices"][0]["message"]["content"],
                    "confidence": 0.85,
                }
        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")

        return {
            "source": "OpenAI GPT-4o",
            "analysis": "Analysis failed",
            "confidence": 0.0,
        }

    async def analyze_with_perplexity(self, social_data: dict, odds_data: dict) -> dict:
        """Analyze with Perplexity Pro"""
        logger.info("🧠 Perplexity Pro analyzing...")

        prompt = self._create_analysis_prompt(social_data, odds_data)

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.perplexity_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return {
                    "source": "Perplexity Pro",
                    "analysis": result["choices"][0]["message"]["content"],
                    "confidence": 0.80,
                }
        except Exception as e:
            logger.error(f"Perplexity analysis error: {e}")

        return {
            "source": "Perplexity Pro",
            "analysis": "Analysis failed",
            "confidence": 0.0,
        }

    def _create_analysis_prompt(self, social_data: dict, odds_data: dict) -> str:
        """Create comprehensive analysis prompt"""
        return f"""
Analyze this comprehensive MLB betting data:

YOUTUBE DATA:
- Total videos analyzed: {social_data.get('youtube', {}).get('total_videos', 0)}
- Search terms: {social_data.get('youtube', {}).get('search_terms', [])}
- Recent videos: {len(social_data.get('youtube', {}).get('videos', []))}

REDDIT DATA:
- Total posts analyzed: {social_data.get('reddit', {}).get('total_posts', 0)}
- Subreddits: {social_data.get('reddit', {}).get('subreddits', [])}

ODDS DATA:
- Games available: {len(odds_data.get('games', []))}

Please provide:
1. Top 3 betting recommendations with confidence levels
2. Injury impact analysis from social media
3. Public sentiment vs sharp money analysis
4. Risk assessment for each pick
5. Overall market sentiment and strategy
6. Specific insights from YouTube and Reddit data

Focus on identifying value bets where public sentiment differs from odds.
"""


class ComprehensiveBettingSystem:
    """Main system that integrates everything"""

    def __init__(self):
        self.social_collector = SocialDataCollector()
        self.ai_council = AICouncilAnalyzer()

    async def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline"""
        logger.info("🚀 Starting Comprehensive AI Council Analysis")

        # Step 1: Collect social data
        youtube_data = await self.social_collector.collect_youtube_data()
        reddit_data = await self.social_collector.collect_reddit_data()

        # Step 2: Get sample odds data (you can integrate your existing odds fetcher)
        odds_data = {
            "games": [
                {
                    "id": "game1",
                    "home_team": "Yankees",
                    "away_team": "Red Sox",
                    "odds": 1.85,
                },
                {
                    "id": "game2",
                    "home_team": "Dodgers",
                    "away_team": "Giants",
                    "odds": 2.10,
                },
            ]
        }

        # Step 3: Run AI council analysis
        social_data = {"youtube": youtube_data, "reddit": reddit_data}

        openai_analysis = await self.ai_council.analyze_with_openai(
            social_data, odds_data
        )
        perplexity_analysis = await self.ai_council.analyze_with_perplexity(
            social_data, odds_data
        )

        # Step 4: Combine and display results
        results = {
            "timestamp": datetime.now().isoformat(),
            "social_data_summary": {
                "youtube_videos": youtube_data["total_videos"],
                "reddit_posts": reddit_data["total_posts"],
            },
            "ai_analyses": [openai_analysis, perplexity_analysis],
            "consensus_recommendations": self._generate_consensus(
                [openai_analysis, perplexity_analysis]
            ),
        }

        # Step 5: Display results
        self._display_results(results)

        return results

    def _generate_consensus(self, analyses: list[dict]) -> dict:
        """Generate consensus from multiple AI analyses"""
        # Simple consensus - you can make this more sophisticated
        valid_analyses = [a for a in analyses if a["confidence"] > 0.5]
        if not valid_analyses:
            return {"recommendations": [], "confidence": 0.0}

        avg_confidence = sum(a["confidence"] for a in valid_analyses) / len(
            valid_analyses
        )

        return {
            "recommendations": [a["analysis"] for a in valid_analyses],
            "confidence": avg_confidence,
            "sources": [a["source"] for a in valid_analyses],
        }

    def _display_results(self, results: dict):
        """Display comprehensive results"""
        logger.info("📊 COMPREHENSIVE ANALYSIS RESULTS")
        logger.info("=" * 50)

        logger.info(
            f"📺 YouTube Videos Analyzed: {results['social_data_summary']['youtube_videos']}"
        )
        logger.info(
            f"📱 Reddit Posts Analyzed: {results['social_data_summary']['reddit_posts']}"
        )

        logger.info("\n🤖 AI COUNCIL ANALYSES:")
        for analysis in results["ai_analyses"]:
            logger.info(
                f"  {analysis['source']}: {analysis['confidence']:.1%} confidence"
            )
            logger.info(f"  Analysis: {analysis['analysis'][:200]}...")

        logger.info(
            f"\n🎯 CONSENSUS CONFIDENCE: {results['consensus_recommendations']['confidence']:.1%}"
        )
        logger.info(
            f"📋 AI SOURCES: {', '.join(results['consensus_recommendations']['sources'])}"
        )


async def main():
    """Main function"""
    system = ComprehensiveBettingSystem()
    results = await system.run_comprehensive_analysis()

    # Save results
    with open("ai_council_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("✅ Analysis complete! Results saved to ai_council_results.json")


if __name__ == "__main__":
    asyncio.run(main())
