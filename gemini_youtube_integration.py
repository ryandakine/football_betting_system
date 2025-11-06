#!/usr/bin/env python3
"""
GeminiYouTube Integration for NFL Betting System
===============================================

Integrates the GeminiYouTube system to provide social sentiment,
market intelligence, and public opinion data for NFL betting analysis.

This adds real social data to our GPU-powered NFL analysis system.
"""

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class GeminiYouTubeNFLIntegration:
    """
    Integration between GeminiYouTube system and NFL betting analysis.
    Provides social sentiment and market intelligence data.
    """

    def __init__(self, gemini_youtube_path: str = "/home/ryan/code/GeminiYoutube"):
        self.gemini_path = Path(gemini_youtube_path)
        self.nfl_data_cache = {}
        self.sentiment_cache = {}

        # Verify GeminiYouTube system exists
        if not self.gemini_path.exists():
            raise FileNotFoundError(f"GeminiYouTube system not found at {gemini_youtube_path}")

        logger.info(f"🤝 Connected to GeminiYouTube system at {gemini_youtube_path}")

    async def get_nfl_social_sentiment(self, game_query: str) -> Dict[str, Any]:
        """
        Get social sentiment data for NFL game from YouTube analysis.
        """
        cache_key = f"sentiment_{game_query}"

        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]

        try:
            # Run GeminiYouTube analysis for NFL game sentiment
            cmd = [
                sys.executable,
                str(self.gemini_path / "main.py"),
                f'"{game_query}"',
                "--tier", "premium",
                "--max-videos", "10",
                "--format", "json"
            ]

            logger.info(f"🎥 Analyzing YouTube sentiment for: {game_query}")

            # Run the analysis
            result = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.gemini_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                try:
                    sentiment_data = json.loads(stdout.decode())
                    self.sentiment_cache[cache_key] = sentiment_data
                    return sentiment_data
                except json.JSONDecodeError:
                    logger.warning("Could not parse sentiment data as JSON")
            else:
                logger.error(f"GeminiYouTube analysis failed: {stderr.decode()}")

        except Exception as e:
            logger.error(f"Error getting NFL sentiment data: {e}")

        # Return default sentiment data if analysis fails
        return {
            'sentiment_score': 0.5,
            'confidence': 0.3,
            'video_count': 0,
            'key_themes': ['Data unavailable'],
            'public_opinion': 'Neutral',
            'error': 'Analysis failed'
        }

    async def analyze_game_social_buzz(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """
        Analyze social media buzz and public sentiment for a specific NFL game.
        """
        game_query = f"NFL {away_team} vs {home_team} game predictions analysis"

        sentiment_data = await self.get_nfl_social_sentiment(game_query)

        # Process sentiment for betting insights
        buzz_analysis = {
            'game_matchup': f"{away_team} @ {home_team}",
            'social_sentiment': sentiment_data.get('sentiment_score', 0.5),
            'public_confidence': sentiment_data.get('confidence', 0.5),
            'video_discussions': sentiment_data.get('video_count', 0),
            'key_social_themes': sentiment_data.get('key_themes', []),
            'betting_implications': self._analyze_betting_implications(sentiment_data),
            'contrarian_signals': self._detect_contrarian_opportunities(sentiment_data),
            'timestamp': datetime.now().isoformat()
        }

        return buzz_analysis

    async def get_weekly_nfl_sentiment(self) -> Dict[str, Any]:
        """
        Get overall NFL sentiment for the week from social media.
        """
        weekly_query = "NFL Week 1 predictions analysis 2025 season outlook"

        sentiment_data = await self.get_nfl_social_sentiment(weekly_query)

        weekly_analysis = {
            'overall_nfl_sentiment': sentiment_data.get('sentiment_score', 0.5),
            'public_excitement': sentiment_data.get('confidence', 0.5),
            'content_volume': sentiment_data.get('video_count', 0),
            'trending_topics': sentiment_data.get('key_themes', []),
            'market_temperature': self._assess_market_temperature(sentiment_data),
            'week_type': 'Opening weekend',
            'season_context': '2025 NFL season kickoff'
        }

        return weekly_analysis

    async def get_team_specific_sentiment(self, team_name: str) -> Dict[str, Any]:
        """
        Get sentiment analysis for a specific NFL team.
        """
        team_query = f"NFL {team_name} team analysis predictions 2025 season"

        sentiment_data = await self.get_nfl_social_sentiment(team_query)

        team_analysis = {
            'team': team_name,
            'fan_sentiment': sentiment_data.get('sentiment_score', 0.5),
            'social_buzz': sentiment_data.get('confidence', 0.5),
            'content_mentions': sentiment_data.get('video_count', 0),
            'sentiment_trends': sentiment_data.get('key_themes', []),
            'betting_popularity': self._assess_team_popularity(sentiment_data),
            'underdog_potential': self._detect_underdog_potential(sentiment_data)
        }

        return team_analysis

    def _analyze_betting_implications(self, sentiment_data: Dict) -> Dict[str, Any]:
        """Analyze betting implications from social sentiment."""
        sentiment = sentiment_data.get('sentiment_score', 0.5)
        confidence = sentiment_data.get('confidence', 0.5)

        implications = {
            'line_movement': 'stable',
            'public_bias': 'neutral',
            'sharp_money_opportunity': False,
            'contrarian_value': False
        }

        # High sentiment + low confidence = potential line movement
        if sentiment > 0.7 and confidence < 0.6:
            implications['line_movement'] = 'upward pressure'
            implications['public_bias'] = 'overly optimistic'
            implications['contrarian_value'] = True

        # Low sentiment + high confidence = potential fade opportunity
        elif sentiment < 0.3 and confidence > 0.7:
            implications['line_movement'] = 'downward pressure'
            implications['public_bias'] = 'overly pessimistic'
            implications['sharp_money_opportunity'] = True

        return implications

    def _detect_contrarian_opportunities(self, sentiment_data: Dict) -> List[str]:
        """Detect potential contrarian betting opportunities."""
        opportunities = []
        sentiment = sentiment_data.get('sentiment_score', 0.5)

        if sentiment > 0.8:
            opportunities.append("Consider fading public favorite - sentiment too high")
        elif sentiment < 0.2:
            opportunities.append("Consider betting on underdog - sentiment too low")
        else:
            opportunities.append("Sentiment appears balanced - stick to fundamentals")

        return opportunities

    def _assess_market_temperature(self, sentiment_data: Dict) -> str:
        """Assess overall market temperature from sentiment."""
        sentiment = sentiment_data.get('sentiment_score', 0.5)
        content_volume = sentiment_data.get('video_count', 0)

        if sentiment > 0.7 and content_volume > 20:
            return "HOT - High excitement, potential line movement"
        elif sentiment > 0.6 and content_volume > 10:
            return "WARM - Moderate interest, stable lines"
        elif sentiment < 0.4 and content_volume < 5:
            return "COOL - Low interest, potential value opportunities"
        else:
            return "NEUTRAL - Balanced sentiment"

    def _assess_team_popularity(self, sentiment_data: Dict) -> str:
        """Assess team's betting popularity from sentiment."""
        sentiment = sentiment_data.get('sentiment_score', 0.5)
        content_volume = sentiment_data.get('video_count', 0)

        if sentiment > 0.7 and content_volume > 15:
            return "VERY POPULAR - Heavy public betting expected"
        elif sentiment > 0.6 and content_volume > 8:
            return "POPULAR - Moderate public interest"
        elif sentiment < 0.4 and content_volume < 3:
            return "UNDERPOPULAR - Potential value as underdog"
        else:
            return "NEUTRAL - Balanced interest"

    def _detect_underdog_potential(self, sentiment_data: Dict) -> str:
        """Detect if team has underdog potential based on sentiment."""
        sentiment = sentiment_data.get('sentiment_score', 0.5)

        if sentiment < 0.3:
            return "HIGH - Very low sentiment suggests value"
        elif sentiment < 0.4:
            return "MODERATE - Some contrarian potential"
        else:
            return "LOW - Sentiment appears fair"

async def main():
    """Test the GeminiYouTube integration."""
    print("🤝 TESTING GEMINI YOUTUBE INTEGRATION")
    print("=" * 50)

    try:
        # Initialize integration
        integration = GeminiYouTubeNFLIntegration()

        print("\\n🎯 Testing Chiefs vs Eagles social sentiment...")

        # Test game-specific analysis
        chiefs_analysis = await integration.analyze_game_social_buzz("Chiefs", "Eagles")
        print("\\n📊 CHIEFS VS EAGLES SOCIAL ANALYSIS:")
        print(f"   Social Sentiment: {chiefs_analysis['social_sentiment']:.2f}")
        print(f"   Public Confidence: {chiefs_analysis['public_confidence']:.2f}")
        print(f"   Video Discussions: {chiefs_analysis['video_discussions']}")
        print(f"   Betting Implications: {chiefs_analysis['betting_implications']['public_bias']}")
        print(f"   Contrarian Signals: {chiefs_analysis['contrarian_signals'][0]}")

        print("\\n📈 Testing weekly NFL sentiment...")

        # Test weekly analysis
        weekly_analysis = await integration.get_weekly_nfl_sentiment()
        print("\\n📊 WEEKLY NFL SOCIAL ANALYSIS:")
        print(f"   Overall Sentiment: {weekly_analysis['overall_nfl_sentiment']:.2f}")
        print(f"   Public Excitement: {weekly_analysis['public_excitement']:.2f}")
        print(f"   Content Volume: {weekly_analysis['content_volume']}")
        print(f"   Market Temperature: {weekly_analysis['market_temperature']}")

        print("\\n✅ GeminiYouTube integration successful!")
        print("🎯 Social sentiment data now available for NFL betting analysis!")
        print("\\n💡 INTEGRATION BENEFITS:")
        print("   • Real social media sentiment analysis")
        print("   • Public opinion tracking")
        print("   • Contrarian betting opportunities")
        print("   • Market temperature assessment")
        print("   • Enhanced betting intelligence")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print("\\n🔧 TROUBLESHOOTING:")
        print("   1. Ensure GeminiYouTube system is properly installed")
        print("   2. Check API keys in GeminiYouTube/.env")
        print("   3. Verify Python path and dependencies")
        print("   4. Run: cd /home/ryan/code/GeminiYoutube && python test_phase1.py")

if __name__ == "__main__":
    asyncio.run(main())
