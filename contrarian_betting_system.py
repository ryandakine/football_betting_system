#!/usr/bin/env python3
"""
Contrarian Betting System
YouTube = Public Sentiment (often fade)
Other Sources = Sharp Money Analysis
AI Council = Smart Contrarian Decisions
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests


# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip loading
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


class ContrarianBettingSystem:
    """Contrarian betting system that analyzes public vs sharp money"""

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.grok_api_key = os.getenv("GROK_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.data_dir = Path("data")
        self.results_dir = Path("analysis")
        self.results_dir.mkdir(exist_ok=True)

    def load_youtube_public_sentiment(self) -> dict:
        """Load YouTube data as PUBLIC SENTIMENT (what to potentially fade)"""
        today = datetime.now().strftime("%Y-%m-%d")

        summary_file = Path("data/youtube_analysis") / f"youtube_summary_{today}.json"
        if not summary_file.exists():
            logger.warning("⚠️ No YouTube data found - using sample data")
            return self._create_sample_public_data()

        with open(summary_file) as f:
            summary = json.load(f)

        # Analyze as PUBLIC SENTIMENT
        public_sentiment = {
            "source": "YouTube (Public Sentiment)",
            "date": today,
            "total_videos": summary["total_videos"],
            "public_favorites": summary["top_teams_mentioned"][
                :5
            ],  # Most popular teams
            "public_betting_trends": summary["top_betting_terms"][
                :5
            ],  # Most popular bets
            "sentiment_breakdown": summary["sentiment_breakdown"],
            "confidence_indicators": summary["videos_with_confidence"],
            "injury_hype": summary[
                "injury_mentions"
            ],  # Public overreacting to injuries
            "weather_hype": summary[
                "weather_mentions"
            ],  # Public overreacting to weather
            "over_hype": summary["over_under_mentions"],  # Public loves overs
            "moneyline_hype": summary["moneyline_mentions"],  # Public loves favorites
        }

        return public_sentiment

    def _create_sample_public_data(self) -> dict:
        """Create sample public sentiment data for testing"""
        return {
            "source": "YouTube (Public Sentiment)",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_videos": 410,
            "public_favorites": [["yankees", 68], ["dodgers", 67], ["mets", 59]],
            "public_betting_trends": [["over", 177], ["best bet", 131], ["prop", 120]],
            "sentiment_breakdown": {"positive": 259, "negative": 9, "neutral": 142},
            "confidence_indicators": 114,
            "injury_hype": 311,
            "weather_hype": 36,
            "over_hype": 232,
            "moneyline_hype": 387,
        }

    async def collect_reddit_sharp_analysis(self) -> dict:
        """Collect Reddit data for sharp money analysis"""
        logger.info("📱 Collecting Reddit sharp money analysis...")

        subreddits = ["sportsbook", "baseball", "fantasybaseball", "dfsports"]
        sharp_analysis = {
            "source": "Reddit (Sharp Money)",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "subreddits_analyzed": subreddits,
            "sharp_contrarian_picks": [],
            "fade_opportunities": [],
            "value_plays": [],
            "public_fade_signals": [],
        }

        # Simulate Reddit data collection (replace with real Reddit API)
        sharp_analysis["sharp_contrarian_picks"] = [
            "Fade Yankees - public overvaluing recent performance",
            "Under in Dodgers game - public expecting high scoring",
            "Tigers +1.5 - public sleeping on them",
            "Mets under 8.5 - public not considering pitching upgrades",
        ]

        sharp_analysis["fade_opportunities"] = [
            "Yankees moneyline - too much public money",
            "Over in Phillies game - public overreacting to recent scoring",
            "Dodgers run line - public expecting blowout",
        ]

        sharp_analysis["value_plays"] = [
            "Mariners +120 - sharp money moving this way",
            "Reds under 9.5 - value in low scoring game",
            "Tigers team total under 4.5 - public overvaluing offense",
        ]

        return sharp_analysis

    async def collect_twitter_sharp_signals(self) -> dict:
        """Collect Twitter data for sharp money signals"""
        logger.info("🐦 Collecting Twitter sharp money signals...")

        # Simulate Twitter data collection (replace with real Twitter API)
        twitter_signals = {
            "source": "Twitter (Sharp Money)",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "sharp_money_movements": [
                "Sharp money hitting Mariners hard",
                "Line movement: Yankees line dropping despite public money",
                "Sharp action on under in Mets game",
                "Public heavy on Dodgers, sharp money on Giants +1.5",
            ],
            "line_movements": [
                "Yankees -150 → -140 (sharp money fading)",
                "Mariners +110 → +120 (sharp money backing)",
                "Mets total 8.5 → 8.0 (sharp money on under)",
            ],
            "contrarian_signals": [
                "Public 85% on Yankees, sharp money 65% on opponent",
                "Public 78% on over, sharp money 45% on under",
                "Public 92% on Dodgers, sharp money 40% on Giants",
            ],
        }

        return twitter_signals

    async def collect_news_analysis(self) -> dict:
        """Collect news data for fundamental analysis"""
        logger.info("📰 Collecting news analysis...")

        # Simulate news data collection (replace with real news API)
        news_analysis = {
            "source": "News (Fundamental Analysis)",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "injury_impacts": [
                "Yankees: Judge day-to-day, not as serious as public thinks",
                "Dodgers: Kershaw returning, public overvaluing impact",
                "Mets: Soto trade actually improves bullpen significantly",
            ],
            "weather_impacts": [
                "Wind blowing out in Yankee Stadium - public overreacting",
                "Clear skies in LA - no weather advantage for Dodgers",
                "Humid conditions in Seattle - favors pitchers",
            ],
            "lineup_changes": [
                "Yankees: Judge likely to play, public overreacting to injury",
                "Mets: New bullpen configuration actually stronger",
                "Mariners: Trade deadline additions improving lineup",
            ],
            "fundamental_factors": [
                "Yankees: Recent performance unsustainable, regression coming",
                "Mariners: Underrated pitching staff, public not noticing",
                "Mets: Bullpen upgrades not priced into lines",
            ],
        }

        return news_analysis

    async def analyze_with_grok(
        self, public_data: dict, sharp_data: dict, news_data: dict
    ) -> dict:
        """Analyze with Grok for contrarian insights"""
        logger.info("🤖 Grok analyzing contrarian opportunities...")

        prompt = f"""
You are a contrarian betting analyst. Your job is to identify when to FADE the public (YouTube sentiment) vs when to FOLLOW sharp money.

## PUBLIC SENTIMENT (YouTube - often fade):
- Most Popular Teams: {public_data['public_favorites']}
- Public Betting Trends: {public_data['public_betting_trends']}
- Public Sentiment: {public_data['sentiment_breakdown']}
- Public Confidence: {public_data['confidence_indicators']} videos with confidence indicators
- Public Injury Hype: {public_data['injury_hype']} videos mentioning injuries
- Public Over Hype: {public_data['over_hype']} videos mentioning overs

## SHARP MONEY SIGNALS (Reddit/Twitter):
- Sharp Contrarian Picks: {sharp_data['sharp_contrarian_picks']}
- Fade Opportunities: {sharp_data['fade_opportunities']}
- Value Plays: {sharp_data['value_plays']}

## NEWS ANALYSIS (Fundamental):
- Injury Impacts: {news_data['injury_impacts']}
- Weather Impacts: {news_data['weather_impacts']}
- Fundamental Factors: {news_data['fundamental_factors']}

## ANALYSIS REQUEST:
Based on this data, provide:

1. **TOP 5 CONTRARIAN BETTING OPPORTUNITIES** - where to fade public sentiment
2. **SHARP MONEY FOLLOWS** - where sharp money aligns with good value
3. **PUBLIC FADE SIGNALS** - specific teams/bets the public is overvaluing
4. **VALUE OPPORTUNITIES** - where lines are mispriced
5. **CONFIDENCE LEVELS** for each recommendation (1-10)
6. **REASONING** - why each bet is contrarian or follows sharp money

Focus on identifying when public sentiment is wrong and sharp money is right.
"""

        try:
            # Grok API call (simulated for now)
            # Replace with actual Grok API integration
            analysis = f"""
## GROK CONTRARIAN ANALYSIS

### TOP 5 CONTRARIAN OPPORTUNITIES:

1. **FADE Yankees Moneyline** (Confidence: 8/10)
   - Public heavily backing Yankees (68 mentions)
   - Sharp money fading, line moving from -150 to -140
   - News: Judge injury not as serious as public thinks
   - RECOMMENDATION: Take opponent +140

2. **FADE Dodgers Run Line** (Confidence: 7/10)
   - Public expecting blowout (67 mentions, high confidence)
   - Sharp money on Giants +1.5
   - RECOMMENDATION: Giants +1.5

3. **FADE Over in Phillies Game** (Confidence: 7/10)
   - Public loves overs (177 mentions)
   - Sharp money on under
   - RECOMMENDATION: Under 9.5

4. **FOLLOW Sharp Money on Mariners** (Confidence: 9/10)
   - Public not heavily backing Mariners
   - Sharp money hitting hard, line moving to +120
   - RECOMMENDATION: Mariners +120

5. **FADE Public Injury Hype** (Confidence: 6/10)
   - Public overreacting to injuries (311 mentions)
   - News shows injuries not as serious
   - RECOMMENDATION: Fade teams with injury hype

### SHARP MONEY FOLLOWS:
- Mariners +120 (sharp money backing)
- Mets under 8.5 (sharp money on under)
- Tigers +1.5 (sharp money on underdog)

### PUBLIC FADE SIGNALS:
- Yankees moneyline (too much public money)
- Dodgers run line (public expecting blowout)
- Overs in high-profile games (public loves scoring)

### VALUE OPPORTUNITIES:
- Giants +1.5 (line mispriced due to public bias)
- Mariners +120 (sharp money moving line)
- Under bets in public games (contrarian value)
"""

            return {
                "source": "Grok (Contrarian Analysis)",
                "analysis": analysis,
                "confidence": 0.85,
            }

        except Exception as e:
            logger.error(f"❌ Grok analysis error: {e}")
            return {
                "source": "Grok",
                "analysis": f"Analysis failed: {e}",
                "confidence": 0.0,
            }

    async def analyze_with_gemini(
        self, public_data: dict, sharp_data: dict, news_data: dict
    ) -> dict:
        """Analyze with Gemini for alternative perspective"""
        logger.info("🤖 Gemini analyzing alternative perspective...")

        prompt = f"""
You are a quantitative betting analyst. Analyze the data and provide statistical insights.

## DATA SUMMARY:
- Public Favorites: {public_data['public_favorites']}
- Sharp Signals: {sharp_data['sharp_contrarian_picks']}
- News Factors: {news_data['fundamental_factors']}

Provide statistical analysis and quantitative betting recommendations.
"""

        try:
            # Gemini API call (simulated for now)
            # Replace with actual Gemini API integration
            analysis = f"""
## GEMINI QUANTITATIVE ANALYSIS

### STATISTICAL INSIGHTS:

1. **Public Sentiment vs Performance Correlation**
   - Teams with high public backing (Yankees, Dodgers) often underperform
   - Historical data shows 65% of public favorites fail to cover
   - RECOMMENDATION: Systematic fade of public favorites

2. **Sharp Money Success Rate**
   - Sharp money has 72% success rate when opposing public
   - Line movements of 10+ points indicate sharp action
   - RECOMMENDATION: Follow sharp money movements

3. **Injury Impact Analysis**
   - Public overvalues injury news by 23%
   - Teams with "questionable" players actually perform 8% better
   - RECOMMENDATION: Fade injury hype

4. **Weather Factor Analysis**
   - Public overreacts to weather by 15%
   - Wind factors priced in at 2x actual impact
   - RECOMMENDATION: Ignore weather hype

5. **Over/Under Efficiency**
   - Public over-bets overs by 18%
   - Sharp money on unders has 68% success rate
   - RECOMMENDATION: Systematic under betting

### QUANTITATIVE RECOMMENDATIONS:

1. **Yankees Fade** (Confidence: 8.5/10)
   - Statistical probability: 67% chance of failure
   - RECOMMENDATION: Opponent +140

2. **Mariners Sharp Follow** (Confidence: 9/10)
   - Sharp money success rate: 78%
   - RECOMMENDATION: Mariners +120

3. **Systematic Under Betting** (Confidence: 7.5/10)
   - Historical success rate: 68%
   - RECOMMENDATION: Under in public games

4. **Injury Hype Fade** (Confidence: 7/10)
   - Statistical edge: 23% overvaluation
   - RECOMMENDATION: Fade injury-affected teams

5. **Line Movement Follow** (Confidence: 8/10)
   - Sharp money indicator: 72% accuracy
   - RECOMMENDATION: Follow significant line movements
"""

            return {
                "source": "Gemini (Quantitative Analysis)",
                "analysis": analysis,
                "confidence": 0.82,
            }

        except Exception as e:
            logger.error(f"❌ Gemini analysis error: {e}")
            return {
                "source": "Gemini",
                "analysis": f"Analysis failed: {e}",
                "confidence": 0.0,
            }

    def generate_contrarian_consensus(
        self, grok_analysis: dict, gemini_analysis: dict
    ) -> dict:
        """Generate contrarian consensus from Grok and Gemini"""
        analyses = [grok_analysis, gemini_analysis]
        successful_analyses = [a for a in analyses if a["confidence"] > 0]

        if not successful_analyses:
            return {
                "consensus": "No successful analyses",
                "confidence": 0.0,
                "sources": [],
            }

        # Calculate weighted average confidence
        total_confidence = sum(a["confidence"] for a in successful_analyses)
        avg_confidence = total_confidence / len(successful_analyses)

        # Combine analyses
        combined_analysis = "\n\n".join(
            [
                f"=== {a['source']} (Confidence: {a['confidence']:.1%}) ===\n{a['analysis']}"
                for a in successful_analyses
            ]
        )

        return {
            "consensus": combined_analysis,
            "confidence": avg_confidence,
            "sources": [a["source"] for a in successful_analyses],
        }

    def save_contrarian_results(
        self,
        public_data: dict,
        sharp_data: dict,
        news_data: dict,
        grok_analysis: dict,
        gemini_analysis: dict,
        consensus: dict,
    ):
        """Save contrarian analysis results"""
        today = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        results = {
            "timestamp": datetime.now().isoformat(),
            "public_sentiment": public_data,
            "sharp_money_signals": sharp_data,
            "news_analysis": news_data,
            "ai_analyses": [grok_analysis, gemini_analysis],
            "contrarian_consensus": consensus,
        }

        results_file = self.results_dir / f"contrarian_betting_results_{today}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 Contrarian results saved to {results_file}")
        return results_file

    async def run_contrarian_analysis(self):
        """Run the complete contrarian analysis"""
        logger.info("🚀 Starting Contrarian Betting Analysis...")

        # Step 1: Load public sentiment (YouTube)
        public_data = self.load_youtube_public_sentiment()
        logger.info(
            f"📊 Public sentiment loaded: {public_data['total_videos']} videos analyzed"
        )

        # Step 2: Collect sharp money signals
        sharp_data = await self.collect_reddit_sharp_analysis()
        twitter_data = await self.collect_twitter_sharp_signals()
        news_data = await self.collect_news_analysis()

        # Combine sharp money data
        combined_sharp_data = {
            "reddit": sharp_data,
            "twitter": twitter_data,
            "news": news_data,
        }

        # Step 3: AI Analysis
        grok_analysis = await self.analyze_with_grok(public_data, sharp_data, news_data)
        gemini_analysis = await self.analyze_with_gemini(
            public_data, sharp_data, news_data
        )

        # Step 4: Generate consensus
        consensus = self.generate_contrarian_consensus(grok_analysis, gemini_analysis)

        # Step 5: Save results
        results_file = self.save_contrarian_results(
            public_data,
            combined_sharp_data,
            news_data,
            grok_analysis,
            gemini_analysis,
            consensus,
        )

        # Step 6: Display summary
        self.display_contrarian_summary(public_data, consensus)

        return results_file

    def display_contrarian_summary(self, public_data: dict, consensus: dict):
        """Display contrarian analysis summary"""
        print("\n" + "=" * 80)
        print("🎯 CONTRARIAN BETTING ANALYSIS RESULTS")
        print("=" * 80)
        print(f"📅 Date: {public_data['date']}")
        print(f"📺 Public Sentiment Videos: {public_data['total_videos']}")
        print(
            f"🏟️ Public Favorites: {[team[0].title() for team in public_data['public_favorites'][:3]]}"
        )
        print(
            f"💰 Public Betting Trends: {[term[0].title() for term in public_data['public_betting_trends'][:3]]}"
        )
        print(
            f"😊 Public Sentiment: {public_data['sentiment_breakdown']['positive']} positive, {public_data['sentiment_breakdown']['negative']} negative"
        )

        print(f"\n🤖 CONTRARIAN AI ANALYSES:")
        for source in consensus["sources"]:
            print(f"  {source}")

        print(f"\n🎯 CONTRARIAN CONFIDENCE: {consensus['confidence']:.1%}")
        print(f"📋 STRATEGY: Fade public sentiment, follow sharp money")

        print("\n" + "=" * 80)


async def main():
    system = ContrarianBettingSystem()
    results_file = await system.run_contrarian_analysis()

    if results_file:
        print(f"\n✅ Contrarian analysis complete! Results saved to: {results_file}")
    else:
        print("\n❌ Analysis failed")


if __name__ == "__main__":
    asyncio.run(main())
