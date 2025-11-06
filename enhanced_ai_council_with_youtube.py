#!/usr/bin/env python3
"""
Enhanced AI Council with Comprehensive YouTube Data
Uses the detailed YouTube analysis for superior betting predictions
"""

import asyncio
import json
import logging
import os
from datetime import datetime
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


class EnhancedAICouncil:
    """Enhanced AI Council using comprehensive YouTube data"""

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.grok_api_key = os.getenv("GROK_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

        self.data_dir = Path("data/youtube_analysis")
        self.results_dir = Path("analysis")
        self.results_dir.mkdir(exist_ok=True)

    def load_youtube_data(self) -> dict:
        """Load today's comprehensive YouTube data"""
        today = datetime.now().strftime("%Y-%m-%d")

        # Load summary
        summary_file = self.data_dir / f"youtube_summary_{today}.json"
        if not summary_file.exists():
            logger.error(f"❌ No YouTube data found for {today}")
            return None

        with open(summary_file) as f:
            summary = json.load(f)

        # Load detailed analysis
        analysis_file = self.data_dir / f"youtube_analysis_{today}.json"
        with open(analysis_file) as f:
            analysis = json.load(f)

        # Load raw videos
        videos_file = self.data_dir / f"youtube_videos_{today}.json"
        with open(videos_file) as f:
            videos = json.load(f)

        return {"summary": summary, "analysis": analysis, "videos": videos}

    def create_comprehensive_prompt(self, youtube_data: dict) -> str:
        """Create a comprehensive prompt using YouTube data"""
        summary = youtube_data["summary"]
        analysis = youtube_data["analysis"]

        # Get top videos with high confidence
        high_confidence_videos = [v for v in analysis if v["confidence_indicators"]]
        top_videos = sorted(
            high_confidence_videos,
            key=lambda x: int(x.get("view_count", 0)),
            reverse=True,
        )[:20]

        # Get injury mentions
        injury_videos = [v for v in analysis if v["injury_mentions"]]

        # Get weather mentions
        weather_videos = [v for v in analysis if v["weather_mentions"]]

        # Get pitching mentions
        pitching_videos = [v for v in analysis if v["pitching_mentions"]]

        prompt = f"""
You are an expert MLB betting analyst with access to comprehensive social media data from today ({summary['date']}).

## YOUTUBE DATA SUMMARY:
- Total Videos Analyzed: {summary['total_videos']}
- Videos with Team Mentions: {summary['videos_with_teams']}
- Videos with Betting Terms: {summary['videos_with_betting_terms']}
- Videos with Confidence Indicators: {summary['videos_with_confidence']}
- Injury Mentions: {summary['injury_mentions']}
- Weather Mentions: {summary['weather_mentions']}
- Lineup Mentions: {summary['lineup_mentions']}
- Pitching Mentions: {summary['pitching_mentions']}
- Prop Bet Mentions: {summary['prop_bet_mentions']}
- Over/Under Mentions: {summary['over_under_mentions']}
- Moneyline Mentions: {summary['moneyline_mentions']}
- Run Line Mentions: {summary['run_line_mentions']}

## SENTIMENT BREAKDOWN:
- Positive: {summary['sentiment_breakdown']['positive']}
- Negative: {summary['sentiment_breakdown']['negative']}
- Neutral: {summary['sentiment_breakdown']['neutral']}

## TOP TEAMS MENTIONED:
{chr(10).join([f"- {team.title()}: {count} mentions" for team, count in summary['top_teams_mentioned'][:10]])}

## TOP BETTING TERMS:
{chr(10).join([f"- {term.title()}: {count} mentions" for term, count in summary['top_betting_terms'][:10]])}

## HIGH CONFIDENCE VIDEOS (Top 10):
"""

        for i, video in enumerate(top_videos[:10], 1):
            prompt += f"""
{i}. "{video['title']}" by {video['channel']}
   - Teams: {', '.join(video['teams_mentioned']) if video['teams_mentioned'] else 'None'}
   - Betting Terms: {', '.join(video['betting_terms']) if video['betting_terms'] else 'None'}
   - Confidence: {', '.join(video['confidence_indicators']) if video['confidence_indicators'] else 'None'}
   - Sentiment: {video['sentiment']}
   - Views: {video.get('view_count', 'Unknown')}
"""

        prompt += f"""

## INJURY MENTIONS ({len(injury_videos)} videos):
"""
        for video in injury_videos[:5]:
            prompt += f"- {video['title']} (Teams: {', '.join(video['teams_mentioned']) if video['teams_mentioned'] else 'None'})\n"

        prompt += f"""

## WEATHER MENTIONS ({len(weather_videos)} videos):
"""
        for video in weather_videos[:5]:
            prompt += f"- {video['title']} (Teams: {', '.join(video['teams_mentioned']) if video['teams_mentioned'] else 'None'})\n"

        prompt += f"""

## PITCHING MENTIONS ({len(pitching_videos)} videos):
"""
        for video in pitching_videos[:5]:
            prompt += f"- {video['title']} (Teams: {', '.join(video['teams_mentioned']) if video['teams_mentioned'] else 'None'})\n"

        prompt += """

## ANALYSIS REQUEST:
Based on this comprehensive YouTube data from today, provide:

1. **TOP 5 BETTING RECOMMENDATIONS** with confidence levels (1-10)
2. **INJURY IMPACT ANALYSIS** - how injuries mentioned affect today's games
3. **WEATHER IMPACT** - how weather factors mentioned affect games
4. **PITCHING ANALYSIS** - insights from pitching discussions
5. **PUBLIC SENTIMENT VS SHARP MONEY** - identify where public sentiment differs from expert analysis
6. **VALUE OPPORTUNITIES** - where the public might be wrong
7. **RISK ASSESSMENT** for each recommendation
8. **CONSENSUS ANALYSIS** - what the majority of experts are saying

Format your response with clear sections and specific betting recommendations with confidence levels.
"""

        return prompt

    async def analyze_with_openai(self, prompt: str) -> dict:
        """Analyze with OpenAI GPT-4o"""
        logger.info("🤖 OpenAI GPT-4o analyzing comprehensive YouTube data...")

        try:
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
                        "content": "You are an expert MLB betting analyst with access to comprehensive social media data. Provide detailed, actionable betting recommendations based on the data provided.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 4000,
                "temperature": 0.3,
            }

            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]

                # Extract confidence level (simple heuristic)
                confidence = 0.8
                if "high confidence" in analysis.lower() or "lock" in analysis.lower():
                    confidence = 0.9
                elif "medium confidence" in analysis.lower():
                    confidence = 0.7
                elif "low confidence" in analysis.lower():
                    confidence = 0.5

                return {
                    "source": "OpenAI GPT-4o",
                    "analysis": analysis,
                    "confidence": confidence,
                }
            else:
                logger.error(f"❌ OpenAI API error: {response.status_code}")
                return {
                    "source": "OpenAI GPT-4o",
                    "analysis": "Analysis failed",
                    "confidence": 0.0,
                }

        except Exception as e:
            logger.error(f"❌ OpenAI analysis error: {e}")
            return {
                "source": "OpenAI GPT-4o",
                "analysis": f"Analysis failed: {e}",
                "confidence": 0.0,
            }

    async def analyze_with_claude(self, prompt: str) -> dict:
        """Analyze with Claude"""
        logger.info("🧠 Claude analyzing comprehensive YouTube data...")

        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.claude_api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }

            data = {
                "model": "claude-3-5-sonnet-20240620",
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}],
            }

            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                analysis = result["content"][0]["text"]

                # Extract confidence level
                confidence = 0.8
                if "high confidence" in analysis.lower() or "lock" in analysis.lower():
                    confidence = 0.9
                elif "medium confidence" in analysis.lower():
                    confidence = 0.7
                elif "low confidence" in analysis.lower():
                    confidence = 0.5

                return {
                    "source": "Claude 3.5 Sonnet",
                    "analysis": analysis,
                    "confidence": confidence,
                }
            else:
                logger.error(f"❌ Claude API error: {response.status_code}")
                return {
                    "source": "Claude 3.5 Sonnet",
                    "analysis": "Analysis failed",
                    "confidence": 0.0,
                }

        except Exception as e:
            logger.error(f"❌ Claude analysis error: {e}")
            return {
                "source": "Claude 3.5 Sonnet",
                "analysis": f"Analysis failed: {e}",
                "confidence": 0.0,
            }

    def generate_consensus(self, analyses: list[dict]) -> dict:
        """Generate consensus from all AI analyses"""
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

    def save_results(self, youtube_data: dict, analyses: list[dict], consensus: dict):
        """Save comprehensive results"""
        today = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        results = {
            "timestamp": datetime.now().isoformat(),
            "youtube_data_summary": youtube_data["summary"],
            "ai_analyses": analyses,
            "consensus": consensus,
        }

        results_file = self.results_dir / f"enhanced_ai_council_results_{today}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 Results saved to {results_file}")
        return results_file

    async def run_enhanced_analysis(self):
        """Run the complete enhanced analysis"""
        logger.info("🚀 Starting Enhanced AI Council Analysis...")

        # Load YouTube data
        youtube_data = self.load_youtube_data()
        if not youtube_data:
            logger.error("❌ No YouTube data available")
            return None

        # Create comprehensive prompt
        prompt = self.create_comprehensive_prompt(youtube_data)

        # Run AI analyses
        analyses = []

        # OpenAI analysis
        openai_result = await self.analyze_with_openai(prompt)
        analyses.append(openai_result)

        # Claude analysis
        claude_result = await self.analyze_with_claude(prompt)
        analyses.append(claude_result)

        # Generate consensus
        consensus = self.generate_consensus(analyses)

        # Save results
        results_file = self.save_results(youtube_data, analyses, consensus)

        # Display summary
        self.display_summary(youtube_data["summary"], analyses, consensus)

        return results_file

    def display_summary(
        self, youtube_summary: dict, analyses: list[dict], consensus: dict
    ):
        """Display analysis summary"""
        print("\n" + "=" * 80)
        print("🎯 ENHANCED AI COUNCIL ANALYSIS RESULTS")
        print("=" * 80)
        print(f"📅 Date: {youtube_summary['date']}")
        print(f"🎥 YouTube Videos Analyzed: {youtube_summary['total_videos']}")
        print(f"🏟️ Teams Mentioned: {youtube_summary['videos_with_teams']}")
        print(f"💰 Betting Terms: {youtube_summary['videos_with_betting_terms']}")
        print(f"🎯 Confidence Indicators: {youtube_summary['videos_with_confidence']}")

        print(f"\n🤖 AI COUNCIL ANALYSES:")
        for analysis in analyses:
            print(f"  {analysis['source']}: {analysis['confidence']:.1%} confidence")

        print(f"\n🎯 CONSENSUS CONFIDENCE: {consensus['confidence']:.1%}")
        print(f"📋 AI SOURCES: {', '.join(consensus['sources'])}")

        print("\n" + "=" * 80)


async def main():
    council = EnhancedAICouncil()
    results_file = await council.run_enhanced_analysis()

    if results_file:
        print(f"\n✅ Enhanced analysis complete! Results saved to: {results_file}")
    else:
        print("\n❌ Analysis failed")


if __name__ == "__main__":
    asyncio.run(main())
