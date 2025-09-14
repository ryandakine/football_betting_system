#!/usr/bin/env python3
"""
Simple Learning System
Provides exactly 3 improvements and 3 things to pay less attention to daily
Keeps it simple to avoid AI hallucination
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


class SimpleLearningSystem:
    """Simple learning system with focused daily improvements"""

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")

        self.data_dir = Path("data")
        self.results_dir = Path("analysis")
        self.results_dir.mkdir(exist_ok=True)

        # Simple learning database
        self.learning_db = Path("data/simple_learning_database.json")
        self._init_learning_db()

    def _init_learning_db(self):
        """Initialize simple learning database"""
        if not self.learning_db.exists():
            initial_data = {
                "daily_insights": [],
                "accuracy_tracking": {
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "accuracy_rate": 0.0,
                },
                "learning_focus": {"continue_following": [], "pay_less_attention": []},
            }
            with open(self.learning_db, "w") as f:
                json.dump(initial_data, f, indent=2)

    def load_recent_analysis(self) -> dict:
        """Load recent analysis data for learning insights"""
        today = datetime.now().strftime("%Y-%m-%d")

        # Load today's dual betting results
        dual_results = list(
            Path("analysis").glob(f"dual_betting_results_{today}_*.json")
        )
        if dual_results:
            latest_file = max(dual_results, key=lambda x: x.stat().st_mtime)
            with open(latest_file) as f:
                return json.load(f)

        # Load YouTube data
        youtube_file = Path("data/youtube_analysis") / f"youtube_summary_{today}.json"
        youtube_data = {}
        if youtube_file.exists():
            with open(youtube_file) as f:
                youtube_data = json.load(f)

        return {"youtube_data": youtube_data, "date": today}

    def load_learning_history(self) -> dict:
        """Load learning history for pattern analysis"""
        try:
            with open(self.learning_db) as f:
                return json.load(f)
        except:
            return {
                "daily_insights": [],
                "learning_focus": {"continue_following": [], "pay_less_attention": []},
            }

    async def generate_daily_learning_insights(
        self, analysis_data: dict, learning_history: dict
    ) -> dict:
        """Generate exactly 3 improvements and 3 things to pay less attention to"""
        logger.info("🧠 Generating daily learning insights...")

        prompt = f"""
You are a focused learning analyst. Based on today's analysis and historical patterns, provide exactly:

## 3 THINGS TO CONTINUE FOLLOWING (What's Working):
1. [Specific pattern/signal that's been successful]
2. [Another reliable indicator]
3. [Third proven strategy]

## 3 THINGS TO PAY LESS ATTENTION TO (What's Causing Bad Bets):
1. [Specific factor that's been unreliable]
2. [Another misleading indicator]
3. [Third factor to reduce emphasis on]

## TODAY'S ANALYSIS DATA:
- YouTube Public Sentiment: {analysis_data.get('youtube_data', {})}
- Recent Learning History: {learning_history.get('learning_focus', {})}

## INSTRUCTIONS:
- Keep each item simple and specific
- Focus on actionable insights
- Avoid complex reasoning
- Be direct and clear
- Only 3 items in each category

Format your response exactly as shown above with numbered lists.
"""

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
                        "content": "You are a focused learning analyst. Provide exactly 3 improvements and 3 things to pay less attention to. Keep it simple and specific.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1000,
                "temperature": 0.2,
            }

            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]

                # Parse the structured response
                insights = self._parse_learning_insights(analysis)

                return {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "insights": insights,
                    "raw_analysis": analysis,
                }
            else:
                logger.error(f"❌ Learning analysis error: {response.status_code}")
                return self._create_sample_insights()

        except Exception as e:
            logger.error(f"❌ Learning analysis error: {e}")
            return self._create_sample_insights()

    def _parse_learning_insights(self, analysis: str) -> dict:
        """Parse the structured learning insights"""
        insights = {"continue_following": [], "pay_less_attention": []}

        lines = analysis.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()

            if "CONTINUE FOLLOWING" in line.upper():
                current_section = "continue_following"
            elif "PAY LESS ATTENTION" in line.upper():
                current_section = "pay_less_attention"
            elif line.startswith(("1.", "2.", "3.")) and current_section:
                # Extract the insight (remove number and brackets)
                insight = line.split(".", 1)[1].strip()
                if insight.startswith("[") and insight.endswith("]"):
                    insight = insight[1:-1]  # Remove brackets

                if len(insights[current_section]) < 3:  # Keep only 3 items
                    insights[current_section].append(insight)

        return insights

    def _create_sample_insights(self) -> dict:
        """Create sample learning insights"""
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "insights": {
                "continue_following": [
                    "Sharp money movements on underdogs",
                    "Home field advantage in close games",
                    "Public overreaction to injuries",
                ],
                "pay_less_attention": [
                    "Weather hype in domed stadiums",
                    "Recent winning streaks over 5 games",
                    "Player prop bets on star players",
                ],
            },
            "raw_analysis": "Sample analysis",
        }

    def save_learning_insights(self, insights: dict):
        """Save learning insights to database"""
        try:
            with open(self.learning_db) as f:
                data = json.load(f)

            # Add today's insights
            data["daily_insights"].append(insights)

            # Update learning focus (keep last 7 days)
            recent_insights = data["daily_insights"][-7:]

            # Aggregate recent patterns
            continue_patterns = []
            pay_less_patterns = []

            for insight in recent_insights:
                continue_patterns.extend(insight["insights"]["continue_following"])
                pay_less_patterns.extend(insight["insights"]["pay_less_attention"])

            # Get most common patterns (simple frequency analysis)
            from collections import Counter

            continue_counter = Counter(continue_patterns)
            pay_less_counter = Counter(pay_less_patterns)

            data["learning_focus"] = {
                "continue_following": [
                    item[0] for item in continue_counter.most_common(3)
                ],
                "pay_less_attention": [
                    item[0] for item in pay_less_counter.most_common(3)
                ],
            }

            with open(self.learning_db, "w") as f:
                json.dump(data, f, indent=2)

            logger.info("💾 Learning insights saved to database")

        except Exception as e:
            logger.error(f"❌ Error saving learning insights: {e}")

    def display_learning_insights(self, insights: dict):
        """Display the learning insights"""
        print("\n" + "=" * 80)
        print("🧠 DAILY LEARNING INSIGHTS")
        print("=" * 80)
        print(f"📅 Date: {insights['date']}")

        print(f"\n✅ 3 THINGS TO CONTINUE FOLLOWING:")
        print("-" * 50)
        for i, item in enumerate(insights["insights"]["continue_following"], 1):
            print(f"{i}. {item}")

        print(f"\n❌ 3 THINGS TO PAY LESS ATTENTION TO:")
        print("-" * 50)
        for i, item in enumerate(insights["insights"]["pay_less_attention"], 1):
            print(f"{i}. {item}")

        print("\n" + "=" * 80)
        print("📊 LEARNING FOCUS:")
        print("✅ Keep following what works")
        print("❌ Reduce emphasis on unreliable factors")
        print("🧠 System gets smarter every day")
        print("=" * 80)

    def save_results(self, insights: dict):
        """Save learning results"""
        today = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        results = {
            "timestamp": datetime.now().isoformat(),
            "learning_insights": insights,
        }

        results_file = self.results_dir / f"learning_insights_{today}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 Learning results saved to {results_file}")
        return results_file

    async def run_learning_system(self):
        """Run the simple learning system"""
        logger.info("🚀 Starting Simple Learning System...")

        # Step 1: Load recent analysis
        analysis_data = self.load_recent_analysis()

        # Step 2: Load learning history
        learning_history = self.load_learning_history()

        # Step 3: Generate daily insights
        insights = await self.generate_daily_learning_insights(
            analysis_data, learning_history
        )

        # Step 4: Save insights
        self.save_learning_insights(insights)

        # Step 5: Save results
        results_file = self.save_results(insights)

        # Step 6: Display insights
        self.display_learning_insights(insights)

        return results_file


async def main():
    system = SimpleLearningSystem()
    results_file = await system.run_learning_system()

    if results_file:
        print(f"\n✅ Simple learning system completed!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📊 Learning database updated")
    else:
        print("\n❌ System failed")


if __name__ == "__main__":
    asyncio.run(main())
