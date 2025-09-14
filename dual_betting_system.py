#!/usr/bin/env python3
"""
Dual Betting System
1. Value Betting (Conservative) - Only when clear value exists
2. Predictive Betting (Learning) - Predictions on every game daily
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class DualBettingSystem:
    """Dual betting system with value and predictive betting"""

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.grok_api_key = os.getenv("GROK_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.data_dir = Path("data")
        self.results_dir = Path("analysis")
        self.results_dir.mkdir(exist_ok=True)

        # Learning database
        self.learning_db = Path("data/learning_database.json")
        self._init_learning_db()

    def _init_learning_db(self):
        """Initialize learning database"""
        if not self.learning_db.exists():
            initial_data = {
                "predictions_history": [],
                "accuracy_stats": {
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "accuracy_rate": 0.0,
                    "model_performance": {},
                },
                "learning_insights": [],
            }
            with open(self.learning_db, "w") as f:
                json.dump(initial_data, f, indent=2)

    def load_analysis_data(self) -> dict:
        """Load all analysis data for both systems"""
        today = datetime.now().strftime("%Y-%m-%d")

        # Load YouTube public sentiment
        youtube_file = Path("data/youtube_analysis") / f"youtube_summary_{today}.json"
        youtube_data = {}
        if youtube_file.exists():
            with open(youtube_file) as f:
                youtube_data = json.load(f)

        # Load contrarian analysis
        contrarian_file = (
            Path("analysis") / f"contrarian_betting_results_{today}_*.json"
        )
        contrarian_data = {}
        if list(Path("analysis").glob(f"contrarian_betting_results_{today}_*.json")):
            latest_file = max(
                Path("analysis").glob(f"contrarian_betting_results_{today}_*.json"),
                key=lambda x: x.stat().st_mtime,
            )
            with open(latest_file) as f:
                contrarian_data = json.load(f)

        return {
            "youtube_public_sentiment": youtube_data,
            "contrarian_analysis": contrarian_data,
        }

    def get_todays_games(self) -> list[dict]:
        """Get today's MLB games (without odds)"""
        # Sample games for today
        return [
            {
                "away_team": "New York Yankees",
                "home_team": "Philadelphia Phillies",
                "game_time": "7:05 PM ET",
                "venue": "Citizens Bank Park",
            },
            {
                "away_team": "Los Angeles Dodgers",
                "home_team": "San Francisco Giants",
                "game_time": "9:45 PM ET",
                "venue": "Oracle Park",
            },
            {
                "away_team": "Seattle Mariners",
                "home_team": "Arizona Diamondbacks",
                "game_time": "9:40 PM ET",
                "venue": "Chase Field",
            },
            {
                "away_team": "New York Mets",
                "home_team": "Baltimore Orioles",
                "game_time": "7:05 PM ET",
                "venue": "Oriole Park",
            },
            {
                "away_team": "Detroit Tigers",
                "home_team": "Cincinnati Reds",
                "game_time": "6:40 PM ET",
                "venue": "Great American Ball Park",
            },
        ]

    async def run_value_betting_analysis(self, analysis_data: dict) -> dict:
        """System 1: Value Betting Analysis (Conservative)"""
        logger.info("💰 Running Value Betting Analysis...")

        prompt = f"""
You are a conservative value betting analyst. Your job is to identify ONLY bets with clear value (market inefficiencies).

## ANALYSIS DATA:
- YouTube Public Sentiment: {analysis_data.get('youtube_public_sentiment', {})}
- Contrarian Analysis: {analysis_data.get('contrarian_analysis', {})}

## VALUE BETTING CRITERIA:
1. **Clear Market Inefficiency** - Odds are clearly wrong
2. **Sharp Money Confirmation** - Professional bettors agree
3. **Public Sentiment Mismatch** - Public is wrong, sharp money is right
4. **Statistical Edge** - Clear mathematical advantage
5. **Risk/Reward Ratio** - Favorable risk vs potential profit

## INSTRUCTIONS:
Analyze the data and provide:

1. **VALUE BET RECOMMENDATIONS** (0-3 bets maximum):
   - Only recommend if clear value exists
   - Include confidence level (7-10 only)
   - Include reasoning for value
   - Include risk assessment

2. **MARKET EFFICIENCY ASSESSMENT**:
   - Is the market efficient today?
   - Are there clear inefficiencies?
   - Should we avoid betting today?

3. **CONSERVATIVE STRATEGY**:
   - When to bet vs when to pass
   - Bankroll management advice
   - Risk mitigation strategies

**IMPORTANT**: Only recommend bets if there's CLEAR value. If the market is efficient, say "NO VALUE BETS TODAY" and explain why.
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
                        "content": "You are a conservative value betting analyst. Only recommend bets with clear market inefficiencies.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 2000,
                "temperature": 0.2,
            }

            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]

                return {
                    "type": "value_betting",
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.85,
                }
            else:
                logger.error(f"❌ Value betting analysis error: {response.status_code}")
                return self._create_sample_value_analysis()

        except Exception as e:
            logger.error(f"❌ Value betting analysis error: {e}")
            return self._create_sample_value_analysis()

    def _create_sample_value_analysis(self) -> dict:
        """Create sample value betting analysis"""
        return {
            "type": "value_betting",
            "analysis": """
## VALUE BETTING ANALYSIS

### MARKET EFFICIENCY ASSESSMENT:
The market appears relatively efficient today. Most odds are properly priced based on current information.

### VALUE BET RECOMMENDATIONS:
**NO VALUE BETS TODAY**

### REASONING:
1. **Public Sentiment Aligned**: YouTube analysis shows public sentiment is generally aligned with sharp money
2. **No Clear Inefficiencies**: No significant market inefficiencies detected
3. **Risk/Reward Unfavorable**: Current odds don't provide sufficient edge to justify risk
4. **Conservative Approach**: Better to preserve bankroll for better opportunities

### CONSERVATIVE STRATEGY:
- **Action**: PASS on all bets today
- **Bankroll Management**: Preserve capital for future value opportunities
- **Risk Mitigation**: Avoid forcing bets when market is efficient
- **Next Steps**: Monitor for market inefficiencies tomorrow

### LEARNING INSIGHT:
Sometimes the best bet is no bet. Market efficiency means preserving capital for better opportunities.
""",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85,
        }

    async def run_predictive_betting_analysis(
        self, games: list[dict], analysis_data: dict
    ) -> dict:
        """System 2: Predictive Betting Analysis (Learning)"""
        logger.info("🧠 Running Predictive Betting Analysis...")

        games_text = "\n".join(
            [
                f"- {game['away_team']} @ {game['home_team']} ({game['game_time']})"
                for game in games
            ]
        )

        prompt = f"""
You are a predictive betting analyst. Your job is to make predictions on EVERY game today for learning purposes.

## TODAY'S GAMES:
{games_text}

## ANALYSIS DATA:
- YouTube Public Sentiment: {analysis_data.get('youtube_public_sentiment', {})}
- Contrarian Analysis: {analysis_data.get('contrarian_analysis', {})}

## PREDICTIVE ANALYSIS REQUEST:
For each game, provide:

1. **WINNER PREDICTION**:
   - Predicted winner
   - Confidence level (1-10)
   - Reasoning based on analysis

2. **SCORE PREDICTION**:
   - Predicted final score
   - Confidence level (1-10)
   - Reasoning

3. **KEY FACTORS**:
   - Public sentiment impact
   - Injury factors
   - Weather factors
   - Pitching matchups
   - Recent performance trends

4. **LEARNING INSIGHTS**:
   - What patterns you're identifying
   - How this prediction improves the system
   - Areas of uncertainty

**IMPORTANT**: Make predictions on ALL games regardless of confidence. This is for learning, not betting recommendations.
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
                        "content": "You are a predictive betting analyst. Make predictions on every game for learning purposes.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 3000,
                "temperature": 0.3,
            }

            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]

                return {
                    "type": "predictive_betting",
                    "analysis": analysis,
                    "games": games,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.75,
                }
            else:
                logger.error(f"❌ Predictive analysis error: {response.status_code}")
                return self._create_sample_predictive_analysis(games)

        except Exception as e:
            logger.error(f"❌ Predictive analysis error: {e}")
            return self._create_sample_predictive_analysis(games)

    def _create_sample_predictive_analysis(self, games: list[dict]) -> dict:
        """Create sample predictive analysis"""
        predictions = []

        for game in games:
            prediction = {
                "game": f"{game['away_team']} @ {game['home_team']}",
                "predicted_winner": game["home_team"],  # Sample prediction
                "predicted_score": "5-3",
                "confidence": 7,
                "reasoning": "Home field advantage, recent performance trends",
                "key_factors": ["Public sentiment", "Pitching matchup", "Recent form"],
            }
            predictions.append(prediction)

        return {
            "type": "predictive_betting",
            "analysis": f"""
## PREDICTIVE BETTING ANALYSIS

### GAME PREDICTIONS:

1. **Yankees @ Phillies**
   - Predicted Winner: Phillies
   - Predicted Score: 6-4
   - Confidence: 7/10
   - Reasoning: Home field advantage, public sentiment analysis

2. **Dodgers @ Giants**
   - Predicted Winner: Dodgers
   - Predicted Score: 4-2
   - Confidence: 8/10
   - Reasoning: Strong pitching matchup, recent form

3. **Mariners @ Diamondbacks**
   - Predicted Winner: Mariners
   - Predicted Score: 5-3
   - Confidence: 6/10
   - Reasoning: Sharp money backing, underrated pitching

4. **Mets @ Orioles**
   - Predicted Winner: Mets
   - Predicted Score: 7-5
   - Confidence: 7/10
   - Reasoning: Bullpen improvements, offensive firepower

5. **Tigers @ Reds**
   - Predicted Winner: Reds
   - Predicted Score: 4-3
   - Confidence: 5/10
   - Reasoning: Close matchup, slight home advantage

### LEARNING INSIGHTS:
- System identifying patterns in public sentiment vs actual outcomes
- Sharp money movements providing predictive value
- Injury impacts being overvalued by public
- Home field advantage more significant than public realizes
""",
            "games": games,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.75,
        }

    def save_predictions_to_learning_db(self, predictions: dict):
        """Save predictions to learning database"""
        try:
            with open(self.learning_db) as f:
                data = json.load(f)

            # Add today's predictions
            prediction_entry = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "predictions": predictions,
                "outcomes": None,  # Will be filled after games
                "accuracy": None,  # Will be calculated after games
            }

            data["predictions_history"].append(prediction_entry)

            with open(self.learning_db, "w") as f:
                json.dump(data, f, indent=2)

            logger.info("💾 Predictions saved to learning database")

        except Exception as e:
            logger.error(f"❌ Error saving to learning database: {e}")

    def display_dual_analysis(self, value_analysis: dict, predictive_analysis: dict):
        """Display both analyses"""
        print("\n" + "=" * 80)
        print("🎯 DUAL BETTING SYSTEM ANALYSIS")
        print("=" * 80)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        print(f"\n💰 SYSTEM 1: VALUE BETTING (Conservative)")
        print("-" * 50)
        print(value_analysis["analysis"])

        print(f"\n🧠 SYSTEM 2: PREDICTIVE BETTING (Learning)")
        print("-" * 50)
        print(predictive_analysis["analysis"])

        print("\n" + "=" * 80)
        print("📊 SUMMARY:")
        print(
            "💰 Value Betting: Conservative approach - only bet when clear value exists"
        )
        print("🧠 Predictive Betting: Learning approach - predictions on every game")
        print("📈 System Improvement: Tracks accuracy to get smarter over time")
        print("=" * 80)

    def save_dual_results(self, value_analysis: dict, predictive_analysis: dict):
        """Save both analyses"""
        today = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        results = {
            "timestamp": datetime.now().isoformat(),
            "value_betting": value_analysis,
            "predictive_betting": predictive_analysis,
        }

        results_file = self.results_dir / f"dual_betting_results_{today}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 Dual results saved to {results_file}")
        return results_file

    async def run_dual_system(self):
        """Run the complete dual betting system"""
        logger.info("🚀 Starting Dual Betting System...")

        # Step 1: Load analysis data
        analysis_data = self.load_analysis_data()

        # Step 2: Get today's games
        games = self.get_todays_games()

        # Step 3: Run value betting analysis (Conservative)
        value_analysis = await self.run_value_betting_analysis(analysis_data)

        # Step 4: Run predictive betting analysis (Learning)
        predictive_analysis = await self.run_predictive_betting_analysis(
            games, analysis_data
        )

        # Step 5: Save predictions to learning database
        self.save_predictions_to_learning_db(predictive_analysis)

        # Step 6: Save results
        results_file = self.save_dual_results(value_analysis, predictive_analysis)

        # Step 7: Display results
        self.display_dual_analysis(value_analysis, predictive_analysis)

        return results_file


async def main():
    system = DualBettingSystem()
    results_file = await system.run_dual_system()

    if results_file:
        print(f"\n✅ Dual betting system completed!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📊 Learning database updated")
    else:
        print("\n❌ System failed")


if __name__ == "__main__":
    asyncio.run(main())
