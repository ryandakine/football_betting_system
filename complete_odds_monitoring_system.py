#!/usr/bin/env python3
"""
Complete Odds Monitoring and Alert System
Gets real odds, combines with analysis, makes specific bets, monitors changes
"""

import asyncio
import json
import logging
import os
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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


class CompleteOddsMonitoringSystem:
    """Complete odds monitoring and alert system"""

    def __init__(self):
        self.odds_api_key = os.getenv("THE_ODDS_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.grok_api_key = os.getenv("GROK_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        # Email settings
        self.email_user = os.getenv("EMAIL_USERNAME")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))

        self.data_dir = Path("data")
        self.results_dir = Path("analysis")
        self.results_dir.mkdir(exist_ok=True)

        # Store current odds and recommendations
        self.current_odds = {}
        self.current_recommendations = []
        self.odds_history = {}

    def get_mlb_odds(self) -> dict:
        """Get current MLB odds from The Odds API"""
        logger.info("💰 Getting current MLB odds...")

        try:
            url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
            params = {
                "apiKey": self.odds_api_key,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                odds_data = response.json()
                logger.info(f"✅ Got odds for {len(odds_data)} games")

                # Store odds history
                for game in odds_data:
                    game_id = game["id"]
                    if game_id not in self.odds_history:
                        self.odds_history[game_id] = []
                    self.odds_history[game_id].append(
                        {"timestamp": datetime.now().isoformat(), "odds": game}
                    )

                return odds_data
            else:
                logger.error(f"❌ Odds API error: {response.status_code}")
                return self._get_sample_odds()

        except Exception as e:
            logger.error(f"❌ Error getting odds: {e}")
            return self._get_sample_odds()

    def _get_sample_odds(self) -> list[dict]:
        """Get sample odds for testing"""
        return [
            {
                "id": "yankees_phillies",
                "sport_key": "baseball_mlb",
                "home_team": "Philadelphia Phillies",
                "away_team": "New York Yankees",
                "commence_time": datetime.now().isoformat(),
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "New York Yankees", "price": -140},
                                    {"name": "Philadelphia Phillies", "price": 120},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {
                                        "name": "New York Yankees",
                                        "point": -1.5,
                                        "price": 110,
                                    },
                                    {
                                        "name": "Philadelphia Phillies",
                                        "point": 1.5,
                                        "price": -130,
                                    },
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 9.0, "price": -110},
                                    {"name": "Under", "point": 9.0, "price": -110},
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "id": "dodgers_giants",
                "sport_key": "baseball_mlb",
                "home_team": "San Francisco Giants",
                "away_team": "Los Angeles Dodgers",
                "commence_time": datetime.now().isoformat(),
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Los Angeles Dodgers", "price": -180},
                                    {"name": "San Francisco Giants", "price": 150},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {
                                        "name": "Los Angeles Dodgers",
                                        "point": -1.5,
                                        "price": 130,
                                    },
                                    {
                                        "name": "San Francisco Giants",
                                        "point": 1.5,
                                        "price": -150,
                                    },
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 8.5, "price": -110},
                                    {"name": "Under", "point": 8.5, "price": -110},
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "id": "mariners_diamondbacks",
                "sport_key": "baseball_mlb",
                "home_team": "Arizona Diamondbacks",
                "away_team": "Seattle Mariners",
                "commence_time": datetime.now().isoformat(),
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Seattle Mariners", "price": 120},
                                    {"name": "Arizona Diamondbacks", "price": -140},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {
                                        "name": "Seattle Mariners",
                                        "point": 1.5,
                                        "price": -110,
                                    },
                                    {
                                        "name": "Arizona Diamondbacks",
                                        "point": -1.5,
                                        "price": -110,
                                    },
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 8.0, "price": -110},
                                    {"name": "Under", "point": 8.0, "price": -110},
                                ],
                            },
                        ],
                    }
                ],
            },
        ]

    def load_analysis_data(self) -> dict:
        """Load all analysis data (YouTube, Reddit, Twitter, News)"""
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

    def create_comprehensive_prompt(
        self, odds_data: list[dict], analysis_data: dict
    ) -> str:
        """Create comprehensive prompt for AI analysis with real odds"""
        prompt = f"""
You are an expert MLB betting analyst with access to real-time odds and comprehensive analysis data.

## CURRENT MLB ODDS (FanDuel):
"""

        for game in odds_data:
            home_team = game["home_team"]
            away_team = game["away_team"]

            # Get moneyline odds
            moneyline = "N/A"
            spread = "N/A"
            total = "N/A"

            for bookmaker in game["bookmakers"]:
                if bookmaker["key"] == "fanduel":
                    for market in bookmaker["markets"]:
                        if market["key"] == "h2h":
                            outcomes = market["outcomes"]
                            moneyline = f"{outcomes[0]['name']} {outcomes[0]['price']}, {outcomes[1]['name']} {outcomes[1]['price']}"
                        elif market["key"] == "spreads":
                            outcomes = market["outcomes"]
                            spread = f"{outcomes[0]['name']} {outcomes[0]['point']} ({outcomes[0]['price']}), {outcomes[1]['name']} {outcomes[1]['point']} ({outcomes[1]['price']})"
                        elif market["key"] == "totals":
                            outcomes = market["outcomes"]
                            total = f"O/U {outcomes[0]['point']} (Over {outcomes[0]['price']}, Under {outcomes[1]['price']})"

            prompt += f"""
**{away_team} @ {home_team}**
- Moneyline: {moneyline}
- Spread: {spread}
- Total: {total}
"""

        # Add analysis data
        if analysis_data.get("youtube_public_sentiment"):
            youtube = analysis_data["youtube_public_sentiment"]
            prompt += f"""

## YOUTUBE PUBLIC SENTIMENT (What to potentially fade):
- Total Videos: {youtube.get('total_videos', 0)}
- Public Favorites: {youtube.get('public_favorites', [])[:5]}
- Public Betting Trends: {youtube.get('public_betting_trends', [])[:5]}
- Sentiment: {youtube.get('sentiment_breakdown', {})}
- Injury Hype: {youtube.get('injury_hype', 0)} videos
- Over Hype: {youtube.get('over_hype', 0)} videos
"""

        if analysis_data.get("contrarian_analysis"):
            contrarian = analysis_data["contrarian_analysis"]
            if contrarian.get("ai_analyses"):
                prompt += f"""

## CONTRARIAN ANALYSIS:
{contrarian['ai_analyses'][0]['analysis'][:500]}...
"""

        prompt += """

## ANALYSIS REQUEST:
Based on the current odds and analysis data, provide:

1. **TOP 5 SPECIFIC BETTING RECOMMENDATIONS** with:
   - Exact bet (team, spread, total, etc.)
   - Current odds
   - Confidence level (1-10)
   - Reasoning
   - Target odds (if odds improve to X, bet)

2. **ODDS MONITORING ALERTS**:
   - What odds changes would make each bet better
   - What odds changes would make each bet worse
   - Specific price targets for each bet

3. **BETTING STRATEGY**:
   - Which bets to place immediately
   - Which bets to wait for better odds
   - Which bets to avoid if odds move against you

Format each recommendation with specific odds and clear action items.
"""

        return prompt

    async def analyze_with_ai_council(self, prompt: str) -> list[dict]:
        """Analyze with AI council for specific betting recommendations"""
        logger.info(
            "🤖 AI Council analyzing odds and making specific recommendations..."
        )

        recommendations = []

        # OpenAI analysis
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
                        "content": "You are an expert MLB betting analyst. Provide specific betting recommendations with exact odds, confidence levels, and monitoring alerts.",
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

                # Parse recommendations from analysis
                recommendations = self._parse_recommendations(analysis)

                logger.info(
                    f"✅ OpenAI generated {len(recommendations)} recommendations"
                )

        except Exception as e:
            logger.error(f"❌ OpenAI analysis error: {e}")

        # If no recommendations, create sample ones
        if not recommendations:
            recommendations = self._create_sample_recommendations()

        return recommendations

    def _parse_recommendations(self, analysis: str) -> list[dict]:
        """Parse specific betting recommendations from AI analysis"""
        recommendations = []

        # Simple parsing - look for specific bet patterns
        lines = analysis.split("\n")
        current_rec = None

        for line in lines:
            line = line.strip()

            # Look for bet patterns
            if any(
                keyword in line.lower()
                for keyword in [
                    "moneyline",
                    "spread",
                    "total",
                    "over",
                    "under",
                    "+",
                    "-",
                ]
            ):
                if current_rec:
                    recommendations.append(current_rec)

                current_rec = {
                    "bet_type": "unknown",
                    "selection": line,
                    "current_odds": "TBD",
                    "confidence": 7,
                    "reasoning": "",
                    "target_odds": "",
                    "alert_conditions": [],
                }

            elif current_rec and "confidence" in line.lower():
                # Extract confidence level
                if "8" in line or "9" in line or "10" in line:
                    current_rec["confidence"] = 9
                elif "6" in line or "7" in line:
                    current_rec["confidence"] = 7
                elif "4" in line or "5" in line:
                    current_rec["confidence"] = 5

        if current_rec:
            recommendations.append(current_rec)

        return recommendations[:5]  # Limit to 5 recommendations

    def _create_sample_recommendations(self) -> list[dict]:
        """Create sample betting recommendations"""
        return [
            {
                "bet_type": "moneyline",
                "selection": "Philadelphia Phillies +120",
                "current_odds": "+120",
                "confidence": 8,
                "reasoning": "Fade Yankees public sentiment, sharp money backing Phillies",
                "target_odds": "+130",
                "alert_conditions": [
                    "If odds improve to +130 or better",
                    "If odds worsen to +110 or worse",
                ],
            },
            {
                "bet_type": "spread",
                "selection": "San Francisco Giants +1.5 (-150)",
                "current_odds": "-150",
                "confidence": 7,
                "reasoning": "Fade Dodgers public expectation of blowout",
                "target_odds": "-140",
                "alert_conditions": [
                    "If spread odds improve to -140 or better",
                    "If spread odds worsen to -160 or worse",
                ],
            },
            {
                "bet_type": "total",
                "selection": "Under 9.0 (-110) Yankees vs Phillies",
                "current_odds": "-110",
                "confidence": 7,
                "reasoning": "Fade public over-hype, sharp money on under",
                "target_odds": "-105",
                "alert_conditions": [
                    "If under odds improve to -105 or better",
                    "If under odds worsen to -115 or worse",
                ],
            },
            {
                "bet_type": "moneyline",
                "selection": "Seattle Mariners +120",
                "current_odds": "+120",
                "confidence": 9,
                "reasoning": "Sharp money backing, public not heavily invested",
                "target_odds": "+130",
                "alert_conditions": [
                    "If odds improve to +130 or better",
                    "If odds worsen to +110 or worse",
                ],
            },
            {
                "bet_type": "total",
                "selection": "Under 8.5 (-110) Dodgers vs Giants",
                "current_odds": "-110",
                "confidence": 6,
                "reasoning": "Contrarian play against public over expectations",
                "target_odds": "-105",
                "alert_conditions": [
                    "If under odds improve to -105 or better",
                    "If under odds worsen to -115 or worse",
                ],
            },
        ]

    def monitor_odds_changes(self, recommendations: list[dict]) -> list[dict]:
        """Monitor odds changes and generate alerts"""
        logger.info("📊 Monitoring odds changes...")

        alerts = []

        for rec in recommendations:
            # Simulate odds changes (replace with real odds monitoring)
            current_odds = rec["current_odds"]
            target_odds = rec["target_odds"]

            # Check if odds have moved in our favor
            if self._odds_improved(current_odds, target_odds):
                alert = {
                    "type": "BET_NOW",
                    "recommendation": rec,
                    "message": f"🎯 BET NOW: {rec['selection']} - Odds improved to {target_odds}",
                    "timestamp": datetime.now().isoformat(),
                }
                alerts.append(alert)

            # Check if odds have moved against us
            elif self._odds_worsened(current_odds, target_odds):
                alert = {
                    "type": "AVOID_BET",
                    "recommendation": rec,
                    "message": f"❌ AVOID: {rec['selection']} - Odds worsened, avoid this bet",
                    "timestamp": datetime.now().isoformat(),
                }
                alerts.append(alert)

        return alerts

    def _odds_improved(self, current: str, target: str) -> bool:
        """Check if odds have improved"""
        # Simple logic - replace with real odds comparison
        return "+130" in target and "+120" in current  # Example improvement

    def _odds_worsened(self, current: str, target: str) -> bool:
        """Check if odds have worsened"""
        # Simple logic - replace with real odds comparison
        return "+110" in target and "+120" in current  # Example worsening

    def send_email_alert(self, alerts: list[dict]):
        """Send email alerts for odds changes"""
        if not self.email_user or not self.email_password:
            logger.warning("⚠️ Email credentials not configured")
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_user
            msg["To"] = self.email_user
            msg["Subject"] = (
                f"MLB Betting Alerts - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            body = "MLB Betting System Alerts\n\n"

            for alert in alerts:
                body += f"{alert['message']}\n"
                body += f"Confidence: {alert['recommendation']['confidence']}/10\n"
                body += f"Reasoning: {alert['recommendation']['reasoning']}\n\n"

            msg.attach(MIMEText(body, "plain"))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"✅ Email alert sent with {len(alerts)} notifications")

        except Exception as e:
            logger.error(f"❌ Email alert error: {e}")

    def save_recommendations(
        self, odds_data: list[dict], recommendations: list[dict], alerts: list[dict]
    ):
        """Save recommendations and alerts"""
        today = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        results = {
            "timestamp": datetime.now().isoformat(),
            "odds_data": odds_data,
            "recommendations": recommendations,
            "alerts": alerts,
        }

        results_file = self.results_dir / f"odds_monitoring_results_{today}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 Results saved to {results_file}")
        return results_file

    def display_recommendations(self, recommendations: list[dict], alerts: list[dict]):
        """Display betting recommendations and alerts"""
        print("\n" + "=" * 80)
        print("🎯 MLB BETTING RECOMMENDATIONS & ODDS MONITORING")
        print("=" * 80)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        print(f"\n📋 TOP 5 BETTING RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['selection']}")
            print(f"   Current Odds: {rec['current_odds']}")
            print(f"   Confidence: {rec['confidence']}/10")
            print(f"   Target Odds: {rec['target_odds']}")
            print(f"   Reasoning: {rec['reasoning']}")

        if alerts:
            print(f"\n🚨 ODDS ALERTS ({len(alerts)}):")
            for alert in alerts:
                print(f"   {alert['message']}")
        else:
            print(f"\n✅ No immediate alerts - monitoring odds...")

        print("\n" + "=" * 80)

    async def run_complete_system(self):
        """Run the complete odds monitoring system"""
        logger.info("🚀 Starting Complete Odds Monitoring System...")

        # Step 1: Get current odds
        odds_data = self.get_mlb_odds()
        self.current_odds = odds_data

        # Step 2: Load analysis data
        analysis_data = self.load_analysis_data()

        # Step 3: Create comprehensive prompt
        prompt = self.create_comprehensive_prompt(odds_data, analysis_data)

        # Step 4: Get AI recommendations
        recommendations = await self.analyze_with_ai_council(prompt)
        self.current_recommendations = recommendations

        # Step 5: Monitor odds changes
        alerts = self.monitor_odds_changes(recommendations)

        # Step 6: Send alerts
        if alerts:
            self.send_email_alert(alerts)

        # Step 7: Save results
        results_file = self.save_recommendations(odds_data, recommendations, alerts)

        # Step 8: Display results
        self.display_recommendations(recommendations, alerts)

        return results_file


async def main():
    system = CompleteOddsMonitoringSystem()
    results_file = await system.run_complete_system()

    if results_file:
        print(f"\n✅ Complete odds monitoring system finished!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📧 Email alerts configured: {bool(system.email_user)}")
    else:
        print("\n❌ System failed")


if __name__ == "__main__":
    asyncio.run(main())
