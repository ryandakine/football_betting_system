#!/usr/bin/env python3
import asyncio
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path

# Import your existing modules
from ultimate_database_manager import DailyPortfolioData, UltimateDatabaseManager
from ultimate_recommendation_engine import UltimateRecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_system")


async def test_with_yesterday():
    # Use yesterday's date
    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    workdir = Path("C:/Users/himse/mlb_betting_system")

    logger.info(f"🎯 Testing with yesterday's data: {yesterday}")

    # Load yesterday's data
    try:
        sentiment_file = workdir / "sentiment" / f"{yesterday}.json"
        odds_file = workdir / "odds" / f"{yesterday}.json"

        sentiment_data = json.loads(sentiment_file.read_text())
        odds_data = json.loads(odds_file.read_text())

        logger.info(f"📊 Loaded: {len(odds_data.get('games', []))} games")

        # Create mock personalities data
        personalities_data = {
            "meta": {"date": yesterday, "total_predictions": 4},
            "personalities": {
                "claude4": {
                    "role": "The Quantitative Analyst",
                    "specialty": "Market inefficiencies",
                    "predictions": [
                        {
                            "game_id": "yankees_redsox_20250621",
                            "pick": "Yankees ML",
                            "confidence": 75,
                            "stake_units": 3,
                            "signal_weights": {
                                "sentiment_weight": 0.4,
                                "odds_weight": 0.6,
                            },
                            "rationale": "Strong line value detected with Yankees",
                            "primary_signal": "line_value",
                            "analyst_type": "quantitative",
                        }
                    ],
                },
                "gpt4": {
                    "role": "The Sentiment Trader",
                    "specialty": "Public betting patterns",
                    "predictions": [
                        {
                            "game_id": "yankees_redsox_20250621",
                            "pick": "Yankees ML",
                            "confidence": 72,
                            "stake_units": 2,
                            "signal_weights": {
                                "sentiment_weight": 0.7,
                                "odds_weight": 0.3,
                            },
                            "rationale": "Public fading opportunity on Red Sox",
                            "primary_signal": "contrarian",
                            "analyst_type": "sentiment",
                        }
                    ],
                },
            },
        }

        # Test recommendation engine
        engine = UltimateRecommendationEngine(unit_size=10.0)
        portfolio = engine.generate_ultimate_portfolio(personalities_data, odds_data)

        # Display results
        print("\n" + "=" * 80)
        print("🏆 TEST PORTFOLIO RESULTS")
        print("=" * 80)

        meta = portfolio.get("portfolio_meta", {})
        print(f"📊 Total Bets: {meta.get('total_bets', 0)}")
        print(f"💰 Total Stake: ${meta.get('total_stake', 0):.2f}")
        print(f"📈 Expected Value: ${meta.get('total_expected_value', 0):.2f}")

        for bet in portfolio.get("top_opportunities", []):
            print(f"\n🎯 {bet['pick']}")
            print(
                f"   Confidence: {bet['confidence']} | Units: {bet['units']} | EV: {bet['expected_value']}"
            )

        print("=" * 80)
        print("✅ YESTERDAY'S DATA TEST SUCCESSFUL!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_with_yesterday())
