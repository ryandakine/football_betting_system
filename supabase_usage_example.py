#!/usr/bin/env python3
"""
Supabase Usage Examples for MLB Betting System
=============================================
Examples of how to integrate Supabase into your existing MLB betting system.
"""

import asyncio
import json
from datetime import datetime

from supabase_client import MLBSupabaseClient, save_ai_predictions, save_recommendations
from supabase_config import SupabaseConfig

# Initialize the Supabase client
supabase_client = MLBSupabaseClient()


async def example_save_ai_predictions():
    """Example: Save AI predictions to Supabase instead of SQLite."""

    # Sample AI prediction data (similar to what your system generates)
    sample_predictions = [
        {
            "prediction_id": "run_001_game_123_claude4",
            "run_id": "run_001",
            "date": "2025-01-15",
            "game_id": "game_123",
            "timestamp": datetime.now().isoformat(),
            "ai_model": "claude4",
            "analyst_role": "Quantitative Analyst",
            "specialty": "Statistical Modeling",
            "analyst_type": "advanced",
            "pick": "Yankees ML",
            "confidence": 75,
            "stake_units": 3,
            "rationale": "Strong pitching matchup and recent form analysis",
            "primary_signal": "pitching_advantage",
            "sentiment_weight": 0.6,
            "odds_weight": 0.4,
            "expected_value": "0.08",
            "fade_strength": "0.02",
            "situational_edge": "0.05",
            "signal_convergence": "0.85",
            "token_estimate": 1500,
        },
        {
            "prediction_id": "run_001_game_123_gpt4",
            "run_id": "run_001",
            "date": "2025-01-15",
            "game_id": "game_123",
            "timestamp": datetime.now().isoformat(),
            "ai_model": "gpt4",
            "analyst_role": "Sentiment Analyst",
            "specialty": "Market Sentiment",
            "analyst_type": "sentiment",
            "pick": "Yankees ML",
            "confidence": 70,
            "stake_units": 2,
            "rationale": "Positive social media sentiment and news flow",
            "primary_signal": "sentiment_positive",
            "sentiment_weight": 0.8,
            "odds_weight": 0.2,
            "expected_value": "0.06",
            "fade_strength": "0.01",
            "situational_edge": "0.03",
            "signal_convergence": "0.78",
            "token_estimate": 1200,
        },
    ]

    try:
        # Save to Supabase
        count = await supabase_client.save_ai_predictions(sample_predictions)
        print(f"✅ Saved {count} AI predictions to Supabase")

        # Retrieve from Supabase
        predictions = await supabase_client.get_ai_predictions(run_id="run_001")
        print(f"📊 Retrieved {len(predictions)} predictions from Supabase")

    except Exception as e:
        print(f"❌ Error: {e}")


async def example_save_recommendations():
    """Example: Save betting recommendations to Supabase."""

    # Sample recommendations data
    sample_recommendations = [
        {
            "timestamp": datetime.now().isoformat(),
            "game_id": "game_123",
            "bet_type": "moneyline",
            "selection": "Yankees",
            "odds": -145,
            "bookmaker": "FanDuel",
            "win_probability": 0.65,
            "confidence": 0.75,
            "kelly_fraction": 0.08,
            "bet_size": 30.0,
            "units": 3,
            "expected_value": 0.08,
            "risk_level": "medium",
            "recommendation": "Strong value on Yankees ML",
            "raw_data": {
                "ai_consensus": 0.72,
                "market_movement": "stable",
                "line_movement": -5,
            },
        }
    ]

    try:
        # Save to Supabase
        count = await supabase_client.save_recommendations(sample_recommendations)
        print(f"✅ Saved {count} recommendations to Supabase")

    except Exception as e:
        print(f"❌ Error: {e}")


async def example_save_odds_data():
    """Example: Save odds data to Supabase."""

    # Sample odds data
    sample_odds = [
        {
            "game_id": "game_123",
            "bookmaker": "FanDuel",
            "home_team": "Yankees",
            "away_team": "Red Sox",
            "commence_time": "2025-01-15T19:05:00Z",
            "moneyline_home": -145,
            "moneyline_away": 125,
            "home_price": -145,
            "away_price": 125,
            "edge_cents": 8,
            "value_rating": "GOOD_VALUE",
            "market": "moneyline",
            "total": 8.5,
            "source": "api_fetch",
            "date": "2025-01-15",
        }
    ]

    try:
        # Save to Supabase
        count = await supabase_client.save_odds_data(sample_odds)
        print(f"✅ Saved {count} odds records to Supabase")

    except Exception as e:
        print(f"❌ Error: {e}")


async def example_save_sentiment_data():
    """Example: Save sentiment data to Supabase."""

    # Sample sentiment data
    sample_sentiment = {
        "date": "2025-01-15",
        "source": "youtube_sentiment",
        "data": {
            "overall": {"bullish": 0.65, "bearish": 0.25, "neutral": 0.10},
            "teams": {
                "Yankees": {
                    "sentiment": "positive",
                    "confidence": 0.78,
                    "mentions": 45,
                },
                "Red Sox": {
                    "sentiment": "negative",
                    "confidence": 0.65,
                    "mentions": 32,
                },
            },
            "videos_analyzed": 12,
            "confidence": 0.72,
        },
        "generated_at": datetime.now().isoformat(),
    }

    try:
        # Save to Supabase
        record_id = await supabase_client.save_sentiment_data(sample_sentiment)
        print(f"✅ Saved sentiment data with ID: {record_id}")

    except Exception as e:
        print(f"❌ Error: {e}")


async def example_migrate_existing_data():
    """Example: How to migrate existing data from SQLite to Supabase."""

    # This would typically be done using the migration script
    # But here's an example of how to do it manually

    # 1. Read from existing SQLite database
    import sqlite3

    try:
        # Connect to existing SQLite database
        conn = sqlite3.connect("gold_standard_betting.db")
        cursor = conn.cursor()

        # Get existing recommendations
        cursor.execute("SELECT * FROM recommendations LIMIT 10")
        rows = cursor.fetchall()

        if rows:
            # Convert to list of dictionaries
            columns = [description[0] for description in cursor.description]
            recommendations = []

            for row in rows:
                rec = dict(zip(columns, row))
                # Convert raw_data JSON string
                if "raw_data" in rec and rec["raw_data"]:
                    try:
                        rec["raw_data"] = json.loads(rec["raw_data"])
                    except:
                        pass
                recommendations.append(rec)

            # Save to Supabase
            count = await supabase_client.save_recommendations(recommendations)
            print(f"✅ Migrated {count} recommendations to Supabase")

        conn.close()

    except Exception as e:
        print(f"❌ Migration error: {e}")


async def example_replace_existing_database_calls():
    """Example: How to replace existing database calls in your code."""

    # OLD WAY (SQLite):
    # db_manager = GoldStandardDatabase()
    # db_manager.save_recommendations(recommendations)

    # NEW WAY (Supabase):
    # await supabase_client.save_recommendations(recommendations)

    # Example with your existing system structure:

    # 1. Replace UltimateDatabaseManager calls
    # OLD:
    # async with UltimateDatabaseManager() as db:
    #     await db.save_ai_predictions(predictions)

    # NEW:
    # await supabase_client.save_ai_predictions(predictions)

    # 2. Replace GoldStandardDatabase calls
    # OLD:
    # db = GoldStandardDatabase()
    # db.save_recommendations(recommendations)

    # NEW:
    # await supabase_client.save_recommendations(recommendations)

    # 3. Replace PerformanceTracker calls
    # OLD:
    # tracker = PerformanceTracker()
    # tracker.save_bet(bet)

    # NEW:
    # await supabase_client.save_professional_bet(bet.__dict__)

    print("✅ Examples of how to replace existing database calls")


async def main():
    """Run all examples."""
    print("🚀 Supabase Integration Examples")
    print("=" * 40)

    # Validate configuration first
    try:
        SupabaseConfig.validate_config()
        print("✅ Supabase configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return

    # Test connection
    if not await supabase_client.test_connection():
        print("❌ Cannot connect to Supabase")
        return

    print("\n📊 Running examples...")

    await example_save_ai_predictions()
    await example_save_recommendations()
    await example_save_odds_data()
    await example_save_sentiment_data()
    await example_replace_existing_database_calls()

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
