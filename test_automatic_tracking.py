#!/usr/bin/env python3
"""
Test Automatic Outcome Tracking
===============================
Test script to verify the automatic outcome tracking system.
"""

import asyncio
import json
import logging
from pathlib import Path

from automatic_outcome_tracker import AutomaticOutcomeTracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_tracking")


async def test_automatic_tracking():
    """Test the automatic outcome tracking system."""
    print("🧪 Testing Automatic Outcome Tracking System")
    print("=" * 50)

    # Initialize tracker
    tracker = AutomaticOutcomeTracker()

    # Test 1: Check status
    print("\n1. Checking tracking status...")
    status = await tracker.get_tracking_status()
    print(f"   Tracking active: {status.get('tracking_active', False)}")
    print(f"   Pending predictions: {status.get('pending_predictions', 0)}")
    print(f"   Completed predictions: {status.get('completed_predictions', 0)}")

    # Test 2: Add manual outcomes
    print("\n2. Adding test manual outcomes...")
    await tracker.add_manual_outcome("test_game_1", "Yankees", 5, 3)
    await tracker.add_manual_outcome("test_game_2", "Red Sox", 2, 4)
    await tracker.add_manual_outcome("test_game_3", "Dodgers", 6, 1)
    print("   ✅ Added 3 test outcomes")

    # Test 3: Check for completed games
    print("\n3. Checking for completed games...")
    await tracker.check_for_completed_games()
    print("   ✅ Check completed")

    # Test 4: Check status again
    print("\n4. Checking status after processing...")
    status = await tracker.get_tracking_status()
    print(f"   Pending predictions: {status.get('pending_predictions', 0)}")
    print(f"   Completed predictions: {status.get('completed_predictions', 0)}")

    # Test 5: Test bridge communication
    print("\n5. Testing bridge communication...")
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8767/ping", timeout=5.0)
            if response.status_code == 200:
                print("   ✅ Bridge is responding")
            else:
                print("   ⚠️ Bridge responded with error")
    except Exception as e:
        print(f"   ❌ Bridge not available: {e}")

    print("\n✅ Testing complete!")


async def test_learning_insights():
    """Test getting learning insights."""
    print("\n🧠 Testing Learning Insights")
    print("=" * 30)

    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8767/learning-insights", timeout=5.0
            )
            if response.status_code == 200:
                insights = response.json()
                print("   Learning insights retrieved:")
                print(
                    f"   - Recent accuracy: {insights.get('learning_insights', {}).get('recent_accuracy', 0):.2%}"
                )
                print(
                    f"   - Total predictions: {insights.get('learning_insights', {}).get('total_predictions', 0)}"
                )
                print(
                    f"   - Active patterns: {insights.get('learning_insights', {}).get('active_patterns', 0)}"
                )
            else:
                print("   ⚠️ Could not get learning insights")
    except Exception as e:
        print(f"   ❌ Error getting insights: {e}")


async def create_test_predictions():
    """Create some test predictions for testing."""
    print("\n📝 Creating test predictions...")

    # This would normally be done by your AI Council
    # For testing, we'll create some dummy predictions
    test_predictions = [
        {
            "game_id": "test_game_1",
            "home_team": "Yankees",
            "away_team": "Red Sox",
            "predicted_winner": "Yankees",
            "confidence": 0.75,
            "stake": 100.0,
            "odds": 1.85,
            "model_name": "test_model",
            "features": {"test": True},
        },
        {
            "game_id": "test_game_2",
            "home_team": "Dodgers",
            "away_team": "Giants",
            "predicted_winner": "Dodgers",
            "confidence": 0.65,
            "stake": 50.0,
            "odds": 2.10,
            "model_name": "test_model",
            "features": {"test": True},
        },
    ]

    # Add these to the learning system
    from simple_learning_integration import SimpleLearningTracker, record_prediction_for_learning

    tracker = SimpleLearningTracker()

    for pred in test_predictions:
        pred_id = record_prediction_for_learning(pred, tracker)
        print(f"   Created prediction: {pred_id}")

    print("   ✅ Test predictions created")


async def main():
    """Run all tests."""
    print("🚀 Starting Automatic Outcome Tracking Tests")
    print("=" * 60)

    # Create test predictions first
    await create_test_predictions()

    # Test the tracking system
    await test_automatic_tracking()

    # Test learning insights
    await test_learning_insights()

    print("\n🎉 All tests completed!")
    print("\nTo start automatic tracking:")
    print("   python start_automatic_tracking.py")
    print("   or")
    print("   start_automatic_tracking.bat")


if __name__ == "__main__":
    asyncio.run(main())
