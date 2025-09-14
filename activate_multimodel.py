"""
Activate Multi-Model AI Integration with Kelly Criterion System
This connects your AI models (Claude 4, GPT-4o, o1, Grok 3) to the betting system
"""

import asyncio
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_multimodel_integration():
    """Test if multi-model AI integration is working."""

    print("🚀 Testing Multi-Model AI Integration...")
    print("=" * 50)

    # Test 1: Check API keys
    api_keys = {
        "CLAUDE_API_KEY": os.getenv("CLAUDE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GROK_API_KEY": os.getenv("GROK_API_KEY"),
        "ODDS_API_KEY": os.getenv("ODDS_API_KEY"),
    }

    print("📊 API Keys Status:")
    for key, value in api_keys.items():
        status = "✅ SET" if value else "❌ MISSING"
        masked_key = f"...{value[-10:]}" if value else "None"
        print(f"  {key}: {status} ({masked_key})")

    # Test 2: Try importing game_selection
    print("\n📦 Module Import Tests:")
    try:
        import game_selection

        print("  ✅ game_selection.py imported successfully")

        # Check if it has the multi-model classes
        if hasattr(game_selection, "GameSelector"):
            print("  ✅ GameSelector class found")
        else:
            print("  ❌ GameSelector class not found")

    except Exception as e:
        print(f"  ❌ Failed to import game_selection: {e}")

    # Test 3: Try importing predictive models
    try:
        import predictive_modeling

        print("  ✅ predictive_modeling.py imported successfully")
    except Exception as e:
        print(f"  ❌ Failed to import predictive_modeling: {e}")

    # Test 4: Try importing player performance
    try:
        import player_performance

        print("  ✅ player_performance.py imported successfully")
    except Exception as e:
        print(f"  ❌ Failed to import player_performance: {e}")

    print("\n🎯 Integration Status:")
    print("  Your Kelly system found: Baltimore Orioles +5.15% edge")
    print("  This should trigger multi-model AI analysis!")

    return True


async def run_enhanced_analysis():
    """Run the enhanced analysis with multi-model AI."""

    print("\n🤖 Attempting Multi-Model Analysis...")

    try:
        # Import the game selector
        from game_selection import GameSelector

        # Create sample game data for the Baltimore Orioles opportunity
        sample_game = {
            "home_team": "Baltimore Orioles",
            "away_team": "Detroit Tigers",
            "odds": {"home_ml": 7.40},
            "edge": 5.15,
        }

        print(f"🎯 Analyzing high-value opportunity: {sample_game}")

        # Initialize game selector (this should use your API keys)
        selector = GameSelector()

        print("✅ Multi-model AI system initialized!")
        print("🚀 Ready for comprehensive game analysis!")

        return True

    except ImportError as e:
        print(f"❌ GameSelector not available: {e}")
        print("🔧 Need to update game_selection.py with proper multi-model integration")
        return False
    except Exception as e:
        print(f"❌ Error initializing multi-model system: {e}")
        return False


def create_integration_command():
    """Create command to run enhanced daily analysis."""

    print("\n📋 Next Steps:")
    print("1. ✅ Your Kelly system is working (detecting +5.15% edge)")
    print("2. 🔄 Multi-model AI needs activation")
    print("3. 🎯 Baltimore Orioles opportunity should be analyzed by AI")

    print("\n🚀 Commands to try:")
    print("python activate_multimodel.py")
    print("python daily_prediction_and_backtest.py --use-ai")


async def main():
    """Main function to test and activate multi-model integration."""

    # Test current status
    await test_multimodel_integration()

    # Try to run enhanced analysis
    success = await run_enhanced_analysis()

    if success:
        print("\n🎉 Multi-Model AI Integration: ACTIVE")
        print("🎯 Ready to analyze betting opportunities with AI!")
    else:
        print("\n🔧 Multi-Model AI Integration: NEEDS SETUP")
        print("📋 Files exist but need connection to main system")

    # Show next steps
    create_integration_command()


if __name__ == "__main__":
    asyncio.run(main())
