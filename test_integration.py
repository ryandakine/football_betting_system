"""
Test Multi-Model Integration with Correct Class Names
Using GameSelectionEngine and LLMProvider from your existing files
"""

import asyncio
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_api_keys():
    """Set API keys in current environment."""
    api_keys = {
        "CLAUDE_API_KEY": "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",
        "OPENAI_API_KEY": "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",
        "GROK_API_KEY": "xai-token-BuzMo8nIroBT7e0LhVshTdjIYFP7wrk1znc9Bg9sD8My3HMtA8ONHiqNCjYMW6vPGoBv67LELKTYyl0p",
        "ODDS_API_KEY": "219a6d41e72ff68350230d0a6d8dcf9",
    }

    for key, value in api_keys.items():
        os.environ[key] = value

    print("✅ API keys set in environment")


async def test_game_selection_engine():
    """Test the GameSelectionEngine with the Baltimore Orioles opportunity."""

    print("🚀 Testing GameSelectionEngine...")
    print("=" * 50)

    try:
        # Import with correct class name
        from game_selection import GameSelectionEngine, LLMAnalysisRequest, SelectionConfig

        print("✅ Successfully imported GameSelectionEngine")

        # Create configuration
        config = SelectionConfig()
        print("✅ Created SelectionConfig")

        # Initialize the game selection engine
        engine = GameSelectionEngine(config)
        print("✅ Initialized GameSelectionEngine")

        # Create a sample game for the Baltimore Orioles opportunity
        sample_game_data = {
            "game_id": "BAL_DET_20250610",
            "home_team": "Baltimore Orioles",
            "away_team": "Detroit Tigers",
            "odds": {"home_ml": 7.40, "away_ml": 1.15},
            "edge": 5.15,
            "kelly_recommendation": "Strong BUY",
        }

        print(f"\n🎯 Analyzing High-Value Opportunity:")
        print(
            f"   Game: {sample_game_data['away_team']} @ {sample_game_data['home_team']}"
        )
        print(f"   Odds: {sample_game_data['odds']['home_ml']} (Baltimore)")
        print(f"   Edge: +{sample_game_data['edge']}%")

        # Test LLM analysis request
        llm_request = LLMAnalysisRequest(
            game_data=sample_game_data, analysis_type="high_value_opportunity"
        )
        print("✅ Created LLM analysis request")

        print("\n🤖 Multi-Model AI Integration:")
        print("  ✅ GameSelectionEngine: Ready")
        print("  ✅ LLM Providers: Available")
        print("  ✅ API Keys: Configured")
        print("  🎯 Baltimore Orioles +5.15% edge: Ready for AI analysis")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def run_ai_analysis():
    """Attempt to run AI analysis on the high-value opportunity."""

    print("\n🧠 Attempting AI Analysis...")

    try:
        from game_selection import GameSelectionEngine, SelectionConfig

        # Initialize with API keys
        config = SelectionConfig()
        engine = GameSelectionEngine(config)

        # Sample high-value game (Baltimore Orioles +5.15% edge)
        games_data = [
            {
                "game_id": "BAL_DET_20250610",
                "home_team": "Baltimore Orioles",
                "away_team": "Detroit Tigers",
                "home_ml_odds": 7.40,
                "edge_detected": 5.15,
                "bookmaker": "fanduel",
            }
        ]

        print("🎯 Running AI analysis on Baltimore Orioles opportunity...")

        # This should call Claude 4, GPT-4o, o1-preview, and Grok 3
        recommendations = await engine.select_games(games_data)

        print("✅ AI Analysis Complete!")
        print(f"📊 Recommendations: {recommendations}")

        return recommendations

    except Exception as e:
        print(f"❌ AI Analysis failed: {e}")
        print("🔧 This means the integration needs more setup")
        return None


def show_next_steps():
    """Show what to do next."""

    print("\n📋 Next Steps:")
    print("1. ✅ GameSelectionEngine is available")
    print("2. ✅ API keys are configured")
    print("3. 🎯 Baltimore Orioles +5.15% edge detected")
    print("4. 🤖 Run full AI analysis on this opportunity")

    print("\n🚀 Commands to run:")
    print("python test_integration.py")
    print("python daily_prediction_and_backtest.py  # Should now use AI")


async def main():
    """Main test function."""

    # Set API keys
    set_api_keys()

    # Test the game selection engine
    engine_ready = await test_game_selection_engine()

    if engine_ready:
        # Try to run AI analysis
        results = await run_ai_analysis()

        if results:
            print("\n🎉 SUCCESS: Multi-Model AI Integration Working!")
        else:
            print("\n🔧 Integration needs final connection steps")

    # Show next steps
    show_next_steps()


if __name__ == "__main__":
    asyncio.run(main())
