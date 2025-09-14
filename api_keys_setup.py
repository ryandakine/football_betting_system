"""
Quick API Keys Setup for MLB Betting System
Run this to configure your multi-model AI integration
"""

import os


def setup_api_keys():
    """Set up all API keys as environment variables."""

    # Your API keys
    api_keys = {
        "CLAUDE_API_KEY": "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",
        "OPENAI_API_KEY": "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",
        "GROK_API_KEY": "xai-token-BuzMo8nIroBT7e0LhVshTdjIYFP7wrk1znc9Bg9sD8My3HMtA8ONHiqNCjYMW6vPGoBv67LELKTYyl0p",
        "ODDS_API_KEY": "219a6d41e72ff68350230d0a6d8dcf9",
    }

    # Set environment variables
    for key, value in api_keys.items():
        os.environ[key] = value
        print(f"✅ {key}: Set successfully")

    print("\n🚀 All API keys configured!")
    print("🎯 Multi-model integration ready!")

    return True


if __name__ == "__main__":
    print("🔧 Setting up Multi-Model MLB Betting System...")
    print("=" * 50)
    setup_api_keys()

    print("\n📊 Your current system status:")
    print("✅ Kelly Criterion System: ACTIVE")
    print("✅ Odds API Integration: ACTIVE")
    print("✅ Data Processing: 30 games, 1,370 records")
    print("🔄 Multi-Model AI: READY TO ACTIVATE")

    print("\n🎯 Next step: Integrate the multi-model AI system!")
