#!/usr/bin/env python3
"""
NFL Live Game Fetcher (Refactored)
Simple wrapper around unified GameFetcher
"""
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.game_fetcher import GameFetcher, GameFetcherError


def main():
    """Fetch NFL games for this week"""
    print("🏈 NFL LIVE GAMES FETCHER")
    print("="*100)

    try:
        # Create NFL game fetcher
        fetcher = GameFetcher(sport='nfl')

        # Run complete workflow
        games = fetcher.run(save=True, display=True)

        print(f"\n✅ NFL games ready for prediction!")
        print(f"   Games fetched: {len(games)}")
        print(f"   Saved to: {fetcher.paths['live_games']}")
        print(f"\n   Next step: python run_nfl_12model_deepseek_v2.py")
        print("="*100)

    except GameFetcherError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check API key: echo $ODDS_API_KEY")
        print("   2. Get new key from: https://the-odds-api.com/")
        print("   3. Export key: export ODDS_API_KEY='your_key_here'")
        sys.exit(1)


if __name__ == '__main__':
    main()
