#!/usr/bin/env python3
"""
Quick start script for Women's College Basketball Betting System
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from main_analyzer import UnifiedWomensCollegeBasketballAnalyzer


async def main():
    """Run the complete analysis."""
    print("🏀 Women's College Basketball Betting System")
    print("=" * 60)

    # Get bankroll from command line or use default
    bankroll = float(sys.argv[1]) if len(sys.argv) > 1 else 50000.0

    print(f"💰 Bankroll: ${bankroll:,.2f}")
    print()

    # Initialize analyzer
    analyzer = UnifiedWomensCollegeBasketballAnalyzer(bankroll=bankroll)

    # Run complete analysis
    results = await analyzer.run_complete_analysis()

    # Display results
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)

    summary = results.get('summary', {})
    print(f"Games Analyzed: {summary.get('games_analyzed', 0)}")
    print(f"High Edge Opportunities: {summary.get('high_edge_opportunities', 0)}")
    print(f"Average Edge: {summary.get('average_edge', 0):.2%}")
    print(f"Parlays Generated: {summary.get('parlays_generated', 0)}")

    # Show high edge games
    if results.get('high_edge_games'):
        print("\n🔥 HIGH EDGE GAMES:")
        for i, game in enumerate(results['high_edge_games'][:5], 1):
            print(f"\n{i}. {game.get('away_team')} @ {game.get('home_team')}")
            print(f"   Edge: {game.get('total_edge', 0):.2%}")
            print(f"   Confidence: {game.get('confidence', 0):.2%}")
            rec = game.get('unified_recommendation', {})
            print(f"   Action: {rec.get('final_action', 'PASS')}")

    # Show top parlays
    if results.get('parlays'):
        print("\n🎰 TOP PARLAYS:")
        for i, parlay in enumerate(results['parlays'][:3], 1):
            print(f"\n{i}. {parlay.get('num_legs')}-Leg Parlay")
            print(f"   Combined Edge: {parlay.get('combined_edge', 0):.2%}")
            print(f"   Stake: ${parlay.get('stake', 0):.2f}")
            print(f"   Potential Payout: ${parlay.get('potential_payout', 0):.2f}")
            print(f"   Legs:")
            for leg in parlay.get('legs', []):
                print(f"     - {leg.get('matchup')}: {leg.get('pick')}")

    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)
