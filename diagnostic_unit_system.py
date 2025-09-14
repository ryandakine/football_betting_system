#!/usr/bin/env python3
"""
DIAGNOSTIC UNIT SYSTEM
Shows all confidence levels to test AI analysis
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


async def run_diagnostic():
    """Run diagnostic to show all AI confidence levels"""
    print("🔍 DIAGNOSTIC: Showing ALL AI Confidence Levels")
    print("=" * 60)

    try:
        from unit_based_betting_system import UnitBasedUltimateBettingSystem

        # Create system but we'll override the confidence threshold
        system = UnitBasedUltimateBettingSystem(
            bankroll=100.0, base_unit_size=5.0, max_units=5
        )

        print("🔄 Fetching games and analyzing with AI...")

        # Fetch games
        games = await system._fetch_games()
        print(f"📊 Found {len(games)} games")

        if not games:
            print("❌ No games available from API")
            return

        print("\n🤖 AI CONFIDENCE ANALYSIS (All Levels):")
        print("-" * 60)

        # Analyze each game and show confidence regardless of threshold
        for i, game in enumerate(games[:5], 1):
            home_team = game.get("home_team", "Home")
            away_team = game.get("away_team", "Away")

            print(f"{i}. {away_team} @ {home_team}")

            # Get AI analysis
            analysis = await system.ai_analyzer.analyze_game_real(game)

            confidence = analysis.get("confidence", 0)
            recommended_team = analysis.get("recommended_team", "Unknown")

            # Show confidence with color coding
            if confidence >= 0.85:
                tier = "🔥 MAXIMUM (5 units)"
            elif confidence >= 0.80:
                tier = "🚀 VERY HIGH (4 units)"
            elif confidence >= 0.75:
                tier = "⭐ HIGH (3 units)"
            elif confidence >= 0.70:
                tier = "✅ MEDIUM (2 units)"
            elif confidence >= 0.65:
                tier = "🟡 LOW (1 unit)"
            else:
                tier = "❌ BELOW THRESHOLD"

            print(f"   🎯 {confidence:.1%} confidence → {recommended_team}")
            print(f"   📊 {tier}")
            print(f"   🤖 Claude: {analysis.get('claude_analysis', 'N/A')[:50]}...")
            print(f"   🤖 OpenAI: {analysis.get('openai_analysis', 'N/A')[:50]}...")
            print()

        # Show what would qualify
        print("📋 SUMMARY:")
        print("   • Your system requires 65%+ confidence (conservative)")
        print("   • If no bets shown above = AI found no games with 65%+ confidence")
        print("   • This protects your bankroll from uncertain bets")
        print("   • Consider running during peak betting hours (afternoon/evening)")

        print("\n🎯 RECOMMENDATIONS:")
        print("   1. ✅ KEEP 65% threshold (conservative is good)")
        print("   2. 🔄 Try continuous monitoring to catch opportunities")
        print("   3. ⏰ Run during 2-6 PM when more games/odds available")
        print("   4. 📊 Today's low confidence might mean unclear matchups")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")


if __name__ == "__main__":
    asyncio.run(run_diagnostic())
