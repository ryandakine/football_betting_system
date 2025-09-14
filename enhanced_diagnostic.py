#!/usr/bin/env python3
"""
ENHANCED DIAGNOSTIC
Shows individual Claude and OpenAI outputs to debug confidence combination
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))


async def run_enhanced_diagnostic():
    """Enhanced diagnostic showing individual AI outputs"""
    print("🔍 ENHANCED DIAGNOSTIC: Individual AI Analysis")
    print("=" * 70)

    try:
        from unit_based_betting_system import UnitBasedUltimateBettingSystem

        system = UnitBasedUltimateBettingSystem(
            bankroll=100.0, base_unit_size=5.0, max_units=5
        )

        print("🔄 Fetching games and analyzing individual AI outputs...")

        games = await system._fetch_games()
        print(f"📊 Found {len(games)} games")

        if not games:
            print("❌ No games available")
            return

        print("\n🤖 INDIVIDUAL AI ANALYSIS BREAKDOWN:")
        print("=" * 70)

        for i, game in enumerate(games[:3], 1):  # Just 3 games for detailed analysis
            home_team = game.get("home_team", "Home")
            away_team = game.get("away_team", "Away")

            print(f"\n{i}. {away_team} @ {home_team}")
            print("-" * 50)

            # Build the prompt manually to see what's being sent
            prompt = system.ai_analyzer._build_prompt(game)
            print(f"📝 PROMPT LENGTH: {len(prompt)} characters")

            # Get individual Claude analysis
            print("\n🧠 CLAUDE ANALYSIS:")
            claude_result = await system.ai_analyzer._analyze_with_claude(prompt)
            if isinstance(claude_result, Exception):
                print(f"   ❌ Claude Error: {claude_result}")
            else:
                claude_prob = claude_result.get("win_probability", "N/A")
                claude_team = claude_result.get("favored_team", "N/A")
                print(f"   🎯 Probability: {claude_prob}")
                print(f"   👥 Favored Team: {claude_team}")
                print(f"   📊 Raw Response: {str(claude_result)[:200]}...")

            # Get individual OpenAI analysis
            print("\n🤖 OPENAI ANALYSIS:")
            openai_result = await system.ai_analyzer._analyze_with_openai(prompt)
            if isinstance(openai_result, Exception):
                print(f"   ❌ OpenAI Error: {openai_result}")
            else:
                openai_prob = openai_result.get("win_probability", "N/A")
                openai_team = openai_result.get("favored_team", "N/A")
                print(f"   🎯 Probability: {openai_prob}")
                print(f"   👥 Favored Team: {openai_team}")
                print(f"   📊 Raw Response: {str(openai_result)[:200]}...")

            # Show combination logic
            print("\n🔗 COMBINATION LOGIC:")
            if not isinstance(claude_result, Exception) and not isinstance(
                openai_result, Exception
            ):
                claude_prob_val = claude_result.get("win_probability", 0.65)
                openai_prob_val = openai_result.get("win_probability", 0.65)

                # Show the exact calculation
                combined = (claude_prob_val * 0.55) + (openai_prob_val * 0.45)

                print(
                    f"   📐 Claude: {claude_prob_val} * 0.55 = {claude_prob_val * 0.55:.3f}"
                )
                print(
                    f"   📐 OpenAI: {openai_prob_val} * 0.45 = {openai_prob_val * 0.45:.3f}"
                )
                print(f"   📐 Combined: {combined:.3f} ({combined:.1%})")

                # Test the actual combination method
                combined_result = system.ai_analyzer._combine_results(
                    claude_result, openai_result, game
                )
                final_confidence = combined_result.get("confidence", 0)
                print(
                    f"   🎯 FINAL RESULT: {final_confidence:.3f} ({final_confidence:.1%})"
                )

                # Highlight the issue
                if abs(final_confidence - 0.612) < 0.001:
                    print(f"   🚨 ISSUE DETECTED: Getting same 61.2% result!")
                    print(f"   🔍 This suggests a bug in the combination logic")

            print("\n" + "=" * 70)

        print("\n🔍 DIAGNOSIS:")
        print("   If all games show ~61.2% confidence:")
        print("   1. ❌ Bug in confidence combination logic")
        print("   2. ❌ Fallback values being used consistently")
        print("   3. ❌ JSON parsing failing, using defaults")

        print("\n💡 POTENTIAL FIXES:")
        print("   1. 🔧 Fix JSON parsing in AI responses")
        print("   2. 🔧 Improve combination algorithm")
        print("   3. 🔧 Add more variation in AI prompts")
        print("   4. 🔧 Use different default values for testing")

    except Exception as e:
        print(f"❌ Enhanced diagnostic failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_enhanced_diagnostic())
