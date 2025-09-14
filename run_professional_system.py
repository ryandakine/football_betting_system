#!/usr/bin/env python3
"""
PROFESSIONAL ULTIMATE BETTING SYSTEM LAUNCHER
Fixed version - ready to run
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))


def check_environment():
    """Check if all required files and dependencies are ready"""
    print("🔍 Checking system requirements...")

    # Check required files
    required_files = ["tri_model_api_config.py", ".env"]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    # Check API keys
    try:
        from tri_model_api_config import get_trimodel_api_keys

        api_keys = get_trimodel_api_keys()

        required_keys = ["odds_api", "claude", "openai"]
        missing_keys = [key for key in required_keys if not api_keys.get(key)]

        if missing_keys:
            print(f"❌ Missing API keys: {missing_keys}")
            print("Please check your .env file and ensure all API keys are set.")
            return False

        print("✅ All requirements satisfied")
        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


async def run_professional_system():
    """Run the professional ultimate betting system"""
    print("🚀 PROFESSIONAL ULTIMATE TRI-MODEL BETTING SYSTEM")
    print("=" * 70)
    print("🎯 Features Enabled:")
    print("  ✅ Real AI Analysis (Claude + OpenAI)")
    print("  ✅ Live Odds Monitoring")
    print("  ✅ Historical Performance Tracking")
    print("  ✅ Smart Alerts & Notifications")
    print("  ✅ Automated Kelly Criterion Position Sizing")
    print("=" * 70)

    try:
        # Import the professional system
        from professional_ultimate_system import ProfessionalUltimateBettingSystem

        # Initialize with professional features
        system = ProfessionalUltimateBettingSystem(bankroll=1000.0)

        print("🔄 Starting professional analysis...")

        # Run the complete professional analysis
        result = await system.run_daily_analysis()

        if result.get("success"):
            print(f"\n🏆 PROFESSIONAL ANALYSIS COMPLETE!")
            print(f"⏱️  Execution Time: {result.get('execution_time', 0):.2f} seconds")
            print(f"🎯 Games Analyzed: {result.get('games_analyzed', 0)}")
            print(f"💡 Recommendations: {result.get('recommendations', 0)}")
            print(
                f"💰 Total Expected Value: ${result.get('total_expected_value', 0):.2f}"
            )

            if result.get("recommendations", 0) > 0:
                print(f"\n📊 PROFESSIONAL RECOMMENDATIONS:")
                recommendations = result.get("recommendations_data", [])
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"  {i}. {rec['team']} @ {rec['odds']}")
                    print(
                        f"     💰 Stake: ${rec['stake']} | EV: ${rec['expected_value']:.2f}"
                    )
                    print(f"     🤖 AI Confidence: {rec['ai_confidence']:.1%}")
                    if rec.get("claude_analysis"):
                        print(f"     🧠 Claude: {rec['claude_analysis'][:100]}...")
                    print()

            # Ask about continuous monitoring
            print("🔄 Would you like to start continuous monitoring?")
            print("   This will:")
            print("   • Monitor odds changes every 15 minutes")
            print("   • Send alerts for significant movements")
            print("   • Run automated analysis at 10 AM and 6 PM daily")
            print()

            response = input("Start continuous monitoring? (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                print("\n🚀 Starting continuous monitoring...")
                print("Press Ctrl+C to stop")
                await system.run_monitoring_loop()
            else:
                print("✅ Professional analysis complete. Run again anytime!")

        else:
            print(f"❌ Professional analysis failed: {result.get('error')}")
            print("This might be due to:")
            print("  • API rate limits")
            print("  • Network connectivity issues")
            print("  • Invalid API keys")
            print("Try running again in a few minutes.")

    except ImportError as import_error:
        print(f"❌ Import Error: {import_error}")
        print("\n🔧 This usually means the professional system file is missing.")
        print("Please ensure 'professional_ultimate_system.py' is in this directory.")
        print("You can create it by saving the professional system code provided.")

    except Exception as error:
        print(f"❌ System Error: {error}")
        print("\n🔍 Troubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Verify API keys in .env file")
        print("  3. Ensure all dependencies are installed")


def main():
    """Main launcher function"""
    print(f"🕐 Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check system requirements
    if not check_environment():
        print("\n❌ System requirements not met. Please fix the issues above.")
        return

    # Run the professional system
    try:
        asyncio.run(run_professional_system())
    except KeyboardInterrupt:
        print("\n⏹️ System stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        print("🏁 Professional Ultimate Betting System shutdown complete")


if __name__ == "__main__":
    main()
