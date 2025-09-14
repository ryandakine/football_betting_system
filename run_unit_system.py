#!/usr/bin/env python3
"""
UNIT-BASED BETTING SYSTEM LAUNCHER
$100 Bankroll | $5 = 1 Unit | Confidence Scaling
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))


def check_unit_system_requirements():
    """Check if unit-based system requirements are met"""
    print("🔍 Checking unit-based system requirements...")

    # Check required files
    required_files = ["tri_model_api_config.py", "unit_based_betting_system.py", ".env"]

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
            return False

        print("✅ All requirements satisfied")
        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


async def run_unit_based_system():
    """Run the unit-based betting system"""
    print("🚀 UNIT-BASED ULTIMATE BETTING SYSTEM")
    print("=" * 70)
    print("💰 BANKROLL MANAGEMENT:")
    print("   Starting Bankroll: $100")
    print("   Base Unit Size: $5")
    print("   Maximum Bet: 5 units ($25)")
    print()
    print("📊 CONFIDENCE-BASED SCALING:")
    print("   65-70% confidence: 1 unit ($5)")
    print("   70-75% confidence: 2 units ($10)")
    print("   75-80% confidence: 3 units ($15)")
    print("   80-85% confidence: 4 units ($20)")
    print("   85%+ confidence: 5 units ($25)")
    print("=" * 70)

    try:
        # Import the unit-based system
        from unit_based_betting_system import UnitBasedUltimateBettingSystem

        # Initialize with $100 bankroll and $5 units
        system = UnitBasedUltimateBettingSystem(
            bankroll=100.0,  # $100 starting bankroll
            base_unit_size=5.0,  # $5 = 1 unit
            max_units=5,  # Maximum 5 units ($25 max bet)
        )

        print("🔄 Starting unit-based analysis...")

        # Run the unit-based analysis
        result = await system.run_unit_based_analysis()

        if result.get("success"):
            print(f"\n🏆 UNIT-BASED ANALYSIS COMPLETE!")
            print(f"⏱️  Execution Time: {result.get('execution_time', 0):.1f} seconds")
            print(f"🎯 Games Analyzed: {result.get('games_analyzed', 0)}")
            print(f"💡 Recommendations: {result.get('recommendations', 0)}")
            print(f"🎲 Total Units: {result.get('total_units', 0)} units")
            print(f"💰 Total Stake: ${result.get('total_stake', 0):.0f}")
            print(f"📈 Expected Value: ${result.get('total_expected_value', 0):.2f}")

            if result.get("recommendations", 0) > 0:
                print(f"\n🥇 UNIT-BASED RECOMMENDATIONS:")
                recommendations = result.get("recommendations_data", [])
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec['team']} vs {rec['opponent']}")
                    print(
                        f"     🎲 {rec['units']} units (${rec['stake']:.0f}) @ {rec['odds']}"
                    )
                    print(
                        f"     🎯 {rec['ai_confidence']:.1%} confidence ({rec['confidence_tier']})"
                    )
                    print(f"     📈 Expected Value: ${rec['expected_value']:.2f}")
                    print()

            # Ask about continuous monitoring
            print("🔄 Would you like to start continuous unit-based monitoring?")
            print("   This will:")
            print("   • Run analysis every 30 minutes")
            print("   • Scale bet sizes based on AI confidence")
            print("   • Track performance by confidence tier")
            print("   • Alert on high-confidence opportunities")
            print()

            response = input("Start continuous monitoring? (y/n): ").lower().strip()
            if response in ["y", "yes"]:
                print("\n🚀 Starting continuous unit-based monitoring...")
                print("Press Ctrl+C to stop")

                # Continuous monitoring loop
                while True:
                    try:
                        print(
                            f"\n⏰ Running scheduled analysis at {datetime.now().strftime('%H:%M:%S')}..."
                        )
                        result = await system.run_unit_based_analysis()

                        if result.get("success"):
                            units = result.get("total_units", 0)
                            stake = result.get("total_stake", 0)
                            print(
                                f"✅ Analysis complete: {result.get('recommendations', 0)} recommendations, {units} units (${stake:.0f})"
                            )
                        else:
                            print(f"❌ Analysis failed: {result.get('error')}")

                        print("⏰ Next analysis in 30 minutes...")
                        await asyncio.sleep(30 * 60)  # 30 minutes

                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        print(f"❌ Monitoring error: {e}")
                        await asyncio.sleep(60)  # Wait 1 minute before retrying
            else:
                print("✅ Unit-based analysis complete. Run again anytime!")

        else:
            print(f"❌ Unit-based analysis failed: {result.get('error')}")
            print("This might be due to:")
            print("  • API rate limits")
            print("  • Network connectivity issues")
            print("  • No games meeting 65%+ confidence threshold")

    except ImportError as import_error:
        print(f"❌ Import Error: {import_error}")
        print("\n🔧 This means the unit-based system file is missing.")
        print("Please ensure 'unit_based_betting_system.py' is in this directory.")

    except Exception as error:
        print(f"❌ System Error: {error}")
        print("\n🔍 Troubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Verify API keys in .env file")
        print("  3. Ensure sufficient API credits")


def main():
    """Main launcher function"""
    print(f"🕐 Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check system requirements
    if not check_unit_system_requirements():
        print("\n❌ System requirements not met. Please fix the issues above.")
        return

    # Run the unit-based system
    try:
        asyncio.run(run_unit_based_system())
    except KeyboardInterrupt:
        print("\n⏹️ Unit-based system stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        print("🏁 Unit-Based Ultimate Betting System shutdown complete")


if __name__ == "__main__":
    main()
