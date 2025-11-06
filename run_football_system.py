#!/usr/bin/env python3
"""
Football Betting System Launcher
Simple script to run the football betting system with different configurations.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Import the football system components
from football_production_main import FootballProductionBettingSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print the system banner."""
    print("=" * 60)
    print("🏈 FOOTBALL BETTING SYSTEM")
    print("=" * 60)
    print("NFL & College Football Predictive Betting System")
    print("Adapted from proven MLB betting architecture")
    print("=" * 60)


def print_usage():
    """Print usage instructions."""
    print("\n📖 USAGE:")
    print("python run_football_system.py [sport] [bankroll] [options]")
    print("\n🏈 SPORTS:")
    print("  nfl    - National Football League")
    print("  ncaaf  - College Football (NCAA)")
    print("\n💰 BANKROLL:")
    print("  Amount in dollars (e.g., 1000, 5000)")
    print("\n⚙️ OPTIONS:")
    print("  --test          - Run in test mode with sample data")
    print("  --verbose       - Enable verbose logging")
    print("  --fake-money    - Use fake money for testing (default: True)")
    print("  --real-money    - Use real money mode (WARNING: places actual bets)")
    print("  --no-api        - Skip API calls, use mock data (for testing)")
    print("  --help          - Show this help message")
    print("\n📋 EXAMPLES:")
    print("  python run_football_system.py nfl 1000")
    print("  python run_football_system.py ncaaf 5000 --test")
    print("  python run_football_system.py nfl 2000 --verbose")
    print("  python run_football_system.py nfl 1000 --fake-money")
    print("  python run_football_system.py nfl 1000 --real-money")
    print("  python run_football_system.py nfl 1000 --no-api --fake-money")


def validate_args(args):
    """Validate command line arguments."""
    if len(args) < 2:
        print("❌ Error: Sport type is required")
        print_usage()
        return False

    sport = args[1].lower()
    if sport not in ["nfl", "ncaaf"]:
        print(f"❌ Error: Invalid sport '{sport}'. Use 'nfl' or 'ncaaf'")
        return False

    # Bankroll validation
    bankroll = 1000.0  # default
    if len(args) >= 3:
        try:
            bankroll = float(args[2])
            if bankroll <= 0:
                print("❌ Error: Bankroll must be positive")
                return False
        except ValueError:
            print("❌ Error: Bankroll must be a number")
            return False

    return True


async def run_system(sport: str, bankroll: float, test_mode: bool = False, verbose: bool = False, fake_money: bool = True, no_api: bool = False):
    """Run the football betting system."""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("🚀 Starting Football Betting System")
    logger.info(f"   Sport: {sport.upper()}")
    logger.info(f"   Bankroll: ${bankroll:,.2f}")
    logger.info(f"   Test Mode: {test_mode}")
    logger.info(f"   Fake Money Mode: {fake_money}")
    logger.info(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Initialize the system
        system = FootballProductionBettingSystem(
            bankroll=bankroll,
            max_exposure_pct=0.10,
            sport_type=sport,
            test_mode=test_mode,
            fake_money=fake_money,
            no_api=no_api
        )

        # Run the production pipeline
        success = await system.run_production_pipeline()

        if success:
            logger.info("🎉 Football betting system completed successfully!")
            print("\n✅ SYSTEM COMPLETED SUCCESSFULLY!")
            print("📊 Check the logs and output files for results.")
        else:
            logger.error("❌ Football betting system failed!")
            print("\n❌ SYSTEM FAILED!")
            print("🔍 Check the logs for error details.")

        return success

    except KeyboardInterrupt:
        logger.info("⚠️ System interrupted by user")
        print("\n⚠️ System interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ System error: {e}", exc_info=True)
        print(f"\n❌ System error: {e}")
        return False


def main():
    """Main function."""
    print_banner()

    # Parse command line arguments
    args = sys.argv

    # Check for help
    if "--help" in args or "-h" in args:
        print_usage()
        return

    # Validate arguments
    if not validate_args(args):
        return

    # Extract arguments
    sport = args[1].lower()
    bankroll = float(args[2]) if len(args) >= 3 else 1000.0
    test_mode = "--test" in args
    verbose = "--verbose" in args
    fake_money = "--fake-money" in args or ("--real-money" not in args)  # Default to fake money
    real_money = "--real-money" in args
    no_api = "--no-api" in args

    # Validate money mode
    if real_money and fake_money:
        print("❌ Error: Cannot specify both --fake-money and --real-money")
        return

    # Show configuration
    print("\n⚙️ CONFIGURATION:")
    print(f"   Sport: {sport.upper()}")
    print(f"   Bankroll: ${bankroll:,.2f}")
    print(f"   Test Mode: {test_mode}")
    print(f"   Verbose: {verbose}")
    print(f"   Money Mode: {'FAKE MONEY TESTING' if fake_money else 'REAL MONEY BETTING'}")

    # Confirm before running real money mode
    if real_money:
        print("\n🚨 WARNING: REAL MONEY MODE SELECTED!")
        print("   This will place actual bets with real money!")
        print("   Make sure you have configured your API keys correctly.")
        print("   Make sure you understand the risks involved.")
        response = input("   Are you sure you want to continue? (yes/N): ").strip().lower()
        if response not in ["yes", "y"]:
            print("❌ System cancelled by user")
            return

    if fake_money:
        print("\n💰 FAKE MONEY MODE:")
        print("   No real bets will be placed.")
        print("   Focus is on testing predictions and performance.")

    print("\n🚀 Starting system...")
    print("=" * 60)

    # Run the system
    success = asyncio.run(run_system(sport, bankroll, test_mode, verbose, fake_money, no_api))

    # Final status
    print("=" * 60)
    if success:
        print("🎉 FOOTBALL BETTING SYSTEM COMPLETED!")
    else:
        print("❌ FOOTBALL BETTING SYSTEM FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
