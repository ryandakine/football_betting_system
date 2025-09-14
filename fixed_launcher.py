#!/usr/bin/env python3
"""
Simple Fixed MLB System Launcher
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


# Load environment variables from .env file
def load_env():
    env_file = Path(".env")
    if env_file.exists():
        print("📋 Loading environment variables from .env file")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    else:
        print("⚠️  No .env file found - using environment variables")


def setup_logging():
    """Setup logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"fixed_system_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"📝 Logging setup complete - log file: {log_file}")
    return logger


async def main():
    """Main launcher function"""
    print("🚀 Fixed Gold Standard MLB Betting System")
    print("=" * 50)

    # Setup
    logger = setup_logging()
    load_env()

    # Create directories
    for directory in ["logs", "results", "data"]:
        Path(directory).mkdir(exist_ok=True)

    try:
        logger.info("🏆 Starting Fixed Gold Standard System...")

        # Import and run the fixed system
        from fixed_gold_standard_mlb_system import FixedGoldStandardMLBSystem

        system = FixedGoldStandardMLBSystem(bankroll=500.0, base_unit_size=5.0)
        results = await system.run_fixed_pipeline()

        # Display results summary
        if results["recommendations"]:
            logger.info("🎯 System execution completed successfully!")
            logger.info(
                f"   • Analyzed: {results['total_opportunities']} opportunities"
            )
            logger.info(
                f"   • Generated: {len(results['recommendations'])} recommendations"
            )
            logger.info(f"   • Execution time: {results['execution_time']:.2f} seconds")

            if results["recommendations"]:
                top_rec = results["recommendations"][0]
                logger.info(
                    f"   • Top recommendation: {top_rec.selection} (EV: ${top_rec.expected_value:.2f})"
                )
        else:
            logger.warning("⚠️  No recommendations generated - check API connections")

        logger.info("✅ System execution completed")
        return 0

    except ImportError as e:
        logger.error(f"❌ Failed to import fixed system: {e}")
        logger.info(
            "💡 Please ensure 'fixed_gold_standard_mlb_system.py' is in the current directory"
        )
        return 1
    except Exception as e:
        logger.error(f"❌ System execution failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
