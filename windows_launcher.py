# windows_launcher.py
import asyncio
import datetime
import json
import logging
import os

from gold_standard_main import GoldStandardMLBSystem

# Configure logging to file and console
log_file = f"logs\\fixed_system_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_env():
    """Load environment variables from .env file"""
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value
    logger.info("[INFO] Loading environment variables from .env file")


async def main():
    logger.info("=" * 60)
    logger.info("FIXED GOLD STANDARD MLB BETTING SYSTEM")
    logger.info("=" * 60)
    logger.info("[SETUP] Logging setup complete - log file: %s", log_file)

    load_env()
    logger.info("[STARTUP] Starting Fixed Gold Standard System...")

    system = GoldStandardMLBSystem(
        bankroll=500.0, base_unit_size=5.0, max_units=5, confidence_threshold=0.60
    )

    try:
        results = await system.run_gold_standard_pipeline()
        logger.info("[SUCCESS] System execution completed successfully!")
        logger.info("    - Analyzed: %d opportunities", results["total_opportunities"])
        logger.info(
            "    - Generated: %d recommendations", len(results["recommendations"])
        )
        logger.info("    - Execution time: %.2f seconds", results["execution_time"])
        if results["recommendations"]:
            logger.info(
                "    - Top recommendation: %s (EV: $%.2f)",
                results["recommendations"][0]["selection"],
                results["recommendations"][0]["expected_value"],
            )
        logger.info("[COMPLETE] System execution completed")
    except KeyboardInterrupt:
        logger.warning("[INTERRUPT] Execution interrupted by user")
    except Exception as e:
        logger.error("[ERROR] Main execution failed: %s", str(e), exc_info=True)

    logger.info("[EXIT] Program finished with code: 0")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
