#!/usr/bin/env python3
"""
Start Automatic Outcome Tracking Service
========================================
Starts the automatic outcome tracker as a background service.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from automatic_outcome_tracker import AutomaticOutcomeTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/automatic_tracking.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("automatic_tracking")


class TrackingService:
    """Service wrapper for automatic outcome tracking."""

    def __init__(self):
        self.tracker = AutomaticOutcomeTracker()
        self.running = False

    async def start_service(self):
        """Start the tracking service."""
        logger.info("🚀 Starting Automatic Outcome Tracking Service...")

        # Create logs directory
        Path("logs").mkdir(exist_ok=True)

        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}, shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.running = True

        try:
            # Start tracking
            await self.tracker.start_tracking()
        except Exception as e:
            logger.error(f"❌ Service error: {e}")
        finally:
            await self.tracker.stop_tracking()
            logger.info("🔚 Service stopped")


async def main():
    """Main entry point."""
    service = TrackingService()
    await service.start_service()


if __name__ == "__main__":
    asyncio.run(main())
