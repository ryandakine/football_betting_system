#!/usr/bin/env python3
"""
Start NFL Live Game Tracking Service
====================================
Starts the NFL live game tracker as an autonomous background service.
Automatically monitors live games and continuously improves betting models.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from nfl_live_tracker import NFLLiveGameTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/nfl_live_tracking.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("nfl_live_tracking")

class NFLTrackingService:
    """Service wrapper for NFL live game tracking."""

    def __init__(self):
        self.tracker = NFLLiveGameTracker()
        self.running = False

    async def start_service(self):
        """Start the NFL tracking service."""
        logger.info("🚀 Starting NFL Live Game Tracking Service...")

        # Create logs directory
        Path("logs").mkdir(exist_ok=True)

        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"📡 Received signal {signum}, shutting down...")
            self.running = False
            self.tracker.stop_tracking()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start tracking
        self.tracker.start_live_tracking()
        self.running = True

        logger.info("✅ NFL Live Tracking Service started successfully!")
        logger.info("📊 Monitoring live NFL games and continuously learning...")
        logger.info("🎯 System will automatically:")
        logger.info("   • Track live game scores and stats")
        logger.info("   • Make real-time predictions")
        logger.info("   • Learn from game outcomes")
        logger.info("   • Update models continuously")

        # Keep service running
        while self.running:
            try:
                # Get current status
                status = self.tracker.get_tracking_status()
                insights = self.tracker.get_learning_insights()

                # Log status every 5 minutes
                logger.info(f"📈 Status: {status['active_games']} active games, "
                          f"{status['learning_stats']['games_processed']} processed, "
                          f"Accuracy: {insights['average_accuracy']:.3f}")

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"❌ Service error: {e}")
                await asyncio.sleep(60)

        logger.info("🏁 NFL Live Tracking Service stopped")

    def get_status(self):
        """Get current service status."""
        return self.tracker.get_tracking_status()

def main():
    """Main entry point."""
    print("🏈 NFL Live Game Tracking Service - 2025 Season")
    print("🎯 Ready for September 2025 NFL season kickoff!")
    print("=" * 50)

    service = NFLTrackingService()

    try:
        asyncio.run(service.start_service())
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down NFL Live Tracking Service...")
        print("💤 System will be ready for 2025 NFL season games!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal service error: {e}")
    finally:
        service.tracker.stop_tracking()
        print("✅ NFL Live Tracking Service stopped")

if __name__ == "__main__":
    main()
