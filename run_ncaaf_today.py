#!/usr/bin/env python3
"""
Quick runner for today's NCAAF (College Football) games
Focuses on getting predictions without API errors
"""

import asyncio
import logging
from datetime import datetime
from football_production_main import FootballProductionBettingSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

async def run_ncaaf_today():
    """Run NCAAF predictions for today's games"""
    
    logger.info("🏈 COLLEGE FOOTBALL BETTING ANALYSIS - SATURDAY")
    logger.info("=" * 60)
    logger.info(f"Date: {datetime.now().strftime('%B %d, %Y')}")
    logger.info("=" * 60)
    
    # Initialize the system for college football
    system = FootballProductionBettingSystem(
        bankroll=1000.0,           # Starting bankroll
        max_exposure_pct=0.10,      # Max 10% exposure
        sport_type="ncaaf",         # College football
        test_mode=False,            # Use real odds data
        fake_money=True,            # Don't place real bets
        no_api=False                # Use API for real odds
    )
    
    # Run the full pipeline
    success = await system.run_production_pipeline()
    
    if success:
        logger.info("\n✅ NCAAF analysis completed successfully!")
        logger.info("📋 Check the production_reports folder for detailed results")
    else:
        logger.error("\n❌ NCAAF analysis failed - check logs for details")
    
    return success

def main():
    """Main entry point"""
    print("\n" + "🏈" * 30)
    print("    COLLEGE FOOTBALL SATURDAY PREDICTOR")
    print("🏈" * 30 + "\n")
    
    # Run the async function
    success = asyncio.run(run_ncaaf_today())
    
    if success:
        print("\n✨ Ready for college football betting!")
        print("📊 Review the recommendations above before placing any bets")
        print("💡 Remember: Only bet what you can afford to lose\n")
    else:
        print("\n⚠️ Something went wrong - please check the logs")

if __name__ == "__main__":
    main()
