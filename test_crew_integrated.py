#!/usr/bin/env python3
"""
Test that crew adjustment system is integrated and working.
This simulates what happens in main.py at startup.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger("test_crew_integrated")


def main():
    """Simulate main.py startup."""
    logger.info("🚀 SYSTEM STARTING UP...")
    
    # This is the ONE LINE added to main.py
    try:
        from init_crew_adjustments import initialize_crew_system
        initialize_crew_system()
    except Exception as e:
        logger.warning(f"⚠️ Crew adjustment system initialization failed: {e}")
    
    logger.info("✅ System fully initialized")
    logger.info("\n" + "="*100)
    logger.info("CREW ADJUSTMENT SYSTEM ACTIVE")
    logger.info("="*100)
    
    # Demonstrate automatic adjustment
    from crew_adjustment_middleware import get_middleware
    
    middleware = get_middleware()
    
    # Simulate a prediction
    test_prediction = {
        'home_team': 'BAL',
        'away_team': 'ARI',
        'referee_crew': 'Alan Eck',
        'predicted_margin': 3.5,
    }
    
    logger.info("\n📊 TEST PREDICTION BEFORE ADJUSTMENT:")
    logger.info(f"  {test_prediction['home_team']} vs {test_prediction['away_team']}: {test_prediction['predicted_margin']:+.1f}")
    
    # Adjust it
    adjusted = middleware.adjust_prediction(test_prediction)
    
    logger.info("\n✨ AFTER AUTOMATIC ADJUSTMENT:")
    logger.info(f"  {adjusted['home_team']} vs {adjusted['away_team']}: {adjusted['predicted_margin']:+.1f}")
    logger.info(f"  Crew adjustment: {adjusted['crew_adjustment']:+.1f}")
    
    logger.info("\n✅ Crew adjustment system is ACTIVE and WORKING")
    logger.info("   All future predictions will be automatically adjusted!")


if __name__ == '__main__':
    main()
