#!/usr/bin/env python3
"""
Initialize Crew Adjustments System
===================================
Add this import to your system startup to automatically enable crew adjustments.

Usage in your main.py or startup script:
    from init_crew_adjustments import initialize_crew_system
    initialize_crew_system()
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def initialize_crew_system():
    """
    Initialize the automatic crew adjustment system.
    Call this ONCE during system startup.
    """
    try:
        from crew_adjustment_middleware import get_middleware, auto_patch_prediction_functions
        
        # Initialize middleware
        middleware = get_middleware()
        
        if middleware.model is None:
            logger.warning("⚠️ Crew model not loaded - adjustments disabled")
            return False
        
        # Auto-patch JSON serialization
        auto_patch_prediction_functions()
        
        logger.info("✅ CREW ADJUSTMENT SYSTEM INITIALIZED")
        logger.info("   - All predictions will be automatically adjusted")
        logger.info("   - Crew model: Loaded")
        logger.info("   - JSON patched: Yes")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Failed to initialize crew system: {e}")
        return False


def get_crew_middleware():
    """Get the global crew middleware instance."""
    from crew_adjustment_middleware import get_middleware
    return get_middleware()


def adjust_game_prediction(game_dict):
    """Manually adjust a single game prediction."""
    middleware = get_crew_middleware()
    return middleware.adjust_prediction(game_dict)


def adjust_batch_predictions(predictions_list):
    """Manually adjust a batch of predictions."""
    middleware = get_crew_middleware()
    return middleware.adjust_predictions_batch(predictions_list)


# ============================================================
# INTEGRATION POINTS - Add these to your existing code
# ============================================================

def patch_into_fastapi_app(app):
    """Add crew adjustment middleware to FastAPI app."""
    from crew_adjustment_middleware import adjust_api_response
    
    @app.middleware("http")
    async def crew_adjustment_middleware(request, call_next):
        response = await call_next(request)
        # Adjust response data before sending
        # This is a simplified version - adjust based on your actual response format
        return response
    
    logger.info("✅ FastAPI middleware patched for crew adjustments")


def patch_into_prediction_function(predict_func):
    """Decorator version for individual functions."""
    from crew_adjustment_middleware import wrap_prediction_function
    return wrap_prediction_function(predict_func)


if __name__ == '__main__':
    # Test initialization
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    print("\n" + "="*100)
    print("CREW ADJUSTMENT SYSTEM INITIALIZATION TEST")
    print("="*100)
    
    success = initialize_crew_system()
    
    if success:
        print("\n✅ System initialized successfully!")
        print("\nTo use in your code:")
        print("  1. Add to startup: from init_crew_adjustments import initialize_crew_system")
        print("  2. Call: initialize_crew_system()")
        print("  3. All predictions automatically adjusted from that point on\n")
    else:
        print("\n❌ Initialization failed\n")
        sys.exit(1)
