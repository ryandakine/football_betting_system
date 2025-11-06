"""
Data Enforcement Layer - BLOCKS simulated data
Raises hard errors if simulated/test data is detected
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)

class DataEnforcementError(Exception):
    """Raised when simulated data is detected"""
    pass

class DataEnforcer:
    """
    Enforces real data only - no simulated data allowed
    """
    
    BLOCKED_KEYWORDS = [
        'simulated',
        'test',
        'dummy',
        'fake',
        'mock',
        'placeholder',
        'example',
        'demo',
        'synthetic',
        'generated'
    ]
    
    @staticmethod
    def validate_data_source(data_dict):
        """Raise error if simulated data is detected"""
        if not isinstance(data_dict, dict):
            return True
        
        # Check data_source field
        source = data_dict.get('data_source', '').lower()
        for blocked in DataEnforcer.BLOCKED_KEYWORDS:
            if blocked in source:
                raise DataEnforcementError(
                    f"❌ BLOCKED: Simulated data detected (data_source: {source}). "
                    f"Only real data allowed. PERIOD."
                )
        
        # Check ai_warning field
        warning = data_dict.get('ai_warning', '').lower()
        if 'simulated' in warning or 'not real' in warning:
            raise DataEnforcementError(
                f"❌ BLOCKED: Simulated data warning detected. Use real data only."
            )
        
        return True
    
    @staticmethod
    def validate_game_data(games_list):
        """Validate list of games contains no simulated data"""
        if not games_list:
            raise DataEnforcementError("❌ No game data provided")
        
        for game in games_list:
            DataEnforcer.validate_data_source(game)
        
        return True
    
    @staticmethod
    def block_if_test_mode():
        """Block execution if we're in test mode with simulated data"""
        test_mode_indicators = [
            'SIMULATED_DATA',
            'TEST_MODE',
            'DEMO_MODE'
        ]
        
        for indicator in test_mode_indicators:
            if os.environ.get(indicator) == 'true':
                raise DataEnforcementError(
                    f"❌ {indicator} is enabled. Use REAL data only. "
                    f"Remove test environment variables."
                )


def enforce_real_data_only(func):
    """Decorator that blocks any function using simulated data"""
    def wrapper(*args, **kwargs):
        # Check if any arg contains simulated data
        for arg in args:
            if isinstance(arg, dict):
                try:
                    DataEnforcer.validate_data_source(arg)
                except DataEnforcementError as e:
                    logger.critical(str(e))
                    raise
        
        # Check kwargs
        for key, val in kwargs.items():
            if isinstance(val, dict):
                try:
                    DataEnforcer.validate_data_source(val)
                except DataEnforcementError as e:
                    logger.critical(str(e))
                    raise
        
        return func(*args, **kwargs)
    return wrapper


if __name__ == "__main__":
    print("Data Enforcement: Active")
    print("Simulated data: BLOCKED")
    print("Real data only: REQUIRED")
