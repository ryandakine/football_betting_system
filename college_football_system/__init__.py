"""
Unified College Football Betting System
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'UnifiedCollegeFootballAnalyzer':
        from college_football_system.main_analyzer import UnifiedCollegeFootballAnalyzer
        return UnifiedCollegeFootballAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['UnifiedCollegeFootballAnalyzer']
