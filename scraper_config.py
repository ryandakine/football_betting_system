#!/usr/bin/env python3
"""
Scraper Configuration - Self-Documenting
=========================================

WHY THIS FILE EXISTS:
- Explicit interface: Agent discovers sources at runtime (not memorizes)
- Context embedded: Each config explains what it provides
- Single source of truth: All scraper metadata in one place

PRINCIPLE: Agent can't hardcode wrong information because system provides it.
"""

from typing import TypedDict, Literal


class ScraperConfig(TypedDict):
    """Type-safe scraper configuration"""
    name: str
    url_pattern: str
    data_type: str
    coverage: str
    rate_limit: int
    reliability: Literal['high', 'medium', 'low']
    why: str
    limitations: str


# ============================================================================
# SCRAPER CONFIGURATIONS
# ============================================================================

SCRAPERS: dict[str, ScraperConfig] = {
    'teamrankings': {
        'name': 'TeamRankings Closing Lines',
        'url_pattern': 'https://www.teamrankings.com/ncf/odds-history/results/?year={year}',
        'data_type': 'closing_lines',
        'coverage': '2015-2024',
        'rate_limit': 60,  # seconds between requests
        'reliability': 'high',
        'why': (
            'Most reliable free source for historical closing spreads. '
            'Closing lines = final market price = what sharp bettors moved it to.'
        ),
        'limitations': (
            'May not have data for all years. Test recent years first (2024, 2023). '
            'Older years (2015-2018) might have gaps or different HTML structure.'
        )
    },

    'covers': {
        'name': 'Covers Historical Archive',
        'url_pattern': 'https://www.covers.com/sports/ncaaf/matchups?selectedDate={date}',
        'data_type': 'opening_and_closing_lines',
        'coverage': '2018-2024',
        'rate_limit': 30,
        'reliability': 'medium',
        'why': (
            'Backup source with both opening and closing lines. '
            'Opening lines = initial market price. '
            'Closing lines = final price after sharp action.'
        ),
        'limitations': (
            'Limited historical depth (only goes back to ~2018). '
            'Requires scraping week-by-week (slower than TeamRankings). '
            'HTML structure changes frequently.'
        )
    },
}


# ============================================================================
# DATA REQUIREMENTS
# ============================================================================

DATA_REQUIREMENTS = {
    'minimum_games_per_year': 500,  # NCAA FBS has ~800 games/year
    'minimum_coverage': 0.80,       # 80% of games must have market spreads
    'minimum_years': 3,             # Need at least 3 years for validation
    'spread_sanity_range': 50.0,    # NCAA spreads rarely exceed ±50
    'required_fields': [
        'year',
        'home_team',
        'away_team',
        'market_spread'
    ]
}


# ============================================================================
# VALIDATION RULES
# ============================================================================

def validate_scraper_config(scraper_name: str) -> None:
    """
    Runtime validation: Agent discovers if scraper is properly configured

    WHY: Fail fast if scraper config is incomplete
    """
    if scraper_name not in SCRAPERS:
        raise ValueError(
            f"Unknown scraper: '{scraper_name}'\n"
            f"Available scrapers: {list(SCRAPERS.keys())}"
        )

    config = SCRAPERS[scraper_name]

    # Validate required fields
    required = ['name', 'url_pattern', 'data_type', 'coverage', 'rate_limit', 'why']
    missing = [f for f in required if f not in config or not config[f]]

    if missing:
        raise ValueError(
            f"Scraper '{scraper_name}' config incomplete\n"
            f"Missing fields: {missing}"
        )

    print(f"✅ Scraper '{scraper_name}' config valid")


def get_scraper_info(scraper_name: str) -> ScraperConfig:
    """
    Get scraper configuration

    Agent calls this at runtime to discover scraper capabilities
    """
    validate_scraper_config(scraper_name)
    return SCRAPERS[scraper_name]


def list_available_scrapers() -> list[str]:
    """List all configured scrapers"""
    return list(SCRAPERS.keys())


def get_recommended_scraper() -> str:
    """
    Get recommended scraper based on reliability

    WHY: Agent doesn't guess - system tells it which to use
    """
    # Sort by reliability (high > medium > low)
    reliability_order = {'high': 0, 'medium': 1, 'low': 2}

    sorted_scrapers = sorted(
        SCRAPERS.items(),
        key=lambda x: reliability_order[x[1]['reliability']]
    )

    return sorted_scrapers[0][0]


# ============================================================================
# SELF-DOCUMENTING USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Display scraper information

    Usage: python scraper_config.py
    """

    print("\n" + "="*80)
    print("📊 AVAILABLE SCRAPERS")
    print("="*80 + "\n")

    for name, config in SCRAPERS.items():
        print(f"🔧 {config['name']} ({name})")
        print(f"   Data Type: {config['data_type']}")
        print(f"   Coverage: {config['coverage']}")
        print(f"   Reliability: {config['reliability']}")
        print(f"   Rate Limit: {config['rate_limit']}s between requests")
        print(f"   \n   Why: {config['why']}")
        print(f"   \n   Limitations: {config['limitations']}")
        print()

    print("\n" + "="*80)
    print("🎯 RECOMMENDED SCRAPER")
    print("="*80 + "\n")

    recommended = get_recommended_scraper()
    rec_config = SCRAPERS[recommended]
    print(f"Use: {rec_config['name']} ({recommended})")
    print(f"Why: {rec_config['why']}")
    print()

    print("\n" + "="*80)
    print("📋 DATA REQUIREMENTS")
    print("="*80 + "\n")

    for key, value in DATA_REQUIREMENTS.items():
        print(f"   {key}: {value}")
    print()
