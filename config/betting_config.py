"""
Unified Betting System Configuration
Shared settings for NCAA and NFL betting systems
"""

# API Configuration
API_CONFIG = {
    'odds_api_base_url': 'https://api.the-odds-api.com/v4/sports',
    'timeout': 10,
    'regions': 'us',
    'markets': 'h2h,spreads,totals',
    'odds_format': 'american'
}

# Sports Configuration
SPORTS = {
    'nfl': {
        'api_key': 'americanfootball_nfl',
        'season_weeks': 18,
        'late_season_start': 8
    },
    'ncaa': {
        'api_key': 'americanfootball_ncaaf',
        'season_weeks': 15,
        'late_season_start': 8
    }
}

# File Paths
PATHS = {
    'nfl': {
        'live_games': 'data/nfl_live_games.json',
        'predictions': 'data/predictions/nfl_prediction_log.json',
        'reports': 'reports/nfl_week_{week}'
    },
    'ncaa': {
        'live_games': 'data/ncaa_live_games.json',
        'predictions': 'data/predictions/prediction_log.json',
        'reports': 'reports/ncaa_week_{week}'
    }
}

# Model Weights (12-model ensemble)
MODEL_WEIGHTS = {
    'spread_ensemble': 0.12,
    'total_ensemble': 0.10,
    'moneyline_ensemble': 0.12,
    'rf_model': 0.09,
    'gb_model': 0.09,
    'spread_edges': 0.10,
    'total_edges': 0.08,
    'moneyline_edges': 0.09,
    'market_consensus': 0.07,
    'contrarian': 0.06,
    'referee': 0.04,
    'injury': 0.04
}

# Confidence Calibration Curves
CALIBRATION_CURVES = {
    'nfl': [
        (0.00, 0.60, 0.87),  # Below 0.60: multiply by 0.87
        (0.60, 0.70, lambda x: 0.52 + (x - 0.60) * 0.6),
        (0.70, 0.80, lambda x: 0.58 + (x - 0.70) * 0.7),
        (0.80, 1.00, lambda x: 0.65 + (x - 0.80) * 0.35)
    ],
    'ncaa': [
        (0.00, 0.60, 0.85),
        (0.60, 0.70, lambda x: 0.51 + (x - 0.60) * 0.65),
        (0.70, 0.80, lambda x: 0.56 + (x - 0.70) * 0.75),
        (0.80, 1.00, lambda x: 0.63 + (x - 0.80) * 0.40)
    ]
}

# Betting Tiers (5-tier system)
BETTING_TIERS = {
    'TIER_1_MEGA_EDGE': {
        'win_rate': 0.88,
        'multiplier': 1.50,
        'bet_size': 0.025,
        'description': 'Power 5 + W1-2 + Non-Conf (HIGHEST CONFIDENCE)'
    },
    'TIER_2_SUPER_EDGE': {
        'win_rate': 0.81,
        'multiplier': 1.35,
        'bet_size': 0.020,
        'description': 'Early Season + Non-Conf (VERY HIGH CONFIDENCE)'
    },
    'TIER_3_STRONG_EDGE': {
        'win_rate': 0.73,
        'multiplier': 1.25,
        'bet_size': 0.015,
        'description': 'Big Ten/Mountain West Home + Conference (HIGH CONFIDENCE)'
    },
    'TIER_4_MODERATE_EDGE': {
        'win_rate': 0.65,
        'multiplier': 1.15,
        'bet_size': 0.012,
        'description': 'Mixed patterns (MODERATE CONFIDENCE)'
    },
    'TIER_5_SELECTIVE': {
        'win_rate': 0.58,
        'multiplier': 1.05,
        'bet_size': 0.010,
        'description': 'Late season / Weak patterns (SELECTIVE ONLY)'
    }
}

# Bankroll Management
BANKROLL_CONFIG = {
    'circuit_breaker_threshold': 0.25,  # Stop at 25% drawdown
    'max_bets_per_week': 5,
    'min_confidence': 0.52,  # Don't bet below 52% confidence
    'max_single_bet': 0.025  # Max 2.5% of bankroll per bet
}

# DeepSeek R1 Configuration
DEEPSEEK_CONFIG = {
    'base_url': 'https://api.deepseek.com',
    'model': 'deepseek-reasoner',
    'max_tokens': 8000,
    'temperature': 1.0
}

# OpenRouter Configuration (Claude)
OPENROUTER_CONFIG = {
    'base_url': 'https://openrouter.ai/api/v1',
    'model': 'anthropic/claude-3.5-sonnet',
    'max_tokens': 4000,
    'temperature': 0.7
}

# Video Scouting Weights
VIDEO_SCOUTING_ADJUSTMENTS = {
    5: 1.00,  # 5/5 checks passed: 100% of bet
    4: 0.70,  # 4/5 checks passed: 70% of bet
    3: 0.50,  # 3/5 checks passed: 50% of bet
    2: 0.00,  # 2/5 or fewer: Skip bet
    1: 0.00,
    0: 0.00
}

# Auto-Skip Conditions
AUTO_SKIP_CONDITIONS = {
    'backup_qb_first_start': True,
    'three_plus_key_injuries': True,
    'wind_over_25mph': True,
    'line_moved_4plus_against': True,
    'eliminated_vs_playoff_team': True,
    'short_week_without_rest': True,
    'cross_country_early_game': True
}

def get_calibration_fn(sport='nfl'):
    """Get calibration function for a sport"""
    curve = CALIBRATION_CURVES.get(sport, CALIBRATION_CURVES['nfl'])

    def calibrate(raw_confidence):
        for min_val, max_val, adjustment in curve:
            if min_val <= raw_confidence < max_val:
                if callable(adjustment):
                    return adjustment(raw_confidence)
                else:
                    return raw_confidence * adjustment
        return raw_confidence

    return calibrate

def get_paths(sport, week=None):
    """Get file paths for a sport"""
    paths = PATHS.get(sport, PATHS['nfl']).copy()
    if week and 'reports' in paths:
        paths['reports'] = paths['reports'].format(week=week)
    return paths

def get_tier_by_confidence(confidence, reasons=None):
    """Assign betting tier based on confidence and edge reasons"""
    if reasons:
        if any('MEGA EDGE' in r for r in reasons):
            return 'TIER_1_MEGA_EDGE'
        elif any('SUPER EDGE' in r for r in reasons):
            return 'TIER_2_SUPER_EDGE'

    if confidence >= 0.73:
        return 'TIER_3_STRONG_EDGE'
    elif confidence >= 0.65:
        return 'TIER_4_MODERATE_EDGE'
    else:
        return 'TIER_5_SELECTIVE'
