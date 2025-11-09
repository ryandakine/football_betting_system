"""
Edge Signal Constants
=====================
Comprehensive list of all edge signals detected by the 11-model system.

Signal Types:
- MODEL_*: Signals from individual models
- REF_*: General referee tendency signals
- TEAM_REF_*: Team-specific referee pairing signals
- ALGO_*: Algorithm consensus signals
- NARRATIVE_*: Game narrative signals
- CONTRARIAN_*: Sharp vs public divergence signals
"""

# ============================================================================
# Model Agreement Signals (Models 1-10)
# ============================================================================
UNANIMOUS_10_MODEL_EDGE = "UNANIMOUS_10_MODEL_EDGE"  # All 10 models agree (very rare!)
STRONG_MODEL_AGREEMENT = "STRONG_MODEL_AGREEMENT"  # Models tightly clustered
ALGO_CONSENSUS_STRONG = "ALGO_CONSENSUS_STRONG"  # XGBoost, NN, Stacking all confident

# ============================================================================
# Individual Model Edge Signals
# ============================================================================
FIRST_HALF_EDGE = "FIRST_HALF_EDGE"  # Model 4: First half spread edge
HOME_TOTAL_OVER_EDGE = "HOME_TOTAL_OVER_EDGE"  # Model 5: Home team total over
HOME_TOTAL_UNDER_EDGE = "HOME_TOTAL_UNDER_EDGE"  # Model 5: Home team total under
AWAY_TOTAL_OVER_EDGE = "AWAY_TOTAL_OVER_EDGE"  # Model 6: Away team total over
AWAY_TOTAL_UNDER_EDGE = "AWAY_TOTAL_UNDER_EDGE"  # Model 6: Away team total under

# ============================================================================
# MODEL 11: REFEREE INTELLIGENCE - THE CENTERPIECE
# ============================================================================

## General Referee Tendency Signals
REF_OVERTIME_SPECIALIST = "REF_OVERTIME_SPECIALIST"  # Ref causes high OT rate (bet over)
REF_HOME_BIAS = "REF_HOME_BIAS"  # Ref historically favors home team
REF_AWAY_BIAS = "REF_AWAY_BIAS"  # Ref historically favors away team (rare!)
REF_HOME_UNDERDOG_VALUE = "REF_HOME_UNDERDOG_VALUE"  # Ref bias + home underdog = ML value
REF_LOW_FLAGS_BLOWOUT = "REF_LOW_FLAGS_BLOWOUT"  # Ref lets teams play, games end quickly
REF_HIGH_PENALTIES_UNDER = "REF_HIGH_PENALTIES_UNDER"  # Ref calls lots of penalties = slower game
REF_CLOSE_GAME_SPECIALIST = "REF_CLOSE_GAME_SPECIALIST"  # Ref keeps games close with flags
REF_OVERSEAS_PATTERN = "REF_OVERSEAS_PATTERN"  # Tony Corrente special (unpredictable)

## Team-Specific Referee Pairing Signals (THE GOLD!)
TEAM_REF_HOME_BIAS = "TEAM_REF_HOME_BIAS"  # Ref favors THIS home team specifically
TEAM_REF_AWAY_BIAS = "TEAM_REF_AWAY_BIAS"  # Ref favors THIS away team specifically
TEAM_REF_UNDERDOG_VALUE = "TEAM_REF_UNDERDOG_VALUE"  # Ref bias + underdog = MASSIVE value
TEAM_REF_HIGH_PENALTIES_UNDER = "TEAM_REF_HIGH_PENALTIES_UNDER"  # Ref calls lots on THIS team
TEAM_REF_OVERTIME_SURGE = "TEAM_REF_OVERTIME_SURGE"  # Ref + team = high OT rate
TEAM_REF_SURGE_PATTERN = "TEAM_REF_SURGE_PATTERN"  # Historical surge pattern repeating

## Super Signal (Multiple Referee Edges Aligned)
REFEREE_EDGE_JACKPOT = "REFEREE_EDGE_JACKPOT"  # 3+ referee edges on same game!

# ============================================================================
# Situational Specialist Signals (Model 10)
# ============================================================================
PRIMETIME_BOOST = "PRIMETIME_BOOST"  # Primetime game adjustment
DIVISIONAL_BOOST = "DIVISIONAL_BOOST"  # Divisional rivalry adjustment
WEATHER_RAIN = "WEATHER_RAIN"  # Rain impact on scoring
WEATHER_SNOW = "WEATHER_SNOW"  # Snow impact on scoring
WEATHER_WIND_HIGH = "WEATHER_WIND_HIGH"  # High wind impact
REST_ADVANTAGE_PLUS = "REST_ADVANTAGE_PLUS"  # Home team well-rested
REST_ADVANTAGE_MINUS = "REST_ADVANTAGE_MINUS"  # Away team well-rested

# ============================================================================
# Narrative Signals
# ============================================================================
NARRATIVE_TRAP = "NARRATIVE_TRAP"  # Trap game detected
NARRATIVE_REVENGE = "NARRATIVE_REVENGE"  # Revenge game narrative
NARRATIVE_HOMECOMING = "NARRATIVE_HOMECOMING"  # Homecoming narrative
NARRATIVE_INJURY_REVENGE = "NARRATIVE_INJURY_REVENGE"  # Injury revenge angle

# ============================================================================
# Sentiment/Contrarian Signals
# ============================================================================
CONTRARIAN_EDGE = "CONTRARIAN_EDGE"  # Sharp vs public divergence
CROWD_ROAR_SIGNAL = "CROWD_ROAR_SIGNAL"  # Crowd sentiment signal
SHARP_PUBLIC_ML_DIVERGENCE = "SHARP_PUBLIC_ML_DIVERGENCE"  # Moneyline divergence
SHARP_PUBLIC_TOTAL_DIVERGENCE = "SHARP_PUBLIC_TOTAL_DIVERGENCE"  # Total divergence

# ============================================================================
# Overtime Specialist Signals
# ============================================================================
OT_SPECIALIST = "OT_SPECIALIST"  # Referee causes overtime frequently

# ============================================================================
# Signal Strength Mapping
# ============================================================================
SIGNAL_STRENGTH = {
    # Critical Signals (Highest Priority)
    REFEREE_EDGE_JACKPOT: 10,
    TEAM_REF_UNDERDOG_VALUE: 9,
    UNANIMOUS_10_MODEL_EDGE: 9,
    TEAM_REF_HOME_BIAS: 8,
    TEAM_REF_AWAY_BIAS: 8,
    TEAM_REF_SURGE_PATTERN: 8,

    # Strong Signals
    STRONG_MODEL_AGREEMENT: 7,
    ALGO_CONSENSUS_STRONG: 7,
    REF_OVERTIME_SPECIALIST: 7,
    TEAM_REF_OVERTIME_SURGE: 7,
    CONTRARIAN_EDGE: 7,

    # Medium Signals
    REF_HOME_BIAS: 6,
    REF_AWAY_BIAS: 6,
    REF_HOME_UNDERDOG_VALUE: 6,
    FIRST_HALF_EDGE: 5,
    HOME_TOTAL_OVER_EDGE: 5,
    AWAY_TOTAL_OVER_EDGE: 5,
    PRIMETIME_BOOST: 5,
    DIVISIONAL_BOOST: 5,

    # Lower Signals
    REF_LOW_FLAGS_BLOWOUT: 4,
    REF_HIGH_PENALTIES_UNDER: 4,
    REF_CLOSE_GAME_SPECIALIST: 4,
    TEAM_REF_HIGH_PENALTIES_UNDER: 4,
    NARRATIVE_TRAP: 4,
    NARRATIVE_REVENGE: 4,

    # Weak Signals (Still Useful)
    REF_OVERSEAS_PATTERN: 3,
    WEATHER_RAIN: 3,
    WEATHER_SNOW: 3,
    WEATHER_WIND_HIGH: 3,
}


# ============================================================================
# Signal Descriptions (for reporting)
# ============================================================================
SIGNAL_DESCRIPTIONS = {
    # Model Agreement
    UNANIMOUS_10_MODEL_EDGE: "All 10 models unanimously agree on this edge (extremely rare!)",
    STRONG_MODEL_AGREEMENT: "Strong agreement across multiple models (low variance)",
    ALGO_CONSENSUS_STRONG: "XGBoost, Neural Net, and Stacking models all confident",

    # Individual Models
    FIRST_HALF_EDGE: "First half spread model very confident",
    HOME_TOTAL_OVER_EDGE: "Home team expected to score above their team total",
    HOME_TOTAL_UNDER_EDGE: "Home team expected to score below their team total",
    AWAY_TOTAL_OVER_EDGE: "Away team expected to score above their team total",
    AWAY_TOTAL_UNDER_EDGE: "Away team expected to score below their team total",

    # Referee - General
    REF_OVERTIME_SPECIALIST: "Referee has high overtime frequency (games go to OT often)",
    REF_HOME_BIAS: "Referee historically favors home teams",
    REF_AWAY_BIAS: "Referee historically favors away teams (very rare pattern)",
    REF_HOME_UNDERDOG_VALUE: "Home underdog + referee bias = moneyline value opportunity",
    REF_LOW_FLAGS_BLOWOUT: "Referee lets teams play, games tend to end in blowouts",
    REF_HIGH_PENALTIES_UNDER: "Referee calls many penalties, slowing down the game",
    REF_CLOSE_GAME_SPECIALIST: "Referee keeps games close with penalty flags",
    REF_OVERSEAS_PATTERN: "Unusual penalty pattern detected (Tony Corrente signature)",

    # Referee - Team-Specific (THE GOLD!)
    TEAM_REF_HOME_BIAS: "Referee has historically favored THIS specific home team",
    TEAM_REF_AWAY_BIAS: "Referee has historically favored THIS specific away team",
    TEAM_REF_UNDERDOG_VALUE: "Team-specific referee bias + underdog status = MASSIVE moneyline edge",
    TEAM_REF_HIGH_PENALTIES_UNDER: "Referee calls excessive penalties on THIS specific team",
    TEAM_REF_OVERTIME_SURGE: "This referee + this team combination produces high OT rate",
    TEAM_REF_SURGE_PATTERN: "Historical surge pattern with this referee and team pairing",

    # Jackpot
    REFEREE_EDGE_JACKPOT: "Multiple referee edges aligned on this game - PLAY BIG!",

    # Situational
    PRIMETIME_BOOST: "Primetime game (SNF/MNF/TNF) with adjusted expectations",
    DIVISIONAL_BOOST: "Divisional rivalry game with heightened intensity",
    WEATHER_RAIN: "Rain expected, may slow down offense",
    WEATHER_SNOW: "Snow expected, significant scoring impact",
    WEATHER_WIND_HIGH: "High winds expected, passing game affected",

    # Narratives
    NARRATIVE_TRAP: "Potential trap game detected - fade the favorite",
    NARRATIVE_REVENGE: "Revenge game narrative detected",
    NARRATIVE_HOMECOMING: "Homecoming narrative (emotional impact)",
    NARRATIVE_INJURY_REVENGE: "Players returning from injury with revenge motive",

    # Sentiment/Contrarian
    CONTRARIAN_EDGE: "Sharp money diverging from public - fade the public!",
    CROWD_ROAR_SIGNAL: "Strong crowd sentiment detected",
    OT_SPECIALIST: "Referee causes games to go to overtime frequently",
}


# ============================================================================
# Betting Recommendations by Signal
# ============================================================================
SIGNAL_BET_RECOMMENDATIONS = {
    # Referee Signals
    REF_OVERTIME_SPECIALIST: ["Bet OVER on low totals", "Consider prop: Will game go to OT?"],
    TEAM_REF_HOME_BIAS: ["Bet HOME spread", "Bet HOME moneyline if small underdog"],
    TEAM_REF_AWAY_BIAS: ["Bet AWAY spread", "Bet AWAY moneyline"],
    TEAM_REF_UNDERDOG_VALUE: ["BET UNDERDOG MONEYLINE - THIS IS THE PLAY!"],
    REFEREE_EDGE_JACKPOT: ["MAX BET - Multiple edges aligned!"],
    REF_HIGH_PENALTIES_UNDER: ["Bet UNDER (more stoppages = fewer possessions)"],
    TEAM_REF_OVERTIME_SURGE: ["Bet OVER on moderate totals"],

    # Model Signals
    UNANIMOUS_10_MODEL_EDGE: ["STRONG PLAY - All models agree"],
    FIRST_HALF_EDGE: ["Consider 1H spread bet"],
    HOME_TOTAL_OVER_EDGE: ["Bet home team total OVER"],

    # Situational
    PRIMETIME_BOOST: ["Trust models more in primetime", "Public often overreacts"],
    CONTRARIAN_EDGE: ["Fade the public - bet opposite of public sentiment"],
}


if __name__ == "__main__":
    print("=" * 80)
    print("EDGE SIGNAL REFERENCE")
    print("=" * 80)

    print(f"\n📊 Total Signals Defined: {len(SIGNAL_DESCRIPTIONS)}")
    print(f"🎯 Critical Signals (Strength 8+): {sum(1 for s in SIGNAL_STRENGTH.values() if s >= 8)}")
    print(f"⚡ Strong Signals (Strength 6-7): {sum(1 for s in SIGNAL_STRENGTH.values() if 6 <= s < 8)}")

    print("\n" + "=" * 80)
    print("TOP 10 SIGNALS BY STRENGTH")
    print("=" * 80)

    sorted_signals = sorted(SIGNAL_STRENGTH.items(), key=lambda x: x[1], reverse=True)
    for i, (signal, strength) in enumerate(sorted_signals[:10], 1):
        desc = SIGNAL_DESCRIPTIONS.get(signal, "No description")
        print(f"\n{i}. {signal} (Strength: {strength})")
        print(f"   {desc}")
        if signal in SIGNAL_BET_RECOMMENDATIONS:
            print(f"   💰 Recommendations: {', '.join(SIGNAL_BET_RECOMMENDATIONS[signal])}")
