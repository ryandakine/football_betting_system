import json
import random
import asyncio
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/referee_conspiracy")
CREW_PATH = DATA_DIR / "crew_game_log.parquet"
SIGNAL_PATH = DATA_DIR / "conspiracy_signals.json"
OUTPUT_PATH = DATA_DIR / "conspiracy_predictions.json"


def safe_read_parquet(path, columns=None):
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=columns or [])
    if columns:
        missing = [c for c in columns if c not in df.columns]
        for col in missing:
            df[col] = np.nan
        return df[columns]
    return df


def load_signal_payload():
    if SIGNAL_PATH.exists():
        try:
            return json.loads(SIGNAL_PATH.read_text())
        except Exception:
            return {}
    return {}


async def fetch_narrative_data(home_team: str, away_team: str) -> dict:
    """Fetch social sentiment/narrative data from Reddit/public sources.
    
    Detects media-created storylines (revenge games, playoff implications,
    QB narratives, coaching drama) that Vegas uses to set lines.
    """
    try:
        from simple_narrative_scraper import SimpleNarrativeScraper
        scraper = SimpleNarrativeScraper()
        narrative = await scraper.get_game_narrative(home_team, away_team)
        
        logger.info(f"📰 Narrative for {away_team}@{home_team}: strength={narrative['narrative_strength']:.2f}, lean={narrative['public_lean']:.2f}, bait={narrative['vegas_bait']}")
        
        return narrative
        
    except Exception as e:
        logger.warning(f"Narrative fetch failed for {away_team}@{home_team}: {e}")
        
        # Fallback: infer from team market size
        big_market = {"KC", "LAC", "DAL", "NE", "SF", "PHI", "NYG", "NYJ", "CHI", "MIA", "GB", "PIT"}
        high_offense = {"KC", "BUF", "MIA", "LAC", "CIN", "DAL", "SF", "PHI", "DET"}
        
        is_big_game = home_team in big_market or away_team in big_market
        has_offense = home_team in high_offense or away_team in high_offense
        
        return {
            "narrative_strength": 0.7 if is_big_game else 0.4,
            "media_hype": 0.6 if is_big_game else 0.3,
            "public_lean": 0.6 if has_offense else 0.5,
            "vegas_bait": is_big_game and has_offense,
            "storylines": [],
            "sentiment_counts": {"over": 0, "under": 0, "hype": 0},
        }


def conspiracy_vote(row, payload, narrative_data=None):
    game_key = row.game_id
    signal = payload.get(game_key, {})
    hashtag_spike = float(signal.get("hashtag_spike", signal.get("hashtag_pct_change", 0.0)))
    odds_drift = float(signal.get("odds_drift", 0.0))
    dark_pool = float(signal.get("dark_pool_alert", 0.0))
    weather_alert = bool(signal.get("weather_alert", False))
    
    # NEW: Narrative/media conspiracy signals
    narrative_strength = 0.0
    vegas_bait = False
    primetime_home_boost = 0.0
    
    # PRIMETIME HOME CONSPIRACY: Stadium/Refs/TV invested in home win
    is_primetime = getattr(row, 'is_primetime', False)
    weekday = getattr(row, 'weekday', 'Sunday')
    
    # Day-of-week betting patterns
    is_thursday = weekday == 'Thursday'
    is_monday = weekday == 'Monday'
    is_sunday_night = weekday == 'Sunday' and is_primetime
    
    if is_primetime:
        # TNF/MNF = lower public action, sharps dominate
        # SNF = max public action, fade opportunities
        if is_thursday or is_monday:
            primetime_home_boost = 0.10  # Lower - sharps dominate these days
            dark_pool = max(dark_pool, 0.3)  # Less conspiracy, more sharp action
            narrative_strength = 0.5  # Moderate narrative on isolated games
        else:  # Sunday Night
            primetime_home_boost = 0.20  # Higher - max public action to fade
            dark_pool = max(dark_pool, 0.6)  # Peak conspiracy on SNF
            narrative_strength = 0.8  # Maximum narrative on SNF
    
    # CONTRARIAN LOGIC: When to fade the public
    contrarian_signal = 0.0
    
    if narrative_data:
        narrative_strength = max(narrative_strength, narrative_data.get("narrative_strength", 0.0))
        media_hype = narrative_data.get("media_hype", 0.0)
        public_lean = narrative_data.get("public_lean", 0.5)
        vegas_bait = narrative_data.get("vegas_bait", False)
        
        # Adjust contrarian threshold based on day of week
        if is_thursday or is_monday:
            # TNF/MNF: Less public action, trust public more (sharps dominate)
            contrarian_threshold = 0.85  # Only fade if >85% public on one side
            contrarian_strength = 0.10  # Weaker fade signal
        elif is_sunday_night:
            # SNF: Max public action, fade aggressively
            contrarian_threshold = 0.70  # Fade if >70% public
            contrarian_strength = 0.20  # Stronger fade signal
        else:
            # Regular games
            contrarian_threshold = 0.75
            contrarian_strength = 0.15
        
        # CONTRARIAN: If public heavily on OVER, consider fading to UNDER
        if public_lean > contrarian_threshold and narrative_strength > 0.6:
            contrarian_signal = -contrarian_strength
            logger.info(f"⚠️ CONTRARIAN: {weekday} - Public {public_lean:.0%} on OVER - fade signal (-{contrarian_strength:.2f})")
        
        # CONTRARIAN: If public heavily on UNDER, fade to OVER
        elif public_lean < (1 - contrarian_threshold) and narrative_strength > 0.6:
            contrarian_signal = contrarian_strength
            logger.info(f"⚠️ CONTRARIAN: {weekday} - Public {public_lean:.0%} on UNDER - fade signal (+{contrarian_strength:.2f})")
        
        # If public heavily on one side + big narrative = Vegas trap
        if public_lean > 0.65 and media_hype > 0.6:
            odds_drift = max(odds_drift, 0.75)  # Boost conspiracy signal
        
        # Media hype on over + high narrative = potential scripted shootout
        if public_lean > 0.6 and narrative_strength > 0.7:
            dark_pool = max(dark_pool, 0.6)  # NFL/TV/Vegas collusion suspected
        
        # Primetime + big spread = home team cover likely (refs help)
        if is_primetime and abs(getattr(row, 'spread_line', 0)) > 7:
            primetime_home_boost += 0.1  # Extra boost for blowout narrative
    
    baseline = 0.5 + 0.15 * hashtag_spike + 0.12 * abs(odds_drift) + 0.1 * dark_pool
    baseline += 0.08 * max(0.0, row.flag_density - 0.11)
    baseline += 0.1 * narrative_strength  # Boost from media storyline
    baseline += primetime_home_boost  # PRIMETIME HOME CONSPIRACY
    baseline += contrarian_signal  # FADE THE PUBLIC
    
    if hashtag_spike >= 3.0:
        baseline = 0.99
    under_prob = float(np.clip(baseline, 0.02, 0.999))
    confidence = float(np.clip(0.55 + 0.25 * hashtag_spike + 0.2 * abs(odds_drift), 0.4, 1.0))
    if weather_alert:
        label = "SCRIPTED_BLACKOUT"
    elif hashtag_spike > 1.5 and row.flag_density > 0.11:
        label = "SCRIPTED_BLUEBALL"
    elif row.flag_density > 0.13 or row.penalties >= 18:
        label = "FLAG_FARM"
    else:
        label = "NORMAL_FLOW"
    return {
        "h_module": label,
        "l_module": {
            "under_prob": under_prob,
            "flag_spike": float(1.2 + 0.6 * hashtag_spike + 0.4 * dark_pool),
        },
        "confidence": confidence,
        "signals": {
            "hashtag_spike": hashtag_spike,
            "odds_drift": odds_drift,
            "dark_pool_alert": dark_pool,
            "weather_alert": weather_alert,
            "narrative_strength": narrative_strength,
            "vegas_bait": vegas_bait,
            "primetime_home_boost": primetime_home_boost,
            "is_primetime": is_primetime,
            "contrarian_signal": float(contrarian_signal),
            "weekday": str(weekday),
            "is_thursday": bool(is_thursday),
            "is_monday": bool(is_monday),
            "is_sunday_night": bool(is_sunday_night),
        },
        "asshole_ref_detected": bool(row.flag_density > 0.135),
        "flag_spike_series": {
            "social_media": float(hashtag_spike * 10),
            "betting_drift": float(odds_drift * 5),
            "dark_web_echo": float(dark_pool * 7),
        },
        "ref_bias_therapy_needed": 64.0 if dark_pool > 0.4 else 22.0,
    }


async def run_conspiracy_analysis():
    """Main async runner with narrative integration."""
    crew = safe_read_parquet(
        CREW_PATH,
        [
            "game_id",
            "home_team",
            "away_team",
            "flag_density",
            "penalties",
            "overtime_plays",
            "total_plays",
            "week",
            "is_primetime",
            "spread_line",
            "weekday",
        ],
    )
    signals = load_signal_payload()
    
    if crew.empty:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps({}, indent=2))
        return
    
    defaults = {
        "flag_density": 0.0,
        "penalties": 0,
        "overtime_plays": 0.0,
        "total_plays": 60.0,
        "week": 0,
        "home_team": "UNK",
        "away_team": "UNK",
        "is_primetime": False,
        "spread_line": 0.0,
        "weekday": "Sunday",
    }
    for key, value in defaults.items():
        if key not in crew.columns:
            crew[key] = value
    crew = crew.fillna(defaults)
    
    # Fetch narrative data for each game
    results = {}
    for idx, row in crew.iterrows():
        narrative_data = await fetch_narrative_data(row.home_team, row.away_team)
        pred = conspiracy_vote(row, signals, narrative_data)
        results[row.game_id] = pred
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_conspiracy_analysis())
