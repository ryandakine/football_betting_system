#!/usr/bin/env python3
"""
Backtest Conspiracy Predictions on First Half of Season
========================================================
Tests the primetime home conspiracy theory against actual results.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path("data/referee_conspiracy")
CREW_PATH = DATA_DIR / "crew_game_log.parquet"
RESULTS_PATH = DATA_DIR / "game_results.json"
BACKTEST_OUTPUT = DATA_DIR / "backtest_results.json"


def load_historical_results():
    """Load any manually recorded results."""
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text())
        except Exception:
            return {}
    return {}


def generate_conspiracy_prediction(row):
    """Generate conspiracy prediction for a game using primetime logic."""
    
    # Base signals
    flag_density = row.get('flag_density', 0.0)
    penalties = row.get('penalties', 0)
    is_primetime = row.get('is_primetime', False)
    spread_line = row.get('spread_line', 0.0)
    
    # Calculate conspiracy signals
    hashtag_spike = 0.0
    odds_drift = 0.0
    dark_pool = 0.0
    narrative_strength = 0.0
    primetime_home_boost = 0.0
    
    # PRIMETIME HOME CONSPIRACY
    if is_primetime:
        primetime_home_boost = 0.15  # Base primetime boost
        dark_pool = max(dark_pool, 0.5)  # NFL/Stadium/TV conspiracy
        narrative_strength = 0.7  # High narrative on primetime
        
        # Extra boost for big spread (blowout narrative)
        if abs(spread_line) > 7:
            primetime_home_boost += 0.1
    
    # Calculate under probability
    baseline = 0.5
    baseline += 0.15 * hashtag_spike
    baseline += 0.12 * abs(odds_drift)
    baseline += 0.1 * dark_pool
    baseline += 0.08 * max(0.0, flag_density - 0.11)
    baseline += 0.1 * narrative_strength
    baseline += primetime_home_boost  # PRIMETIME CONSPIRACY
    
    under_prob = float(np.clip(baseline, 0.02, 0.999))
    
    # Determine prediction
    if under_prob > 0.75:
        # HIGH under prob on primetime = potential conspiracy, flip to OVER
        predicted_over = True if is_primetime else False
        confidence = "HIGH_CONSPIRACY" if is_primetime else "HIGH"
    elif under_prob > 0.55:
        predicted_over = False
        confidence = "LEAN_UNDER"
    elif under_prob < 0.45:
        predicted_over = True
        confidence = "LEAN_OVER"
    else:
        predicted_over = None
        confidence = "COIN_FLIP"
    
    return {
        "under_prob": under_prob,
        "predicted_over": predicted_over,
        "confidence": confidence,
        "is_primetime": is_primetime,
        "primetime_home_boost": primetime_home_boost,
        "dark_pool": dark_pool,
        "narrative_strength": narrative_strength,
    }


def backtest_season(season=2025, weeks_start=1, weeks_end=8):
    """Backtest predictions on historical games."""
    
    # Load crew data with actual scores
    crew = pd.read_parquet(CREW_PATH)
    
    # Filter to season and weeks
    games = crew[
        (crew['season'] == season) & 
        (crew['week'] >= weeks_start) & 
        (crew['week'] <= weeks_end) &
        (crew['total_line'] > 0)  # Only games with betting lines
    ].copy()
    
    if games.empty:
        print(f"❌ No games found for {season} weeks {weeks_start}-{weeks_end}")
        return
    
    print(f"{'='*80}")
    print(f"BACKTESTING CONSPIRACY PREDICTIONS: {season} Weeks {weeks_start}-{weeks_end}")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, game in games.iterrows():
        # Skip if no actual score
        if pd.isna(game['home_score']) or pd.isna(game['away_score']):
            continue
        
        # Generate prediction
        pred = generate_conspiracy_prediction(game)
        
        # Calculate actual result
        actual_total = game['home_score'] + game['away_score']
        total_line = game['total_line']
        actual_over = actual_total > total_line
        
        # Check if prediction was correct
        if pred['predicted_over'] is None:
            correct = None  # Coin flip
        else:
            correct = (pred['predicted_over'] == actual_over)
        
        result = {
            "game_id": game['game_id'],
            "week": int(game['week']),
            "home_team": game['home_team'],
            "away_team": game['away_team'],
            "home_score": int(game['home_score']),
            "away_score": int(game['away_score']),
            "actual_total": int(actual_total),
            "total_line": float(total_line),
            "actual_over": bool(actual_over),
            "predicted_over": pred['predicted_over'],
            "under_prob": float(pred['under_prob']),
            "confidence": pred['confidence'],
            "correct": correct,
            "is_primetime": bool(pred['is_primetime']),
            "primetime_home_boost": float(pred['primetime_home_boost']),
        }
        
        results.append(result)
    
    # Save results
    BACKTEST_OUTPUT.write_text(json.dumps(results, indent=2))
    
    # Calculate statistics
    total_predictions = len([r for r in results if r['predicted_over'] is not None])
    correct_predictions = len([r for r in results if r['correct'] is True])
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Primetime statistics
    primetime_games = [r for r in results if r['is_primetime']]
    primetime_predictions = [r for r in primetime_games if r['predicted_over'] is not None]
    primetime_correct = [r for r in primetime_predictions if r['correct'] is True]
    primetime_accuracy = len(primetime_correct) / len(primetime_predictions) if primetime_predictions else 0
    
    # Regular games
    regular_games = [r for r in results if not r['is_primetime']]
    regular_predictions = [r for r in regular_games if r['predicted_over'] is not None]
    regular_correct = [r for r in regular_predictions if r['correct'] is True]
    regular_accuracy = len(regular_correct) / len(regular_predictions) if regular_predictions else 0
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"Total Games: {len(results)}")
    print(f"Games with Predictions: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Overall Accuracy: {accuracy:.1%}")
    print()
    print(f"🏈 PRIMETIME GAMES (TNF/SNF/MNF)")
    print(f"  Total: {len(primetime_games)}")
    print(f"  Predictions: {len(primetime_predictions)}")
    print(f"  Correct: {len(primetime_correct)}")
    print(f"  Accuracy: {primetime_accuracy:.1%}")
    print()
    print(f"📅 REGULAR GAMES")
    print(f"  Total: {len(regular_games)}")
    print(f"  Predictions: {len(regular_predictions)}")
    print(f"  Correct: {len(regular_correct)}")
    print(f"  Accuracy: {regular_accuracy:.1%}")
    print(f"{'='*80}")
    
    # Show sample predictions
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    for result in results[:5]:
        status = "✅" if result['correct'] else "❌" if result['correct'] is not None else "⚠️"
        primetime_flag = "🌙" if result['is_primetime'] else "☀️"
        print(f"{status} {primetime_flag} Week {result['week']}: {result['away_team']}@{result['home_team']}")
        print(f"   Predicted: {'OVER' if result['predicted_over'] else 'UNDER' if result['predicted_over'] is not None else 'SKIP'}")
        print(f"   Actual: {result['actual_total']} ({'OVER' if result['actual_over'] else 'UNDER'}, line {result['total_line']})")
        print(f"   Confidence: {result['confidence']} (under_prob={result['under_prob']:.2f})")
    
    print(f"\n💾 Full results saved to: {BACKTEST_OUTPUT}")
    
    return results


def main():
    results = backtest_season(season=2025, weeks_start=1, weeks_end=8)


if __name__ == "__main__":
    main()
