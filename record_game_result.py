#!/usr/bin/env python3
"""
Record Game Result for Learning System
=======================================
Feeds actual game results back into the learning system to improve future predictions.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/referee_conspiracy")
CREW_PATH = DATA_DIR / "crew_game_log.parquet"
FUSED_PATH = DATA_DIR / "fused_hr_predictions.json"
RESULTS_PATH = DATA_DIR / "game_results.json"


def load_results_history():
    """Load existing results history."""
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_results_history(results):
    """Save results history."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


def record_result(game_id: str, home_score: int, away_score: int):
    """Record a game result and update prediction accuracy."""
    
    # Load prediction
    if not FUSED_PATH.exists():
        print(f"❌ No predictions found at {FUSED_PATH}")
        sys.exit(1)
    
    with open(FUSED_PATH) as f:
        predictions = json.load(f)
    
    if game_id not in predictions:
        print(f"❌ Game {game_id} not found in predictions")
        print(f"Available games: {list(predictions.keys())[:10]}")
        sys.exit(1)
    
    pred = predictions[game_id]
    
    # Load game metadata
    crew = pd.read_parquet(CREW_PATH)
    game_row = crew[crew['game_id'] == game_id]
    
    if game_row.empty:
        print(f"❌ Game {game_id} not found in crew_game_log")
        sys.exit(1)
    
    game_row = game_row.iloc[0]
    
    # Calculate actuals
    total = home_score + away_score
    total_line = game_row.get('total_line', 0)
    actual_over = total > total_line
    
    # Get prediction
    predicted_under_prob = pred.get('under_prob', 0.5)
    predicted_over = predicted_under_prob < 0.5
    
    # Check if prediction was correct
    over_under_correct = (predicted_over and actual_over) or (not predicted_over and not actual_over)
    
    # Load results history
    results = load_results_history()
    
    # Record result (convert numpy types to native Python)
    result_entry = {
        "game_id": str(game_id),
        "home_team": str(game_row['home_team']),
        "away_team": str(game_row['away_team']),
        "home_score": int(home_score),
        "away_score": int(away_score),
        "total": int(total),
        "total_line": float(total_line),
        "actual_over": bool(actual_over),
        "predicted_under_prob": float(predicted_under_prob),
        "predicted_over": bool(predicted_over),
        "over_under_correct": bool(over_under_correct),
        "prediction_winner": str(pred.get('winner', 'N/A')),
        "prediction_script": str(pred.get('h_module', 'N/A')),
        "narrative_strength": float(pred.get('signals', {}).get('narrative_strength', 0)),
        "vegas_bait": bool(pred.get('signals', {}).get('vegas_bait', False)),
        "recorded_at": datetime.now().isoformat(),
    }
    
    results[game_id] = result_entry
    save_results_history(results)
    
    # Print summary
    print("="*80)
    print(f"🏈 GAME RESULT RECORDED: {game_row['away_team']} @ {game_row['home_team']}")
    print("="*80)
    print(f"Score: {game_row['home_team']} {home_score}, {game_row['away_team']} {away_score}")
    print(f"Total: {total} (Line: {total_line})")
    print(f"Result: {'OVER' if actual_over else 'UNDER'}")
    print()
    print("PREDICTION vs REALITY")
    print(f"  Predicted: {'OVER' if predicted_over else 'UNDER'} ({predicted_under_prob:.1%} under)")
    print(f"  Actual: {'OVER' if actual_over else 'UNDER'}")
    print(f"  Result: {'✅ CORRECT' if over_under_correct else '❌ WRONG'}")
    print()
    print("PREDICTION METADATA")
    print(f"  Winner: {pred['winner']}")
    print(f"  Script: {pred['h_module']}")
    print(f"  Narrative Strength: {result_entry['narrative_strength']:.2f}")
    print(f"  Vegas Bait: {result_entry['vegas_bait']}")
    print("="*80)
    
    # Calculate overall accuracy
    total_predictions = len(results)
    correct_predictions = sum(1 for r in results.values() if r['over_under_correct'])
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"\n📊 OVERALL ACCURACY: {correct_predictions}/{total_predictions} ({accuracy:.1%})")
    
    # Analyze by model winner
    print("\n🤖 ACCURACY BY MODEL WINNER:")
    for model in ['BEAST', 'SNIPER', 'CONSPIRACY']:
        model_results = [r for r in results.values() if r['prediction_winner'] == model]
        if model_results:
            model_correct = sum(1 for r in model_results if r['over_under_correct'])
            model_accuracy = model_correct / len(model_results)
            print(f"  {model}: {model_correct}/{len(model_results)} ({model_accuracy:.1%})")
    
    return over_under_correct


def main():
    parser = argparse.ArgumentParser(description="Record game result for learning")
    parser.add_argument("game_id", help="Game ID (e.g., 2025_08_MIN_LAC)")
    parser.add_argument("home_score", type=int, help="Home team score")
    parser.add_argument("away_score", type=int, help="Away team score")
    
    args = parser.parse_args()
    
    record_result(args.game_id, args.home_score, args.away_score)


if __name__ == "__main__":
    main()
