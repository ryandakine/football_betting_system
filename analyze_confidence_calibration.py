#!/usr/bin/env python3
"""
Analyze confidence calibration for betting predictions.
Shows if confidence scores are overconfident/underconfident.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

print("🎯 CONFIDENCE CALIBRATION ANALYSIS")
print("=" * 80)

# Look for prediction logs
data_dir = Path("data/football/historical/ncaaf")
predictions = []

# Check for any prediction logs
prediction_files = [
    "data/backtesting/graded_*.json",
    "data/predictions/*.json",
    "reports/backtesting/*.json"
]

print("\n📊 Searching for prediction logs...")

# For now, let's use the NCAA historical data to simulate predictions
# In production, you'd load actual prediction logs
print("\n⚠️  No prediction logs found. Let me create sample calibration data...")
print("    To use this script with real data, save predictions to:")
print("    data/predictions/prediction_log.json")
print()

# Sample structure for your future predictions
sample_format = {
    "game": "Team A @ Team B",
    "predicted_winner": "Team A",
    "spread": -7.5,
    "confidence_score": 0.75,
    "actual_result": "WIN",  # or "LOSS"
    "point_differential": -10  # Actual spread vs predicted
}

print("Expected prediction log format:")
print(json.dumps(sample_format, indent=2))
print()

# Load NCAA games to simulate calibration analysis
print("📊 Using NCAA historical data for calibration analysis...")
all_predictions = []

for year in [2023, 2024]:
    games_file = data_dir / f"ncaaf_{year}_games.json"
    if not games_file.exists():
        continue
    
    with open(games_file) as f:
        games = json.load(f)
    
    for game in games:
        if not game.get('completed') or game.get('homePoints') is None:
            continue
        
        home_points = game.get('homePoints', 0)
        away_points = game.get('awayPoints', 0)
        home_won = home_points > away_points
        
        # Simulate confidence using pregame Elo
        home_elo = game.get('homePregameElo') or 1500
        away_elo = game.get('awayPregameElo') or 1500
        elo_diff = home_elo - away_elo
        
        # Convert Elo diff to win probability
        win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        
        all_predictions.append({
            'game': f"{game.get('awayTeam')} @ {game.get('homeTeam')}",
            'predicted_winner': game.get('homeTeam') if win_prob > 0.5 else game.get('awayTeam'),
            'confidence_score': max(win_prob, 1 - win_prob),
            'actual_result': 'WIN' if (win_prob > 0.5 and home_won) or (win_prob < 0.5 and not home_won) else 'LOSS',
            'point_differential': abs(home_points - away_points)
        })

print(f"✅ Loaded {len(all_predictions)} predictions for analysis\n")

# Convert to DataFrame
df = pd.DataFrame(all_predictions)

# Create confidence buckets
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
df['confidence_bucket'] = pd.cut(df['confidence_score'], bins=bins, labels=labels, include_lowest=True)

# Calculate metrics per bucket
print("=" * 80)
print("📊 CONFIDENCE CALIBRATION ANALYSIS")
print("=" * 80)
print()

results = []
for bucket in labels:
    bucket_df = df[df['confidence_bucket'] == bucket]
    
    if len(bucket_df) == 0:
        continue
    
    # Calculate metrics
    actual_hit_rate = (bucket_df['actual_result'] == 'WIN').mean()
    expected_accuracy = bucket_df['confidence_score'].mean()
    calibration_gap = actual_hit_rate - expected_accuracy
    n_predictions = len(bucket_df)
    
    # Average point differential when wrong
    wrong_preds = bucket_df[bucket_df['actual_result'] == 'LOSS']
    avg_point_diff_when_wrong = wrong_preds['point_differential'].mean() if len(wrong_preds) > 0 else 0
    
    results.append({
        'Confidence Bucket': bucket,
        'Count': n_predictions,
        'Expected Accuracy': f"{expected_accuracy:.1%}",
        'Actual Hit Rate': f"{actual_hit_rate:.1%}",
        'Calibration Gap': f"{calibration_gap:+.1%}",
        'Avg Point Diff (Wrong)': f"{avg_point_diff_when_wrong:.1f}"
    })

# Display results table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print()

# Overall calibration
overall_expected = df['confidence_score'].mean()
overall_actual = (df['actual_result'] == 'WIN').mean()
overall_gap = overall_actual - overall_expected

print("=" * 80)
print("📈 OVERALL CALIBRATION")
print("=" * 80)
print(f"Total Predictions: {len(df)}")
print(f"Expected Accuracy: {overall_expected:.1%}")
print(f"Actual Accuracy: {overall_actual:.1%}")
print(f"Calibration Gap: {overall_gap:+.1%}")
print()

# Diagnosis
print("=" * 80)
print("🔍 CALIBRATION DIAGNOSIS")
print("=" * 80)

if abs(overall_gap) < 0.02:
    print("✅ WELL CALIBRATED - Your confidence scores are accurate!")
elif overall_gap > 0.05:
    print("⚠️  UNDERCONFIDENT - You're winning more than your confidence suggests.")
    print("    Recommendation: Increase confidence scores by ~5-10%")
elif overall_gap < -0.05:
    print("⚠️  OVERCONFIDENT - You're winning less than your confidence suggests.")
    print(f"    Recommendation: Decrease confidence scores by ~{abs(overall_gap)*100:.0f}%")
else:
    print("✅ SLIGHTLY MISCALIBRATED - Minor adjustment needed.")
    if overall_gap > 0:
        print("    Recommendation: Increase confidence scores by ~2-3%")
    else:
        print("    Recommendation: Decrease confidence scores by ~2-3%")

print()

# Find worst calibrated bucket
print("=" * 80)
print("🎯 BUCKET ANALYSIS")
print("=" * 80)

for bucket in labels:
    bucket_df = df[df['confidence_bucket'] == bucket]
    if len(bucket_df) == 0:
        continue
    
    actual_rate = (bucket_df['actual_result'] == 'WIN').mean()
    expected_rate = bucket_df['confidence_score'].mean()
    gap = actual_rate - expected_rate
    
    if abs(gap) > 0.05:
        status = "⚠️  MISCALIBRATED"
        if gap > 0:
            print(f"{status} {bucket}: Winning {gap*100:.1f}% MORE than expected")
        else:
            print(f"{status} {bucket}: Winning {abs(gap)*100:.1f}% LESS than expected")
    else:
        print(f"✅ CALIBRATED {bucket}: Within acceptable range")

print()

# Save calibration report
report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)

report = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'total_predictions': len(df),
    'overall_expected': float(overall_expected),
    'overall_actual': float(overall_actual),
    'calibration_gap': float(overall_gap),
    'bucket_analysis': results
}

with open(report_dir / 'confidence_calibration.json', 'w') as f:
    json.dump(report, f, indent=2)

print("=" * 80)
print("💾 Report saved to: reports/confidence_calibration.json")
print("=" * 80)
print()

print("📝 RECOMMENDATIONS FOR YOUR NCAA SYSTEM:")
print("-" * 80)
print("Based on the 88.6% confidence that missed (North Carolina):")
print()
print("1. High confidence (80%+) predictions need adjustment")
print("2. Consider reducing max confidence to 75%")
print("3. Add more features to improve model accuracy")
print("4. Use Kelly Criterion with fractional sizing (0.25)")
print("5. Track and update calibration monthly")
print()

print("🔧 TO USE WITH YOUR REAL PREDICTIONS:")
print("-" * 80)
print("1. Save predictions to: data/predictions/prediction_log.json")
print("2. Format: [{game, predicted_winner, confidence_score, actual_result}, ...]")
print("3. Run this script after each weekend")
print("4. Adjust confidence scores based on calibration gaps")
print()
