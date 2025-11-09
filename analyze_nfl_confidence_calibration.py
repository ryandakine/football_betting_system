#!/usr/bin/env python3
"""
NFL Confidence Calibration Analysis
Analyzes actual NFL prediction logs to calibrate confidence scores.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

print("🏈 NFL CONFIDENCE CALIBRATION ANALYSIS")
print("=" * 80)

# Load NFL backtesting data
backtest_dir = Path("data/backtesting")
graded_files = list(backtest_dir.glob("graded_*.parquet"))

if not graded_files:
    print("\n⚠️  No graded prediction files found in data/backtesting/")
    print("   Run your NFL backtest first to generate data")
    exit(1)

print(f"\n📊 Loading NFL prediction data...")
print(f"Found {len(graded_files)} graded prediction files\n")

# Load all graded predictions
all_predictions = []
for file in graded_files:
    try:
        df = pd.read_parquet(file)
        print(f"  ✅ Loaded {file.name}: {len(df)} predictions")
        all_predictions.append(df)
    except Exception as e:
        print(f"  ⚠️  Could not load {file.name}: {e}")

if not all_predictions:
    print("\n❌ No data loaded successfully")
    exit(1)

# Combine all predictions
df = pd.concat(all_predictions, ignore_index=True)
print(f"\n✅ Total predictions loaded: {len(df)}")

# Check required columns
required_cols = ['spread_confidence', 'spread_correct']
missing = [col for col in required_cols if col not in df.columns]

if missing:
    print(f"\n⚠️  Missing columns: {missing}")
    print(f"Available columns: {list(df.columns)[:20]}")
    
    # Try alternative column names
    if 'overall_confidence' in df.columns:
        df['spread_confidence'] = df['overall_confidence']
        print("✅ Using 'overall_confidence' as confidence score")
    
    if 'spread_result' in df.columns:
        df['spread_correct'] = (df['spread_result'] == 'WIN')
        print("✅ Using 'spread_result' for results")
    elif 'spread_graded' in df.columns:
        df['spread_correct'] = (df['spread_graded'] == 'WIN')
        print("✅ Using 'spread_graded' for results")
    elif 'result' in df.columns:
        df['spread_correct'] = (df['result'] == 'WIN')
        print("✅ Using 'result' for results")

# Filter valid predictions
df = df[df['spread_confidence'].notna() & df['spread_correct'].notna()].copy()
print(f"✅ Valid predictions: {len(df)}")

# Create confidence buckets
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
df['confidence_bucket'] = pd.cut(df['spread_confidence'], bins=bins, labels=labels, include_lowest=True)

# Calculate metrics per bucket
print("\n" + "=" * 80)
print("📊 NFL CONFIDENCE CALIBRATION ANALYSIS")
print("=" * 80)
print()

results = []
for bucket in labels:
    bucket_df = df[df['confidence_bucket'] == bucket]
    
    if len(bucket_df) == 0:
        continue
    
    # Metrics
    actual_hit_rate = bucket_df['spread_correct'].mean()
    expected_accuracy = bucket_df['spread_confidence'].mean()
    calibration_gap = actual_hit_rate - expected_accuracy
    n_predictions = len(bucket_df)
    
    # Point differential for wrong predictions
    if 'point_differential' in bucket_df.columns:
        wrong_preds = bucket_df[~bucket_df['spread_correct']]
        avg_point_diff = wrong_preds['point_differential'].abs().mean() if len(wrong_preds) > 0 else 0
    else:
        avg_point_diff = 0
    
    results.append({
        'Confidence Bucket': bucket,
        'Count': n_predictions,
        'Expected Accuracy': f"{expected_accuracy:.1%}",
        'Actual Hit Rate': f"{actual_hit_rate:.1%}",
        'Calibration Gap': f"{calibration_gap:+.1%}",
        'Avg Point Diff (Wrong)': f"{avg_point_diff:.1f}" if avg_point_diff > 0 else "N/A"
    })

# Display table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print()

# Overall calibration
overall_expected = df['spread_confidence'].mean()
overall_actual = df['spread_correct'].mean()
overall_gap = overall_actual - overall_expected

print("=" * 80)
print("📈 OVERALL NFL CALIBRATION")
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
    print("✅ WELL CALIBRATED - NFL confidence scores are accurate!")
elif overall_gap > 0.05:
    print("⚠️  UNDERCONFIDENT - Winning more than confidence suggests.")
    print(f"    Recommendation: Increase confidence scores by ~{overall_gap*100:.0f}%")
elif overall_gap < -0.05:
    print("⚠️  OVERCONFIDENT - Winning less than confidence suggests.")
    print(f"    Recommendation: Decrease confidence scores by ~{abs(overall_gap)*100:.0f}%")
else:
    print("✅ SLIGHTLY MISCALIBRATED - Minor adjustment needed.")
    if overall_gap > 0:
        print("    Recommendation: Increase confidence scores by ~2-3%")
    else:
        print("    Recommendation: Decrease confidence scores by ~2-3%")

print()

# Bucket analysis
print("=" * 80)
print("🎯 BUCKET ANALYSIS")
print("=" * 80)

for bucket in labels:
    bucket_df = df[df['confidence_bucket'] == bucket]
    if len(bucket_df) == 0:
        continue
    
    actual_rate = bucket_df['spread_correct'].mean()
    expected_rate = bucket_df['spread_confidence'].mean()
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

# Save report
report_dir = Path("reports")
report_dir.mkdir(exist_ok=True)

report = {
    'sport': 'NFL',
    'timestamp': pd.Timestamp.now().isoformat(),
    'total_predictions': len(df),
    'overall_expected': float(overall_expected),
    'overall_actual': float(overall_actual),
    'calibration_gap': float(overall_gap),
    'bucket_analysis': results
}

with open(report_dir / 'nfl_confidence_calibration.json', 'w') as f:
    json.dump(report, f, indent=2)

print("=" * 80)
print("💾 Report saved to: reports/nfl_confidence_calibration.json")
print("=" * 80)
print()

print("📝 NFL CALIBRATION FORMULA:")
print("-" * 80)
if overall_gap < -0.05:
    # Overconfident
    print(f"Apply this calibration to your NFL predictions:")
    print()
    print("```python")
    print("if confidence > 0.80:")
    print(f"    calibrated = confidence * {0.85:.2f}")
    print("elif confidence > 0.60:")
    print(f"    calibrated = confidence * {0.90:.2f}")
    print("else:")
    print(f"    calibrated = confidence")
    print("```")
elif overall_gap > 0.05:
    # Underconfident
    print(f"Your NFL predictions are actually BETTER than confidence suggests!")
    print(f"You can safely increase confidence by {overall_gap*100:.0f}%")
else:
    print("✅ Current NFL confidence scores are well-calibrated")
    print("   Continue using as-is")

print()
