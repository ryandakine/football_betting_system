#!/usr/bin/env python3
"""
NCAA 12-Model Super Intelligence Training Pipeline
==================================================
Train all 12 models including:
- Models 1-10: Core prediction models
- Model 11: Officiating Bias Model
- Model 12: Prop Bet Specialist Model

Based on proven NFL training approach with NCAA-specific features.
"""

import sys
import logging
from pathlib import Path
from ncaa_models.feature_engineering import NCAAFeatureEngineer
from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline for all 12 models"""
    print("\n" + "="*70)
    print("🤖 NCAA 12-MODEL SUPER INTELLIGENCE TRAINING")
    print("="*70)
    print()
    print("Models 1-10: Core Prediction Models")
    print("  1. Spread Ensemble (RF + GB + Ridge)")
    print("  2. Total Ensemble (RF + GB)")
    print("  3. Moneyline Ensemble (RF + LR)")
    print("  4. First Half Spread (GB)")
    print("  5. Home Team Total (RF)")
    print("  6. Away Team Total (RF)")
    print("  7. Alt Spread (RF + GB with confidence intervals)")
    print("  8. XGBoost Super Ensemble (300 estimators)")
    print("  9. Neural Network Deep (3 layers: 128→64→32)")
    print(" 10. Stacking Meta-Learner (learns from other 9)")
    print()
    print("Specialized Models:")
    print(" 11. Officiating Bias Model (conference crew analysis)")
    print(" 12. Prop Bet Specialist (player/team props)")
    print()
    print("="*70)

    # Initialize feature engineer
    logger.info("\n🔧 Initializing feature engineering...")
    feature_engineer = NCAAFeatureEngineer(
        data_dir="data/football/historical/ncaaf"
    )

    # Initialize model orchestrator
    logger.info("🔧 Initializing 12-model orchestrator...")
    orchestrator = SuperIntelligenceOrchestrator(
        models_dir="models/ncaa_12_model"
    )

    # Determine which seasons to train on
    print()
    seasons_input = input("Which seasons to train on? (e.g., 2023,2024 or 2015-2024 or press Enter for 2023-2024): ").strip()

    if not seasons_input:
        seasons = [2023, 2024]  # Default
        logger.info("Using default: 2023-2024")
    elif '-' in seasons_input:
        # Range specified
        start, end = seasons_input.split('-')
        seasons = list(range(int(start), int(end) + 1))
    elif ',' in seasons_input:
        # List specified
        seasons = [int(s.strip()) for s in seasons_input.split(',')]
    else:
        # Single season
        seasons = [int(seasons_input)]

    logger.info(f"\n✅ Training on seasons: {seasons}")

    # Train all 12 models
    try:
        orchestrator.train_all(feature_engineer, seasons=seasons)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save all models
    logger.info("\n💾 Saving all 12 models...")
    for model_name, model in orchestrator.models.items():
        if model.is_trained:
            model_path = orchestrator.models_dir / f"{model_name}.pkl"
            try:
                model.save(model_path)
                logger.info(f"  ✅ Saved {model_name}")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not save {model_name}: {e}")

    # Print summary
    print("\n" + "="*70)
    print("✅ 12-MODEL SUPER INTELLIGENCE TRAINING COMPLETE!")
    print("="*70)
    print()
    print("📊 Models Trained:")
    trained_count = sum(1 for m in orchestrator.models.values() if m.is_trained)
    print(f"  {trained_count} / 12 models successfully trained")
    print()
    print("📍 Next Steps:")
    print("  1. Analyze confidence calibration:")
    print("     python analyze_ncaa_12_models_calibration.py")
    print()
    print("  2. Make calibrated predictions:")
    print("     python predict_ncaa_12_models.py")
    print()
    print("  3. Run comprehensive backtest:")
    print("     python backtest_ncaa_improved.py")
    print()
    print("  4. View officiating bias analysis:")
    print("     python test_ncaa_officiating_bias.py")
    print()
    print("="*70)

    # Print model details
    print("\n📈 MODEL PERFORMANCE SUMMARY:")
    print("="*70)

    for model_name, model in orchestrator.models.items():
        if model.is_trained:
            print(f"\n{model.name}:")
            print(f"  Status: ✅ Trained")
            if hasattr(model, 'feature_importance') and model.feature_importance:
                # Show top 3 features
                sorted_features = sorted(
                    model.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                print(f"  Top Features:")
                for feat, importance in sorted_features:
                    print(f"    - {feat}: {importance:.4f}")
        else:
            print(f"\n{model_name}:")
            print(f"  Status: ⚠️  Not trained")

    print("\n" + "="*70)
    print("🎯 MODELS READY FOR PREDICTIONS!")
    print("="*70)


if __name__ == "__main__":
    main()
