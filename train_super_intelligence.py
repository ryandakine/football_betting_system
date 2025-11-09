#!/usr/bin/env python3
"""
Train the 10-Model Super Intelligence System
Run this to train all models on historical NCAA data
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
    """Main training pipeline"""
    print("\n" + "="*70)
    print("🤖 NCAA 10-MODEL SUPER INTELLIGENCE TRAINING")
    print("="*70)

    # Initialize feature engineer
    logger.info("Initializing feature engineering...")
    feature_engineer = NCAAFeatureEngineer(
        data_dir="data/football/historical/ncaaf"
    )

    # Initialize model orchestrator
    logger.info("Initializing model orchestrator...")
    orchestrator = SuperIntelligenceOrchestrator(
        models_dir="models/ncaa"
    )

    # Determine which seasons to train on
    seasons_input = input("\nWhich seasons to train on? (e.g., 2023,2024 or 2015-2024): ").strip()

    if '-' in seasons_input:
        # Range specified
        start, end = seasons_input.split('-')
        seasons = list(range(int(start), int(end) + 1))
    elif ',' in seasons_input:
        # List specified
        seasons = [int(s.strip()) for s in seasons_input.split(',')]
    else:
        # Single season or default
        if seasons_input:
            seasons = [int(seasons_input)]
        else:
            seasons = [2023, 2024]  # Default

    print(f"\nTraining on seasons: {seasons}")
    confirm = input("Proceed? (y/n): ").strip().lower()

    if confirm != 'y':
        print("Training cancelled.")
        return

    # Train all models
    try:
        orchestrator.train_all(feature_engineer, seasons=seasons)

        # Save trained models
        print("\n💾 Saving models...")
        orchestrator.save_all()

        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print("\nAll 10 models trained and saved to: models/ncaa/")
        print("\nModels trained:")
        for i, (name, model) in enumerate(orchestrator.models.items(), 1):
            status = "✅" if model.is_trained else "❌"
            print(f"  {i:2d}. {status} {model.name}")

        print("\n🎯 Next steps:")
        print("1. Test models: python test_super_intelligence.py")
        print("2. Run agent with models: python ncaa_agent.py --manual")
        print("3. Generate picks: Use 'picks' command in agent")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
