#!/usr/bin/env python3
"""
Training Pipeline for Gradient Boosting Ensemble
================================================

Trains spread, total, and moneyline models on historical NFL game data
with proper validation and performance reporting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, Tuple, List

from gradient_boosting_ensemble import GradientBoostingEnsemble

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """Orchestrates training of all ensemble models"""
    
    def __init__(self, 
                 data_dir: str = 'backtesting_data',
                 output_dir: str = 'models'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.ensemble = GradientBoostingEnsemble(
            spread_model_path=str(self.output_dir / 'spread_ensemble.pkl'),
            total_model_path=str(self.output_dir / 'total_ensemble.pkl'),
            ml_model_path=str(self.output_dir / 'moneyline_ensemble.pkl'),
        )
        
        self.metrics_history = {
            'spread': [],
            'total': [],
            'moneyline': [],
        }
    
    def load_training_data(self) -> pd.DataFrame:
        """Load and combine all backtesting data from parquet or CSV"""
        logger.info("Loading training data...")
        
        # Try to load from parquet first (real data)
        try:
            df = pd.read_parquet('data/referee_conspiracy/crew_game_log.parquet')
            logger.info(f"Loaded {len(df)} real games from crew_game_log.parquet")
            return df
        except FileNotFoundError:
            logger.warning("crew_game_log.parquet not found")
        
        # Fall back to CSV files
        data_files = list(self.data_dir.glob('*.csv'))
        if not data_files:
            logger.error(f"No data files found in {self.data_dir} or data/referee_conspiracy/")
            raise FileNotFoundError("No training data available. Cannot proceed without real data.")
        
        dfs = []
        for file in data_files:
            try:
                df = pd.read_csv(file)
                logger.info(f"Loaded {len(df)} records from {file.name}")
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file.name}: {e}")
        
        if not dfs:
            raise FileNotFoundError("Failed to load any CSV files")
        
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total records: {len(df)}")
        return df
    
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Prepare features for training.
        
        Returns: (feature_df, targets_dict, feature_config)
        """
        logger.info("Preparing features...")
        
        # Create target labels from actual game results
        targets = {}
        df_copy = df.copy()
        
        # Spread: 1 if home covers (wins > spread), 0 if away covers
        margin = df_copy['home_score'] - df_copy['away_score']
        df_copy['spread_result'] = (margin > df_copy['spread_line']).astype(int)
        targets['spread_result'] = df_copy['spread_result']
        
        # Total: 1 if OVER, 0 if UNDER
        df_copy['total_result'] = (df_copy['home_score'] + df_copy['away_score'] > df_copy['total_line']).astype(int)
        targets['total_result'] = df_copy['total_result']
        
        # Moneyline: 1 if home wins, 0 if away wins
        df_copy['ml_result'] = (df_copy['home_score'] > df_copy['away_score']).astype(int)
        targets['ml_result'] = df_copy['ml_result']
        
        # Select numeric features (exclude IDs and text)
        exclude_cols = {
            'game_id', 'away_team', 'home_team', 'gameday', 'stadium',
            'weekday', 'gametime', 'game_type', 'season',
            'home_score', 'away_score',  # Actual outcomes
            'spread_result', 'total_result', 'ml_result',  # Our targets
            'BJ', 'DJ', 'FJ', 'LJ', 'SJ', 'U', 'referee',  # Ref crew identifiers
        }
        
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        # Remove columns with too many NaNs
        feature_cols = [c for c in feature_cols if df_copy[c].isna().sum() / len(df_copy) < 0.2]
        
        logger.info(f"Selected {len(feature_cols)} features from {len(numeric_cols)} numeric columns")
        logger.info(f"Features: {feature_cols}")
        
        # Prepare feature matrix
        X = df_copy[feature_cols].fillna(df_copy[feature_cols].mean())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return X, targets, {
            'feature_cols': feature_cols,
            'num_features': len(feature_cols),
            'training_samples': len(X),
        }
    
    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """Train all three ensemble models"""
        
        X, targets, feature_config = self.prepare_features(df)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_config': feature_config,
            'models': {},
        }
        
        # Train spread model
        logger.info("\n=== TRAINING SPREAD MODEL ===")
        y_spread = targets['spread_result']
        metrics = self.ensemble.train_spread_model(X, y_spread)
        results['models']['spread'] = {
            'auc': metrics.auc_score,
            'accuracy': metrics.accuracy,
            'f1': metrics.f1,
            'rmse': metrics.rmse,
            'mae': metrics.mae,
            'cv_mean': metrics.cross_val_mean,
            'cv_std': metrics.cross_val_std,
        }
        
        # Top features
        top_features = self.ensemble.get_feature_importance('spread', top_n=10)
        results['models']['spread']['top_features'] = top_features
        logger.info(f"\nTop Spread Features: {top_features}")
        
        # Train total model
        logger.info("\n=== TRAINING TOTAL MODEL ===")
        y_total = targets['total_result']
        metrics = self.ensemble.train_total_model(X, y_total)
        results['models']['total'] = {
            'auc': metrics.auc_score,
            'accuracy': metrics.accuracy,
            'f1': metrics.f1,
            'rmse': metrics.rmse,
            'mae': metrics.mae,
            'cv_mean': metrics.cross_val_mean,
            'cv_std': metrics.cross_val_std,
        }
        
        top_features = self.ensemble.get_feature_importance('total', top_n=10)
        results['models']['total']['top_features'] = top_features
        logger.info(f"\nTop Total Features: {top_features}")
        
        # Train moneyline model
        logger.info("\n=== TRAINING MONEYLINE MODEL ===")
        y_ml = targets['ml_result']
        metrics = self.ensemble.train_moneyline_model(X, y_ml)
        results['models']['moneyline'] = {
            'auc': metrics.auc_score,
            'accuracy': metrics.accuracy,
            'f1': metrics.f1,
            'rmse': metrics.rmse,
            'mae': metrics.mae,
            'cv_mean': metrics.cross_val_mean,
            'cv_std': metrics.cross_val_std,
        }
        
        top_features = self.ensemble.get_feature_importance('moneyline', top_n=10)
        results['models']['moneyline']['top_features'] = top_features
        logger.info(f"\nTop Moneyline Features: {top_features}")
        
        return results
    
    def run_training(self) -> Dict:
        """Complete training pipeline"""
        logger.info("Starting ensemble training pipeline...")
        
        # Load data
        df = self.load_training_data()
        
        # Train models
        results = self.train_all_models(df)
        
        # Save models
        logger.info("\nSaving trained models...")
        self.ensemble.save_models()
        
        # Save metrics
        metrics_file = self.output_dir / 'training_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
        
        return results
    
    def print_summary(self, results: Dict):
        """Print training summary"""
        print("\n" + "="*60)
        print("ENSEMBLE TRAINING SUMMARY")
        print("="*60)
        
        print(f"\nTraining Time: {results['timestamp']}")
        config = results['data_config']
        print(f"Training Samples: {config['training_samples']}")
        print(f"Features: {config['num_features']}")
        
        print("\n" + "-"*60)
        print("MODEL PERFORMANCE")
        print("-"*60)
        
        for model_name, metrics in results['models'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  AUC Score:    {metrics['auc']:.4f}")
            print(f"  Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  F1 Score:     {metrics['f1']:.4f}")
            print(f"  Cross-Val:    {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
            print(f"  RMSE:         {metrics['rmse']:.4f}")
            print(f"  MAE:          {metrics['mae']:.4f}")


if __name__ == '__main__':
    trainer = EnsembleTrainer()
    results = trainer.run_training()
    trainer.print_summary(results)
