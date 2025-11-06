#!/usr/bin/env python3
"""
Gradient Boosting Ensemble for NFL Predictions
===============================================

Uses XGBoost and LightGBM for superior predictive power:
- Handles non-linear relationships better than linear models
- Built-in feature importance analysis
- Proper cross-validation and regularization
- Ensemble voting for consensus predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, 
    mean_squared_error, mean_absolute_error
)
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_name: str
    auc_score: float
    accuracy: float
    f1: float
    rmse: float
    mae: float
    cross_val_mean: float
    cross_val_std: float


class GradientBoostingEnsemble:
    """
    Production NFL prediction ensemble using XGBoost + LightGBM
    """
    
    def __init__(self, 
                 spread_model_path: str = 'models/spread_ensemble.pkl',
                 total_model_path: str = 'models/total_ensemble.pkl',
                 ml_model_path: str = 'models/moneyline_ensemble.pkl'):
        
        self.spread_model_path = Path(spread_model_path)
        self.total_model_path = Path(total_model_path)
        self.ml_model_path = Path(ml_model_path)
        
        self.spread_models = {}
        self.total_models = {}
        self.ml_models = {}
        
        self.spread_scaler = StandardScaler()
        self.total_scaler = StandardScaler()
        self.ml_scaler = StandardScaler()
        
        self.feature_importance_spread = {}
        self.feature_importance_total = {}
        self.feature_importance_ml = {}
        
    def train_spread_model(self, X: pd.DataFrame, y: pd.Series, 
                          cv_folds: int = 5) -> ModelMetrics:
        """
        Train ensemble for spread prediction (home cover or away cover).
        
        y: 1 = home cover, 0 = away cover
        """
        logger.info(f"Training spread ensemble with {len(X)} samples")
        
        # Preprocessing
        X_scaled = self.spread_scaler.fit_transform(X)
        
        # XGBoost classifier
        if HAS_XGBOOST:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
            )
            xgb_model.fit(X_scaled, y, verbose=0)
            self.spread_models['xgboost'] = xgb_model
            logger.info("XGBoost spread model trained")
        
        # LightGBM classifier
        if HAS_LIGHTGBM:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(X_scaled, y)
            self.spread_models['lightgbm'] = lgb_model
            logger.info("LightGBM spread model trained")
        
        # Gradient Boosting from sklearn as fallback
        from sklearn.ensemble import GradientBoostingClassifier
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        gb_model.fit(X_scaled, y)
        self.spread_models['sklearn_gb'] = gb_model
        logger.info("Sklearn GradientBoosting spread model trained")
        
        # Feature importance (average across models)
        importances = []
        if 'xgboost' in self.spread_models:
            importances.append(self.spread_models['xgboost'].feature_importances_)
        if 'lightgbm' in self.spread_models:
            importances.append(self.spread_models['lightgbm'].feature_importances_)
        importances.append(self.spread_models['sklearn_gb'].feature_importances_)
        
        avg_importance = np.mean(importances, axis=0)
        self.feature_importance_spread = dict(zip(X.columns, avg_importance))
        
        # Evaluate
        y_pred_proba = self._ensemble_predict_proba(X_scaled, 'spread')
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = ModelMetrics(
            model_name='spread_ensemble',
            auc_score=roc_auc_score(y, y_pred_proba),
            accuracy=accuracy_score(y, y_pred),
            f1=f1_score(y, y_pred, zero_division=0),
            rmse=np.sqrt(mean_squared_error(y, y_pred_proba)),
            mae=mean_absolute_error(y, y_pred_proba),
            cross_val_mean=0.0,
            cross_val_std=0.0,
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.spread_models['sklearn_gb'], X_scaled, y, 
            cv=cv, scoring='roc_auc'
        )
        metrics.cross_val_mean = scores.mean()
        metrics.cross_val_std = scores.std()
        
        logger.info(f"Spread AUC: {metrics.auc_score:.4f}, CV: {metrics.cross_val_mean:.4f} ± {metrics.cross_val_std:.4f}")
        
        return metrics
    
    def train_total_model(self, X: pd.DataFrame, y: pd.Series,
                         cv_folds: int = 5) -> ModelMetrics:
        """
        Train ensemble for total prediction (OVER or UNDER).
        
        y: 1 = OVER, 0 = UNDER
        """
        logger.info(f"Training total ensemble with {len(X)} samples")
        
        X_scaled = self.total_scaler.fit_transform(X)
        
        # XGBoost
        if HAS_XGBOOST:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
            )
            xgb_model.fit(X_scaled, y, verbose=0)
            self.total_models['xgboost'] = xgb_model
        
        # LightGBM
        if HAS_LIGHTGBM:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(X_scaled, y)
            self.total_models['lightgbm'] = lgb_model
        
        # Sklearn GradientBoosting
        from sklearn.ensemble import GradientBoostingClassifier
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        gb_model.fit(X_scaled, y)
        self.total_models['sklearn_gb'] = gb_model
        
        # Feature importance
        importances = []
        if 'xgboost' in self.total_models:
            importances.append(self.total_models['xgboost'].feature_importances_)
        if 'lightgbm' in self.total_models:
            importances.append(self.total_models['lightgbm'].feature_importances_)
        importances.append(self.total_models['sklearn_gb'].feature_importances_)
        
        avg_importance = np.mean(importances, axis=0)
        self.feature_importance_total = dict(zip(X.columns, avg_importance))
        
        # Evaluate
        y_pred_proba = self._ensemble_predict_proba(X_scaled, 'total')
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = ModelMetrics(
            model_name='total_ensemble',
            auc_score=roc_auc_score(y, y_pred_proba),
            accuracy=accuracy_score(y, y_pred),
            f1=f1_score(y, y_pred, zero_division=0),
            rmse=np.sqrt(mean_squared_error(y, y_pred_proba)),
            mae=mean_absolute_error(y, y_pred_proba),
            cross_val_mean=0.0,
            cross_val_std=0.0,
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.total_models['sklearn_gb'], X_scaled, y,
            cv=cv, scoring='roc_auc'
        )
        metrics.cross_val_mean = scores.mean()
        metrics.cross_val_std = scores.std()
        
        logger.info(f"Total AUC: {metrics.auc_score:.4f}, CV: {metrics.cross_val_mean:.4f} ± {metrics.cross_val_std:.4f}")
        
        return metrics
    
    def train_moneyline_model(self, X: pd.DataFrame, y: pd.Series,
                             cv_folds: int = 5) -> ModelMetrics:
        """
        Train ensemble for moneyline prediction (home win or away win).
        
        y: 1 = home win, 0 = away win
        """
        logger.info(f"Training moneyline ensemble with {len(X)} samples")
        
        X_scaled = self.ml_scaler.fit_transform(X)
        
        # XGBoost
        if HAS_XGBOOST:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
            )
            xgb_model.fit(X_scaled, y, verbose=0)
            self.ml_models['xgboost'] = xgb_model
        
        # LightGBM
        if HAS_LIGHTGBM:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(X_scaled, y)
            self.ml_models['lightgbm'] = lgb_model
        
        # Sklearn GradientBoosting
        from sklearn.ensemble import GradientBoostingClassifier
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        gb_model.fit(X_scaled, y)
        self.ml_models['sklearn_gb'] = gb_model
        
        # Feature importance
        importances = []
        if 'xgboost' in self.ml_models:
            importances.append(self.ml_models['xgboost'].feature_importances_)
        if 'lightgbm' in self.ml_models:
            importances.append(self.ml_models['lightgbm'].feature_importances_)
        importances.append(self.ml_models['sklearn_gb'].feature_importances_)
        
        avg_importance = np.mean(importances, axis=0)
        self.feature_importance_ml = dict(zip(X.columns, avg_importance))
        
        # Evaluate
        y_pred_proba = self._ensemble_predict_proba(X_scaled, 'ml')
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = ModelMetrics(
            model_name='moneyline_ensemble',
            auc_score=roc_auc_score(y, y_pred_proba),
            accuracy=accuracy_score(y, y_pred),
            f1=f1_score(y, y_pred, zero_division=0),
            rmse=np.sqrt(mean_squared_error(y, y_pred_proba)),
            mae=mean_absolute_error(y, y_pred_proba),
            cross_val_mean=0.0,
            cross_val_std=0.0,
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.ml_models['sklearn_gb'], X_scaled, y,
            cv=cv, scoring='roc_auc'
        )
        metrics.cross_val_mean = scores.mean()
        metrics.cross_val_std = scores.std()
        
        logger.info(f"ML AUC: {metrics.auc_score:.4f}, CV: {metrics.cross_val_mean:.4f} ± {metrics.cross_val_std:.4f}")
        
        return metrics
    
    def _ensemble_predict_proba(self, X: np.ndarray, model_type: str) -> np.ndarray:
        """
        Ensemble voting across all models.
        
        Weights models by their expected performance.
        """
        if model_type == 'spread':
            models = self.spread_models
        elif model_type == 'total':
            models = self.total_models
        else:
            models = self.ml_models
        
        predictions = []
        weights = []
        
        # XGBoost gets higher weight if available (usually best performer)
        if 'xgboost' in models:
            predictions.append(models['xgboost'].predict_proba(X)[:, 1])
            weights.append(0.4)
        
        # LightGBM second highest weight
        if 'lightgbm' in models:
            predictions.append(models['lightgbm'].predict_proba(X)[:, 1])
            weights.append(0.35)
        
        # Sklearn as tiebreaker
        if 'sklearn_gb' in models:
            predictions.append(models['sklearn_gb'].predict_proba(X)[:, 1])
            weights.append(0.25)
        
        # Normalize weights
        if weights:
            weights = np.array(weights) / np.sum(weights)
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def predict_spread(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict spread winner with probabilities.
        
        Returns: (predictions, probabilities)
        """
        X_scaled = self.spread_scaler.transform(X)
        proba = self._ensemble_predict_proba(X_scaled, 'spread')
        predictions = (proba >= 0.5).astype(int)
        return predictions, proba
    
    def predict_total(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict total with probabilities (1=OVER, 0=UNDER).
        """
        X_scaled = self.total_scaler.transform(X)
        proba = self._ensemble_predict_proba(X_scaled, 'total')
        predictions = (proba >= 0.5).astype(int)
        return predictions, proba
    
    def predict_moneyline(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict moneyline winner with probabilities (1=home, 0=away).
        """
        X_scaled = self.ml_scaler.transform(X)
        proba = self._ensemble_predict_proba(X_scaled, 'ml')
        predictions = (proba >= 0.5).astype(int)
        return predictions, proba
    
    def save_models(self):
        """Save ensemble to disk"""
        self.spread_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.spread_model_path, 'wb') as f:
            pickle.dump({
                'models': self.spread_models,
                'scaler': self.spread_scaler,
                'importance': self.feature_importance_spread,
            }, f)
        
        with open(self.total_model_path, 'wb') as f:
            pickle.dump({
                'models': self.total_models,
                'scaler': self.total_scaler,
                'importance': self.feature_importance_total,
            }, f)
        
        with open(self.ml_model_path, 'wb') as f:
            pickle.dump({
                'models': self.ml_models,
                'scaler': self.ml_scaler,
                'importance': self.feature_importance_ml,
            }, f)
        
        logger.info(f"Models saved to {self.spread_model_path.parent}")
    
    def load_models(self):
        """Load ensemble from disk"""
        try:
            with open(self.spread_model_path, 'rb') as f:
                data = pickle.load(f)
                self.spread_models = data['models']
                self.spread_scaler = data['scaler']
                self.feature_importance_spread = data['importance']
            
            with open(self.total_model_path, 'rb') as f:
                data = pickle.load(f)
                self.total_models = data['models']
                self.total_scaler = data['scaler']
                self.feature_importance_total = data['importance']
            
            with open(self.ml_model_path, 'rb') as f:
                data = pickle.load(f)
                self.ml_models = data['models']
                self.ml_scaler = data['scaler']
                self.feature_importance_ml = data['importance']
            
            logger.info("Models loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"Model files not found: {e}")
    
    def get_feature_importance(self, model_type: str = 'spread', top_n: int = 15) -> Dict[str, float]:
        """Get top N most important features"""
        if model_type == 'spread':
            importance = self.feature_importance_spread
        elif model_type == 'total':
            importance = self.feature_importance_total
        else:
            importance = self.feature_importance_ml
        
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example: Train ensemble on synthetic data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=500, n_features=50, n_informative=20, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    ensemble = GradientBoostingEnsemble()
    metrics = ensemble.train_spread_model(X_df, y)
    
    print(f"\nMetrics: {metrics}")
