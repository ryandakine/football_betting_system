#!/usr/bin/env python3
"""
Enhanced AI Council Training with Referee Features
===================================================

Improved training pipeline with:
1. Referee bias features from autopsy reports
2. Better Total Expert model using regression + threshold
3. Specialized models for high/low total games
4. Weather-adjusted scoring predictions
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, mean_absolute_error
import joblib
import os
from datetime import datetime
from extract_referee_training_data import RefereeDataExtractor

class EnhancedNFLAICouncil:
    """
    Enhanced Multi-model AI Council with referee features
    """
    
    def __init__(self):
        # Core models
        self.models = {
            'spread_expert': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
            'contrarian': LogisticRegression(random_state=42, max_iter=1000),
            'home_advantage': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        }
        
        # New: Total prediction models (regression-based)
        self.total_models = {
            'total_regressor': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
            'total_high_games': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
            'total_low_games': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
            'total_weather_adjusted': Ridge(alpha=1.0, random_state=42)
        }
        
        # Referee data extractor
        self.referee_extractor = RefereeDataExtractor()
        
        self.feature_columns = []
        self.total_feature_columns = []
        self.is_trained = False
        
    def load_historical_data(self, filepath='data/nfl_training_data.json'):
        """Load and prepare historical NFL data"""
        
        # Try to load enhanced data first
        enhanced_path = filepath.replace('.json', '_enhanced.json')
        if os.path.exists(enhanced_path):
            print(f"✅ Loading enhanced training data with ALL features...")
            filepath = enhanced_path
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        games = data['games']
        print(f"📊 Loaded {len(games)} historical games")
        
        # Convert to DataFrame
        df = pd.DataFrame(games)
        
        # Filter only games with complete data
        df = df[df['spread'].notna() & df['total'].notna()].copy()
        
        print(f"✅ {len(df)} games with complete betting data")
        
        return df
    
    def engineer_spread_features(self, df):
        """Create features for spread predictions"""
        
        features = df.copy()
        
        # Team encoding (one-hot for each team)
        teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        for team in teams:
            features[f'home_is_{team}'] = (features['home_team'] == team).astype(int)
            features[f'away_is_{team}'] = (features['away_team'] == team).astype(int)
        
        # Week features
        features['week_early'] = (features['week'] <= 6).astype(int)
        features['week_mid'] = ((features['week'] > 6) & (features['week'] <= 12)).astype(int)
        features['week_late'] = (features['week'] > 12).astype(int)
        features['week_playoffs'] = (features['week'] > 18).astype(int)
        
        # Spread features
        features['spread_abs'] = features['spread'].abs()
        features['spread_large'] = (features['spread_abs'] > 7).astype(int)
        features['spread_moderate'] = ((features['spread_abs'] >= 3) & (features['spread_abs'] <= 7)).astype(int)
        features['spread_small'] = (features['spread_abs'] < 3).astype(int)
        
        # Add advanced situational features
        self._add_advanced_features(features, teams)
        
        # Add referee features
        self._add_referee_features(features)
        
        return features
    
    def engineer_total_features(self, df):
        """Create features specifically for total (over/under) predictions"""
        
        features = df.copy()
        
        # Team scoring features (historical averages)
        teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        
        # Calculate rolling team scoring averages
        for team in teams:
            # Home scoring average (last 5 games)
            home_mask = features['home_team'] == team
            if home_mask.any():
                features.loc[home_mask, f'{team}_home_ppg_l5'] = (
                    features.loc[home_mask, 'home_score'].rolling(5, min_periods=1).mean()
                )
            
            # Away scoring average (last 5 games)
            away_mask = features['away_team'] == team
            if away_mask.any():
                features.loc[away_mask, f'{team}_away_ppg_l5'] = (
                    features.loc[away_mask, 'away_score'].rolling(5, min_periods=1).mean()
                )
            
            # Points allowed (defense)
            if home_mask.any():
                features.loc[home_mask, f'{team}_home_pa_l5'] = (
                    features.loc[home_mask, 'away_score'].rolling(5, min_periods=1).mean()
                )
            if away_mask.any():
                features.loc[away_mask, f'{team}_away_pa_l5'] = (
                    features.loc[away_mask, 'home_score'].rolling(5, min_periods=1).mean()
                )
        
        # Fill team-specific rolling stats with defaults
        for col in features.columns:
            if '_ppg_l5' in col or '_pa_l5' in col:
                features[col] = features[col].fillna(21.0)  # NFL average ~21 ppg
        
        # Total line features
        features['total_high'] = (features['total'] > 47).astype(int)
        features['total_low'] = (features['total'] < 42).astype(int)
        features['total_moderate'] = ((features['total'] >= 42) & (features['total'] <= 47)).astype(int)
        
        # Weather impact on scoring
        if 'is_dome' in features.columns:
            features['weather_scoring_penalty'] = features.apply(
                lambda row: 0 if row.get('is_dome') else (
                    3.0 if pd.notna(row.get('temperature')) and row.get('temperature') < 20 else
                    2.0 if pd.notna(row.get('wind_speed')) and row.get('wind_speed') > 20 else
                    1.5 if pd.notna(row.get('precipitation')) and row.get('precipitation') > 0 else 0
                ), axis=1
            )
        else:
            features['weather_scoring_penalty'] = 0
        
        # Pace of play (if available from EPA data)
        if 'home_epa_per_play' in features.columns:
            features['combined_offensive_pace'] = (
                features['home_epa_per_play'].fillna(0) + 
                features['away_epa_per_play'].fillna(0)
            )
        else:
            features['combined_offensive_pace'] = 0
        
        # Week features (scoring changes throughout season)
        features['week_early'] = (features['week'] <= 6).astype(int)
        features['week_mid'] = ((features['week'] > 6) & (features['week'] <= 12)).astype(int)
        features['week_late'] = (features['week'] > 12).astype(int)
        
        # Division games (typically lower scoring)
        if 'is_division_game' in features.columns:
            features['division_game'] = features['is_division_game'].astype(int)
        else:
            features['division_game'] = 0
        
        # Primetime games (can affect scoring)
        if 'is_primetime' in features.columns:
            features['primetime'] = features['is_primetime'].astype(int)
        else:
            features['primetime'] = 0
        
        # Add referee features (refs impact penalties and scoring)
        self._add_referee_features(features)
        
        return features
    
    def _add_advanced_features(self, features, teams):
        """Add advanced statistical features"""
        
        # Weather impact
        if 'is_dome' in features.columns:
            features['weather_factor'] = features.apply(
                lambda row: 0 if row.get('is_dome') else (
                    1.0 if pd.notna(row.get('temperature')) and row.get('temperature') < 32 else
                    1.0 if pd.notna(row.get('wind_speed')) and row.get('wind_speed') > 15 else
                    0.5 if pd.notna(row.get('precipitation')) and row.get('precipitation') > 0 else 0
                ), axis=1
            )
        else:
            features['weather_factor'] = 0
        
        # Injury impact
        if 'home_injury_score' in features.columns:
            features['home_injury_impact'] = features['home_injury_score'].fillna(0)
            features['away_injury_impact'] = features['away_injury_score'].fillna(0)
            features['injury_advantage'] = features['away_injury_impact'] - features['home_injury_impact']
        else:
            features['home_injury_impact'] = 0
            features['away_injury_impact'] = 0
            features['injury_advantage'] = 0
        
        # Rest advantage
        if 'home_rest_days' in features.columns:
            features['rest_differential'] = features['home_rest_days'] - features['away_rest_days']
            features['home_short_rest'] = (features['home_rest_days'] < 6).astype(int)
            features['away_short_rest'] = (features['away_rest_days'] < 6).astype(int)
        else:
            features['rest_differential'] = 0
            features['home_short_rest'] = 0
            features['away_short_rest'] = 0
        
        # Travel fatigue
        if 'away_travel_distance' in features.columns:
            features['travel_fatigue'] = features['away_travel_distance'].fillna(0) / 1000.0
            features['cross_country'] = (features['away_travel_distance'] > 2000).astype(int)
        else:
            features['travel_fatigue'] = 0
            features['cross_country'] = 0
        
        # Division rivalry
        if 'is_division_game' in features.columns:
            features['division_game'] = features['is_division_game'].astype(int)
        else:
            features['division_game'] = 0
        
        # Primetime
        if 'is_primetime' in features.columns:
            features['primetime'] = features['is_primetime'].astype(int)
        else:
            features['primetime'] = 0
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
    
    def _add_referee_features(self, features):
        """Add referee bias features from autopsy data"""
        
        # Load referee data
        if not self.referee_extractor.referee_profiles:
            self.referee_extractor.parse_all_reports()
        
        # Initialize referee feature columns
        ref_features = {
            'ref_avg_margin': 0.0,
            'ref_avg_penalties': 6.0,
            'ref_penalty_diff': 0.0,
            'ref_odds_delta': 0.0,
            'ref_overtime_rate': 6.0,
            'ref_is_high_penalties': 0,
            'ref_is_low_flags': 0,
            'ref_is_overtime_frequent': 0,
            'ref_home_advantage': 0.0,
            'ref_penalty_advantage': 0.0
        }
        
        for col, default_val in ref_features.items():
            if col not in features.columns:
                features[col] = default_val
        
        # Populate referee features for each game
        if 'referee_name' in features.columns:
            for idx, row in features.iterrows():
                game_ref_features = self.referee_extractor.create_game_referee_features({
                    'home_team': row.get('home_team'),
                    'away_team': row.get('away_team'),
                    'referee': row.get('referee_name', 'Unknown'),
                    'season': row.get('season'),
                    'week': row.get('week')
                })
                
                for feat_name, feat_value in game_ref_features.items():
                    if feat_name in ref_features:
                        features.at[idx, feat_name] = feat_value
    
    def train_council(self, df):
        """Train all models in the AI Council"""
        
        print(f"\n🧠 TRAINING ENHANCED AI COUNCIL")
        print(f"=" * 60)
        
        # Engineer spread features
        spread_features = self.engineer_spread_features(df)
        
        # Select feature columns for spread models
        exclude_cols = ['game_id', 'date', 'home_team', 'away_team', 'venue', 
                       'home_score', 'away_score', 'home_winner', 'spread_result', 
                       'total_result', 'season', 'week', 'spread', 'total', 
                       'attendance', 'neutral_site', 'temperature', 'wind_speed',
                       'precipitation', 'home_injury_score', 'away_injury_score',
                       'home_rest_days', 'away_rest_days', 'away_travel_distance',
                       'is_dome', 'is_division_game', 'is_primetime', 'weather_impact',
                       'injury_differential', 'rest_advantage', 'referee_name', 'crew_id']
        
        self.feature_columns = [col for col in spread_features.columns if col not in exclude_cols and not col.endswith('_ppg_l5') and not col.endswith('_pa_l5')]
        
        X_spread = spread_features[self.feature_columns].values
        
        # Train spread expert
        y_spread = spread_features['spread_result'].values
        X_train, X_test, y_train, y_test = train_test_split(X_spread, y_spread, test_size=0.2, random_state=42)
        
        print(f"\n📈 Training Spread Expert...")
        self.models['spread_expert'].fit(X_train, y_train)
        spread_acc = accuracy_score(y_test, self.models['spread_expert'].predict(X_test))
        print(f"   Accuracy: {spread_acc:.2%}")
        
        # Train contrarian
        y_contrarian = 1 - spread_features['home_winner'].values
        X_train, X_test, y_train, y_test = train_test_split(X_spread, y_contrarian, test_size=0.2, random_state=42)
        
        print(f"\n📈 Training Contrarian Model...")
        self.models['contrarian'].fit(X_train, y_train)
        contrarian_acc = accuracy_score(y_test, self.models['contrarian'].predict(X_test))
        print(f"   Accuracy: {contrarian_acc:.2%}")
        
        # Train home advantage
        y_home = spread_features['home_winner'].values
        X_train, X_test, y_train, y_test = train_test_split(X_spread, y_home, test_size=0.2, random_state=42)
        
        print(f"\n📈 Training Home Advantage Model...")
        self.models['home_advantage'].fit(X_train, y_train)
        home_acc = accuracy_score(y_test, self.models['home_advantage'].predict(X_test))
        print(f"   Accuracy: {home_acc:.2%}")
        
        # Train Total Expert models
        print(f"\n\n🎯 TRAINING TOTAL EXPERT MODELS")
        print(f"=" * 60)
        
        total_features = self.engineer_total_features(df)
        
        # Select features for total models
        self.total_feature_columns = [col for col in total_features.columns 
                                      if col not in exclude_cols and col != 'total_result']
        
        X_total = total_features[self.total_feature_columns].values
        
        # 1. Total Regressor (predicts actual total points)
        y_total_points = (total_features['home_score'] + total_features['away_score']).values
        X_train, X_test, y_train, y_test = train_test_split(X_total, y_total_points, test_size=0.2, random_state=42)
        
        print(f"\n📈 Training Total Points Regressor...")
        self.total_models['total_regressor'].fit(X_train, y_train)
        predictions = self.total_models['total_regressor'].predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        print(f"   Mean Absolute Error: {mae:.2f} points")
        
        # 2. High Total Games Specialist
        high_total_mask = total_features['total'] > 47
        if high_total_mask.sum() > 100:  # Need enough samples
            X_high = total_features.loc[high_total_mask, self.total_feature_columns].values
            y_high = total_features.loc[high_total_mask, 'total_result'].values
            X_train, X_test, y_train, y_test = train_test_split(X_high, y_high, test_size=0.2, random_state=42)
            
            print(f"\n📈 Training High Total Games Specialist ({len(X_high)} games)...")
            self.total_models['total_high_games'].fit(X_train, y_train)
            high_acc = accuracy_score(y_test, self.total_models['total_high_games'].predict(X_test))
            print(f"   Accuracy: {high_acc:.2%}")
        else:
            print(f"\n⚠️ Skipping High Total Specialist (insufficient data)")
        
        # 3. Low Total Games Specialist
        low_total_mask = total_features['total'] < 42
        if low_total_mask.sum() > 100:
            X_low = total_features.loc[low_total_mask, self.total_feature_columns].values
            y_low = total_features.loc[low_total_mask, 'total_result'].values
            X_train, X_test, y_train, y_test = train_test_split(X_low, y_low, test_size=0.2, random_state=42)
            
            print(f"\n📈 Training Low Total Games Specialist ({len(X_low)} games)...")
            self.total_models['total_low_games'].fit(X_train, y_train)
            low_acc = accuracy_score(y_test, self.total_models['total_low_games'].predict(X_test))
            print(f"   Accuracy: {low_acc:.2%}")
        else:
            print(f"\n⚠️ Skipping Low Total Specialist (insufficient data)")
        
        # 4. Weather-Adjusted Total Model
        y_total_result = total_features['total_result'].values
        X_train, X_test, y_train, y_test = train_test_split(X_total, y_total_result, test_size=0.2, random_state=42)
        
        print(f"\n📈 Training Weather-Adjusted Total Model...")
        self.total_models['total_weather_adjusted'].fit(X_train, y_train)
        weather_predictions = np.where(self.total_models['total_weather_adjusted'].predict(X_test) > 0.5, 1, 0)
        weather_acc = accuracy_score(y_test, weather_predictions)
        print(f"   Accuracy: {weather_acc:.2%}")
        
        self.is_trained = True
        
        print(f"\n✅ Enhanced AI Council training complete!")
        print(f"\n📊 MODEL PERFORMANCE SUMMARY")
        print(f"=" * 60)
        print(f"   Spread Expert: {spread_acc:.2%}")
        print(f"   Total Regressor: {mae:.2f} MAE")
        print(f"   Weather-Adjusted Total: {weather_acc:.2%}")
        print(f"   Contrarian: {contrarian_acc:.2%}")
        print(f"   Home Advantage: {home_acc:.2%}")
        
    def save_models(self, model_dir='models'):
        """Save trained models to disk"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save spread models
        for name, model in self.models.items():
            filepath = os.path.join(model_dir, f'{name}_nfl_model.pkl')
            joblib.dump(model, filepath)
            print(f"💾 Saved {name} to {filepath}")
        
        # Save total models
        for name, model in self.total_models.items():
            filepath = os.path.join(model_dir, f'{name}_nfl_model.pkl')
            joblib.dump(model, filepath)
            print(f"💾 Saved {name} to {filepath}")
        
        # Save feature columns
        feature_path = os.path.join(model_dir, 'nfl_features.json')
        with open(feature_path, 'w') as f:
            json.dump({
                'spread_features': self.feature_columns,
                'total_features': self.total_feature_columns
            }, f, indent=2)
        print(f"💾 Saved feature lists to {feature_path}")

def main():
    """Main training pipeline"""
    
    print("🏈 ENHANCED NFL AI COUNCIL TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize Enhanced AI Council
    council = EnhancedNFLAICouncil()
    
    # Load historical data
    df = council.load_historical_data()
    
    # Train all models
    council.train_council(df)
    
    # Save models
    council.save_models()
    
    print(f"\n✅ Training complete! Enhanced AI Council ready for predictions.")
    print(f"   Referee features integrated")
    print(f"   Total Expert improved with regression + specialists")

if __name__ == "__main__":
    main()
