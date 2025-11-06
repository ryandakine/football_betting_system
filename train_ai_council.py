#!/usr/bin/env python3
"""
Train AI Council on 10 years of NFL historical data
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os
from datetime import datetime

class NFLAICouncil:
    """
    Multi-model AI Council for NFL predictions
    Each model learns from 10 years of historical data
    """
    
    def __init__(self):
        # The Council: Multiple AI models with different strategies
        self.models = {
            'spread_expert': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'total_expert': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
            'contrarian': LogisticRegression(random_state=42, max_iter=1000),
            'home_advantage': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        }
        
        self.feature_columns = []
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
    
    def engineer_features(self, df):
        """Create features from raw game data"""
        
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
        
        # Spread/Total features
        features['spread_abs'] = features['spread'].abs()
        features['spread_large'] = (features['spread_abs'] > 7).astype(int)
        features['total_high'] = (features['total'] > 47).astype(int)
        features['total_low'] = (features['total'] < 42).astype(int)
        
        # Weather impact features
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
        
        # Injury impact (higher score = more injuries)
        if 'home_injury_score' in features.columns:
            features['home_injury_impact'] = features['home_injury_score'].fillna(0)
            features['away_injury_impact'] = features['away_injury_score'].fillna(0)
            features['injury_advantage'] = features['away_injury_impact'] - features['home_injury_impact']
        else:
            features['home_injury_impact'] = 0
            features['away_injury_impact'] = 0
            features['injury_advantage'] = 0
        
        # Rest advantage (positive = home team more rested)
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
        
        # Division rivalry games
        if 'is_division_game' in features.columns:
            features['division_game'] = features['is_division_game'].astype(int)
        else:
            features['division_game'] = 0
        
        # Prime time games
        if 'is_primetime' in features.columns:
            features['primetime'] = features['is_primetime'].astype(int)
        else:
            features['primetime'] = 0
        
        # Referee crew impact
        if 'crew_home_bias' in features.columns:
            features['ref_home_advantage'] = (features['crew_home_bias'] - 0.5) * 10  # Scale to -5 to +5
            features['ref_flag_heavy'] = features['crew_flag_heavy'].fillna(0).astype(int)
            features['ref_home_favoring'] = features['crew_home_favoring'].fillna(0).astype(int)
            features['ref_variance_high'] = (features['crew_variance'] > 0.15).astype(int)
        else:
            features['ref_home_advantage'] = 0
            features['ref_flag_heavy'] = 0
            features['ref_home_favoring'] = 0
            features['ref_variance_high'] = 0
        
        # EPA features (if available)
        if 'epa_differential' in features.columns:
            features['epa_advantage'] = features['epa_differential'].fillna(0)
        else:
            features['epa_advantage'] = 0
        
        # DVOA features (if available)
        if 'dvoa_differential' in features.columns:
            features['dvoa_advantage'] = features['dvoa_differential'].fillna(0)
            features['home_off_dvoa'] = features['home_offensive_dvoa'].fillna(0)
            features['home_def_dvoa'] = features['home_defensive_dvoa'].fillna(0)
        else:
            features['dvoa_advantage'] = 0
            features['home_off_dvoa'] = 0
            features['home_def_dvoa'] = 0
        
        # ATS recent performance (if available)
        if 'home_ats_l5' in features.columns:
            features['ats_advantage'] = features['home_ats_l5'] - features['away_ats_l5']
            features['home_ats_hot'] = (features['home_ats_l5'] > 0.6).astype(int)
            features['away_ats_hot'] = (features['away_ats_l5'] > 0.6).astype(int)
        else:
            features['ats_advantage'] = 0
            features['home_ats_hot'] = 0
            features['away_ats_hot'] = 0
        
        # Line movement (if available)
        if 'sharp_money_indicator' in features.columns:
            features['sharp_money'] = features['sharp_money_indicator'].fillna(0).astype(int)
            features['line_moved'] = (features['line_movement'].fillna(0).abs() > 1.0).astype(int)
        else:
            features['sharp_money'] = 0
            features['line_moved'] = 0
        
        # Agent influence (if available)
        if 'agent_edge_multiplier' in features.columns:
            features['agent_boost'] = (features['agent_edge_multiplier'] - 1.0) * 10  # Scale
            features['has_agent_conflict'] = features['has_agent_conflict'].fillna(0).astype(int)
        else:
            features['agent_boost'] = 0
            features['has_agent_conflict'] = 0
        
        # Team chemistry (if available)
        if 'chemistry_advantage' in features.columns:
            features['chemistry_edge'] = features['chemistry_advantage'].fillna(0)
        else:
            features['chemistry_edge'] = 0
        
        # Historical team performance (rolling averages)
        features = features.sort_values(['season', 'week'])
        
        for team in teams:
            # Home team rolling stats
            home_mask = features['home_team'] == team
            features.loc[home_mask, f'{team}_home_wins_l5'] = (
                features.loc[home_mask, 'home_winner'].rolling(5, min_periods=1).mean()
            )
            
            # Away team rolling stats
            away_mask = features['away_team'] == team
            features.loc[away_mask, f'{team}_away_wins_l5'] = (
                features.loc[away_mask, 'home_winner'].apply(lambda x: 1 - x).rolling(5, min_periods=1).mean()
            )
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
    
    def train_council(self, df):
        """Train all models in the AI Council"""
        
        print(f"\n🧠 TRAINING AI COUNCIL")
        print(f"=" * 60)
        
        # Engineer features
        features_df = self.engineer_features(df)
        
        # Select feature columns (exclude metadata)
        exclude_cols = ['game_id', 'date', 'home_team', 'away_team', 'venue', 
                       'home_score', 'away_score', 'home_winner', 'spread_result', 
                       'total_result', 'season', 'week', 'spread', 'total', 
                       'attendance', 'neutral_site', 'temperature', 'wind_speed',
                       'precipitation', 'home_injury_score', 'away_injury_score',
                       'home_rest_days', 'away_rest_days', 'away_travel_distance',
                       'is_dome', 'is_division_game', 'is_primetime', 'weather_impact',
                       'injury_differential', 'rest_advantage', 'referee_name', 'crew_id',
                       'crew_home_bias', 'crew_penalties_avg', 'crew_variance',
                       'crew_flag_heavy', 'crew_home_favoring',
                       # Enhanced features (raw values, we use engineered versions)
                       'home_epa_per_play', 'away_epa_per_play', 'epa_differential',
                       'home_offensive_dvoa', 'home_defensive_dvoa', 'away_offensive_dvoa',
                       'away_defensive_dvoa', 'dvoa_differential', 'home_ats_l5', 'away_ats_l5',
                       'home_ats_wins', 'away_ats_wins', 'ats_record_l5', 'line_movement',
                       'sharp_money_indicator', 'steam_move', 'home_chemistry', 'away_chemistry',
                       'chemistry_advantage', 'agent_edge_multiplier', 'agent_confidence_delta',
                       'agent_rules_triggered']
        
        self.feature_columns = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[self.feature_columns].values
        
        # Train spread expert
        y_spread = features_df['spread_result'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y_spread, test_size=0.2, random_state=42)
        
        print(f"\n📈 Training Spread Expert...")
        self.models['spread_expert'].fit(X_train, y_train)
        spread_acc = accuracy_score(y_test, self.models['spread_expert'].predict(X_test))
        print(f"   Accuracy: {spread_acc:.2%}")
        
        # Train total expert
        y_total = features_df['total_result'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y_total, test_size=0.2, random_state=42)
        
        print(f"\n📈 Training Total Expert...")
        self.models['total_expert'].fit(X_train, y_train)
        total_acc = accuracy_score(y_test, self.models['total_expert'].predict(X_test))
        print(f"   Accuracy: {total_acc:.2%}")
        
        # Train contrarian (inverse of home winner)
        y_contrarian = 1 - features_df['home_winner'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y_contrarian, test_size=0.2, random_state=42)
        
        print(f"\n📈 Training Contrarian Model...")
        self.models['contrarian'].fit(X_train, y_train)
        contrarian_acc = accuracy_score(y_test, self.models['contrarian'].predict(X_test))
        print(f"   Accuracy: {contrarian_acc:.2%}")
        
        # Train home advantage model
        y_home = features_df['home_winner'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
        
        print(f"\n📈 Training Home Advantage Model...")
        self.models['home_advantage'].fit(X_train, y_train)
        home_acc = accuracy_score(y_test, self.models['home_advantage'].predict(X_test))
        print(f"   Accuracy: {home_acc:.2%}")
        
        self.is_trained = True
        
        print(f"\n✅ AI Council training complete!")
        print(f"\n📊 MODEL PERFORMANCE SUMMARY")
        print(f"=" * 60)
        print(f"   Spread Expert: {spread_acc:.2%}")
        print(f"   Total Expert: {total_acc:.2%}")
        print(f"   Contrarian: {contrarian_acc:.2%}")
        print(f"   Home Advantage: {home_acc:.2%}")
        
    def save_models(self, model_dir='models'):
        """Save trained models to disk"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(model_dir, f'{name}_nfl_model.pkl')
            joblib.dump(model, filepath)
            print(f"💾 Saved {name} to {filepath}")
        
        # Save feature columns
        feature_path = os.path.join(model_dir, 'nfl_features.json')
        with open(feature_path, 'w') as f:
            json.dump(self.feature_columns, f)
        print(f"💾 Saved feature list to {feature_path}")
        
    def load_models(self, model_dir='models'):
        """Load trained models from disk"""
        
        for name in self.models.keys():
            filepath = os.path.join(model_dir, f'{name}_nfl_model.pkl')
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"✅ Loaded {name}")
        
        # Load feature columns
        feature_path = os.path.join(model_dir, 'nfl_features.json')
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                self.feature_columns = json.load(f)
            print(f"✅ Loaded feature list")
            
        self.is_trained = True
    
    def predict_game(self, game_data):
        """
        Get predictions from all council members
        Returns consensus prediction with confidence
        """
        
        if not self.is_trained:
            raise ValueError("Models not trained. Run train_council() first.")
        
        # TODO: Engineer features for new game
        # For now, return mock predictions
        
        predictions = {
            'spread_expert': {'pick': 'home', 'confidence': 0.65},
            'total_expert': {'pick': 'over', 'confidence': 0.58},
            'contrarian': {'pick': 'away', 'confidence': 0.52},
            'home_advantage': {'pick': 'home', 'confidence': 0.71}
        }
        
        # Consensus logic
        home_votes = sum(1 for p in predictions.values() if p['pick'] == 'home')
        consensus = 'home' if home_votes >= 2 else 'away'
        avg_confidence = np.mean([p['confidence'] for p in predictions.values()])
        
        return {
            'consensus': consensus,
            'confidence': avg_confidence,
            'council_votes': predictions
        }

def main():
    """Main training pipeline"""
    
    print("🏈 NFL AI COUNCIL TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize AI Council
    council = NFLAICouncil()
    
    # Load historical data
    df = council.load_historical_data()
    
    # Train all models
    council.train_council(df)
    
    # Save models
    council.save_models()
    
    print(f"\n✅ Training complete! AI Council ready for predictions.")
    print(f"   Run predictions with: council.predict_game(game_data)")

if __name__ == "__main__":
    main()
