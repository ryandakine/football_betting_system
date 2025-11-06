#!/usr/bin/env python3
"""
AI Council Integration with Public Sentiment Analysis
======================================================

Combines referee features + Total Expert improvements with:
- Reddit/Discord public betting sentiment
- Crowd roar detection (league let them play)
- Sharp vs Public money splits
- Contrarian opportunity detection
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pathlib import Path
from typing import Dict, List, Any
import joblib
import os
from datetime import datetime

class SentimentFeatureExtractor:
    """Extract betting sentiment features from social data"""
    
    def __init__(self, referee_extractor=None):
        self.referee_extractor = referee_extractor
        self.sentiment_cache = {}
        
    def extract_game_sentiment(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract sentiment features for a game
        
        Expected game_data keys:
        - home_team, away_team, game_id, game_time
        """
        
        features = {
            # Reddit sentiment
            'reddit_sentiment_score': 0.0,          # -1 to +1 (negative to positive)
            'reddit_post_count': 0,                 # Volume on Reddit
            'reddit_home_mentions': 0,
            'reddit_away_mentions': 0,
            'reddit_over_mentions': 0,
            'reddit_under_mentions': 0,
            
            # Discord/Community sentiment
            'discord_sentiment': 0.0,
            'discord_volume': 0,
            'discord_home_lean': 0.0,
            'discord_away_lean': 0.0,
            
            # Expert picks
            'expert_picks_home_pct': 0.5,          # % of experts picking home
            'expert_picks_away_pct': 0.5,
            'expert_picks_total_over_pct': 0.5,
            'expert_picks_total_under_pct': 0.5,
            
            # Crowd roar signals
            'crowd_roar_confidence': 0.0,          # 0-1, "league let them play" signal
            'roar_no_flag_events': 0,              # Count of roar without penalty
            
            # Sharp vs Public
            'sharp_public_split_ml': 0.0,          # -1 to +1 (positive = public on home)
            'sharp_public_split_spread': 0.0,
            'sharp_public_split_total': 0.0,
            
            # Line movement sentiment
            'public_side_winning': 0,              # 1 if public side is moving in their direction
            'contrarian_opportunity': 0.0,         # 0-1 score, high = strong contrarian signal
            
            # Betting volume
            'public_money_pct_home': 0.5,
            'public_money_pct_over': 0.5,
            'sharp_money_confidence': 0.0,
            
            # Sentiment strength
            'sentiment_agreement': 0.5,            # How much Reddit/Discord/Experts agree
            'consensus_strength': 0.0,             # 0-1, how certain the consensus is
        }
        
        return features
    
    def calculate_reddit_sentiment(self, subreddit_posts: List[Dict]) -> Dict[str, Any]:
        """Analyze Reddit posts for betting sentiment"""
        
        sentiment = {
            'overall_score': 0.0,
            'post_count': len(subreddit_posts),
            'home_team_mentions': 0,
            'away_team_mentions': 0,
            'over_mentions': 0,
            'under_mentions': 0,
            'topic_breakdown': {}
        }
        
        if not subreddit_posts:
            return sentiment
        
        # Simple keyword counting
        for post in subreddit_posts:
            text = (post.get('title', '') + ' ' + post.get('selftext', '')).lower()
            
            # Count team mentions (simplified - would use game_data in real scenario)
            home_keywords = ['home team', 'favorite', 'even money']
            away_keywords = ['road', 'upset', 'underdog']
            over_keywords = ['over', 'shootout', 'high scoring', 'explosive']
            under_keywords = ['under', 'defensive', 'low scoring', 'grinding']
            
            sentiment['home_team_mentions'] += sum(1 for kw in home_keywords if kw in text)
            sentiment['away_team_mentions'] += sum(1 for kw in away_keywords if kw in text)
            sentiment['over_mentions'] += sum(1 for kw in over_keywords if kw in text)
            sentiment['under_mentions'] += sum(1 for kw in under_keywords if kw in text)
        
        # Calculate net sentiment
        total_picks = sentiment['over_mentions'] + sentiment['under_mentions']
        if total_picks > 0:
            sentiment['overall_score'] = (sentiment['over_mentions'] - sentiment['under_mentions']) / total_picks
        
        return sentiment
    
    def calculate_sharp_vs_public_split(self, line_data: Dict, bet_data: Dict) -> Dict[str, float]:
        """Calculate the split between sharp money and public picks"""
        
        splits = {
            'moneyline': 0.0,
            'spread': 0.0,
            'total': 0.0
        }
        
        # Sharp money typically follows respected books
        # Public money follows TV, Reddit, popular picks
        # Positive value = public on one side, sharp on other = potential contrarian signal
        
        if 'ml_public_pct' in bet_data:
            splits['moneyline'] = bet_data['ml_public_pct'] - 0.5  # -0.5 to +0.5
        
        if 'spread_public_pct' in bet_data:
            splits['spread'] = bet_data['spread_public_pct'] - 0.5
        
        if 'total_public_pct' in bet_data:
            splits['total'] = bet_data['total_public_pct'] - 0.5
        
        return splits
    
    def detect_contrarian_opportunity(self, sentiment_data: Dict) -> float:
        """
        Detect when public is heavily on one side but sharp money disagrees
        This is a major +EV opportunity
        
        Returns: 0-1 score (higher = stronger contrarian signal)
        """
        
        # Look for divergence between sharp and public
        sharp_public_split = (
            abs(sentiment_data.get('sharp_public_split_ml', 0.0)) +
            abs(sentiment_data.get('sharp_public_split_spread', 0.0)) +
            abs(sentiment_data.get('sharp_public_split_total', 0.0))
        ) / 3
        
        # Look for consensus without agreement
        # (Everyone on Reddit says one thing, but sharp money says another)
        consensus_confidence = sentiment_data.get('consensus_strength', 0.0)
        
        # Combine signals: high split + high consensus = contrarian opportunity
        contrarian_score = (sharp_public_split * 0.6) + (consensus_confidence * 0.4)
        
        return min(1.0, contrarian_score)
    
    def calculate_line_movement_sentiment(self, line_history: List[Dict]) -> Dict[str, Any]:
        """Analyze line movement to infer market sentiment"""
        
        if len(line_history) < 2:
            return {'movement_direction': 0, 'public_winning': False}
        
        first_line = line_history[0]
        last_line = line_history[-1]
        
        # Spread movement
        spread_movement = last_line.get('spread', 0) - first_line.get('spread', 0)
        
        # Total movement
        total_movement = last_line.get('total', 0) - first_line.get('total', 0)
        
        return {
            'spread_movement': spread_movement,
            'total_movement': total_movement,
            'public_on_home': spread_movement < 0,  # Line moved away from home = public on home
            'public_on_over': total_movement < 0,   # Line moved down = public on under
        }


class SentimentEnhancedAICouncil:
    """AI Council that integrates sentiment analysis into predictions"""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.sentiment_extractor = SentimentFeatureExtractor()
        self.feature_columns = []
        self.sentiment_features = None
        
    def load_models(self):
        """Load trained models"""
        
        model_files = [
            'spread_expert_nfl_model.pkl',
            'contrarian_nfl_model.pkl',
            'home_advantage_nfl_model.pkl',
            'total_regressor_nfl_model.pkl',
            'total_high_games_nfl_model.pkl',
            'total_low_games_nfl_model.pkl',
            'total_weather_adjusted_nfl_model.pkl'
        ]
        
        for model_file in model_files:
            path = self.models_dir / model_file
            if path.exists():
                self.models[model_file.replace('_nfl_model.pkl', '')] = joblib.load(str(path))
        
        # Load feature columns
        features_path = self.models_dir / 'nfl_features.json'
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_data = json.load(f)
                self.feature_columns = feature_data.get('spread_features', [])
        
        print(f"✅ Loaded {len(self.models)} models")
    
    def engineer_sentiment_features(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Create sentiment features for all games"""
        
        features = {col: [] for col in [
            'reddit_sentiment_score', 'reddit_post_count',
            'expert_picks_home_pct', 'crowd_roar_confidence',
            'sharp_public_split_ml', 'sharp_public_split_spread', 'sharp_public_split_total',
            'public_side_winning', 'contrarian_opportunity',
            'sentiment_agreement', 'consensus_strength'
        ]}
        
        for _, game in df.iterrows():
            game_sentiment = self.sentiment_extractor.extract_game_sentiment({
                'home_team': game.get('home_team'),
                'away_team': game.get('away_team'),
                'game_id': game.get('game_id')
            })
            
            for feat_name, feat_value in game_sentiment.items():
                if feat_name in features:
                    features[feat_name].append(feat_value)
        
        return features
    
    def make_prediction_with_sentiment(self, game_data: Dict) -> Dict[str, Any]:
        """
        Make prediction incorporating sentiment analysis
        
        Output includes:
        - Model predictions (spread, total, moneyline)
        - Sentiment analysis (public, sharp, contrarian signals)
        - Confidence scores
        - Edge scoring
        """
        
        # Extract sentiment
        sentiment = self.sentiment_extractor.extract_game_sentiment(game_data)
        
        prediction = {
            'game_id': game_data.get('game_id'),
            'home_team': game_data.get('home_team'),
            'away_team': game_data.get('away_team'),
            'timestamp': datetime.now().isoformat(),
            
            # Model Predictions
            'predictions': {
                'spread': {},
                'moneyline': {},
                'total': {},
            },
            
            # Sentiment Analysis
            'sentiment': {
                'reddit_sentiment': sentiment.get('reddit_sentiment_score', 0.0),
                'expert_consensus': sentiment.get('expert_picks_home_pct', 0.5),
                'crowd_roar_signal': sentiment.get('crowd_roar_confidence', 0.0),
                'sharp_public_divergence': abs(sentiment.get('sharp_public_split_ml', 0.0)),
            },
            
            # Edge Signals
            'edges': {
                'contrarian_opportunity': sentiment.get('contrarian_opportunity', 0.0),
                'consensus_strength': sentiment.get('consensus_strength', 0.0),
                'public_money_advantage': sentiment.get('public_money_pct_home', 0.5),
            },
            
            # Combined Signal
            'recommendation': self._generate_recommendation(sentiment),
            'confidence': 0.0
        }
        
        return prediction
    
    def _generate_recommendation(self, sentiment: Dict) -> Dict[str, Any]:
        """Generate betting recommendation based on sentiment"""
        
        rec = {
            'type': 'neutral',
            'strength': 0.0,
            'reason': '',
            'size': 'small'
        }
        
        # Check for strong contrarian signals
        if sentiment.get('contrarian_opportunity', 0.0) > 0.7:
            rec['type'] = 'contrarian'
            rec['strength'] = sentiment['contrarian_opportunity']
            rec['reason'] = 'Sharp money heavily diverging from public'
            rec['size'] = 'large' if sentiment['contrarian_opportunity'] > 0.85 else 'medium'
        
        # Check for crowd roar (league let them play)
        elif sentiment.get('crowd_roar_confidence', 0.0) > 0.6:
            rec['type'] = 'crowd_roar_edge'
            rec['strength'] = sentiment['crowd_roar_confidence']
            rec['reason'] = 'Crowd roar without penalty - 5% next drive edge'
            rec['size'] = 'medium'
        
        # Check for strong consensus
        elif sentiment.get('sentiment_agreement', 0.5) > 0.75:
            if sentiment.get('consensus_strength', 0.0) > 0.7:
                rec['type'] = 'consensus'
                rec['strength'] = sentiment['consensus_strength']
                rec['reason'] = 'Strong consensus among Reddit, experts, and sharp money'
                rec['size'] = 'medium'
        
        return rec
    
    def train_sentiment_aware_models(self, df: pd.DataFrame, sentiment_features: Dict):
        """Retrain models with sentiment features included"""
        
        print("\n🎯 TRAINING SENTIMENT-AWARE AI COUNCIL")
        print("=" * 60)
        
        # Add sentiment features to dataframe
        for feat_name, feat_values in sentiment_features.items():
            if len(feat_values) == len(df):
                df[feat_name] = feat_values
        
        # Extend feature columns to include sentiment
        all_features = self.feature_columns + list(sentiment_features.keys())
        
        X = df[all_features].values
        
        # Train/retrain models with sentiment features
        models_to_train = {
            'spread_expert': ('spread_result', 'Spread Expert'),
            'contrarian': ('home_winner', 'Contrarian'),
            'home_advantage': ('home_winner', 'Home Advantage'),
        }
        
        for model_name, (target_col, display_name) in models_to_train.items():
            if target_col not in df.columns:
                continue
            
            y = df[target_col].values
            X_train, X_test, y_train, y_test = self._train_test_split(X, y)
            
            if model_name in self.models:
                model = self.models[model_name]
                model.fit(X_train, y_train)
                
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(y_test, model.predict(X_test))
                print(f"📈 {display_name} (with sentiment): {acc:.2%}")
    
    def _train_test_split(self, X, y, test_size=0.2):
        """Split data for training"""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def export_predictions(self, predictions: List[Dict], output_path: str = 'predictions_with_sentiment.json'):
        """Export predictions with sentiment analysis"""
        
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'summary': {
                    'total_games': len(predictions),
                    'contrarian_opportunities': sum(1 for p in predictions if p['recommendation']['type'] == 'contrarian'),
                    'strong_consensus': sum(1 for p in predictions if p['recommendation']['type'] == 'consensus'),
                    'crowd_roar_edges': sum(1 for p in predictions if p['recommendation']['type'] == 'crowd_roar_edge'),
                }
            }, f, indent=2)
        
        print(f"💾 Predictions exported to {output_path}")


def main():
    """Main execution"""
    
    print("🏈 SENTIMENT-ENHANCED AI COUNCIL")
    print("=" * 60)
    
    # Initialize
    council = SentimentEnhancedAICouncil()
    council.load_models()
    
    # Example: Make prediction with sentiment
    test_game = {
        'game_id': 'KC_vs_NE_2025_01_15',
        'home_team': 'Chiefs',
        'away_team': 'Patriots',
        'total_line': 44.5,
        'spread': -7.0,
        'referee': 'Bill Vinovich'
    }
    
    print(f"\n🎮 Making prediction for {test_game['home_team']} vs {test_game['away_team']}")
    prediction = council.make_prediction_with_sentiment(test_game)
    
    print(f"\n📊 Prediction:")
    print(f"   Recommendation: {prediction['recommendation']['type'].upper()}")
    print(f"   Strength: {prediction['recommendation']['strength']:.2%}")
    print(f"   Size: {prediction['recommendation']['size']}")
    print(f"   Reason: {prediction['recommendation']['reason']}")
    
    print(f"\n💬 Sentiment Analysis:")
    print(f"   Reddit Sentiment: {prediction['sentiment']['reddit_sentiment']:.2f}")
    print(f"   Expert Consensus: {prediction['sentiment']['expert_consensus']:.1%} home")
    print(f"   Sharp/Public Divergence: {prediction['sentiment']['sharp_public_divergence']:.2%}")
    print(f"   Crowd Roar Signal: {prediction['sentiment']['crowd_roar_signal']:.2%}")
    
    print(f"\n🎯 Edge Opportunities:")
    print(f"   Contrarian Score: {prediction['edges']['contrarian_opportunity']:.2%}")
    print(f"   Consensus Strength: {prediction['edges']['consensus_strength']:.2%}")

if __name__ == "__main__":
    main()
