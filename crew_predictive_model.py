#!/usr/bin/env python3
import json
import statistics
from pathlib import Path
from typing import Union

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "referee_conspiracy"
GAME_RECORDS_PATH = DATA_DIR / "game_records.json"
MODEL_OUTPUT_PATH = DATA_DIR / "crew_prediction_model.pkl"

def load_game_records(records_path: Union[str, Path] = GAME_RECORDS_PATH):
    """Load extracted game records."""
    path = Path(records_path)
    with path.open('r') as f:
        return json.load(f)

def build_features(games):
    """Convert game records to feature vectors."""
    X = []
    y = []
    
    crew_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    
    # Get unique values
    crews = list(set(g['crew'] for g in games))
    teams = list(set(g['team'] for g in games))
    
    crew_encoder.fit(crews)
    team_encoder.fit(teams)
    
    for game in games:
        if game['crew'] not in crews or game['team'] not in teams:
            continue
        
        # Features: crew_id, team_id, week, year
        features = [
            crew_encoder.transform([game['crew']])[0],  # Crew ID
            team_encoder.transform([game['team']])[0],  # Team ID
            game.get('week', 0) or 0,  # Week (0 if unknown)
            game.get('year', 2020),  # Year
        ]
        
        X.append(features)
        y.append(game['margin'])
    
    return np.array(X), np.array(y), crew_encoder, team_encoder

def train_model(X, y):
    """Train random forest model."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    return model

def get_feature_importance(model):
    """Get feature importance."""
    return model.feature_importances_

def predict_game_margin(model, crew_id, team_id, week, year, crew_encoder, team_encoder):
    """Predict margin for a specific game."""
    features = np.array([[crew_id, team_id, week, year]])
    prediction = model.predict(features)[0]
    return prediction

def main():
    data = load_game_records()
    games = data['games']
    
    print("\n" + "="*120)
    print("BUILDING CREW PREDICTION MODEL")
    print("="*120)
    print(f"Training on {len(games)} game records...\n")
    
    # Build features
    X, y, crew_encoder, team_encoder = build_features(games)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Crews: {len(crew_encoder.classes_)}")
    print(f"Teams: {len(team_encoder.classes_)}\n")
    
    # Train model
    model = train_model(X, y)
    
    # Feature importance
    importance = get_feature_importance(model)
    feature_names = ['Crew', 'Team', 'Week', 'Year']
    
    print("Feature Importance:")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f"  {name:10s}: {imp:.4f}")
    
    # Model score
    train_score = model.score(X, y)
    print(f"\nModel R² score: {train_score:.4f}")
    
    # Calculate residuals
    predictions = model.predict(X)
    residuals = y - predictions
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    print(f"RMSE: {rmse:.2f} points")
    print(f"MAE: {mae:.2f} points")
    
    # Show example predictions for top crews
    print("\n" + "="*120)
    print("EXAMPLE PREDICTIONS - TOP SUSPICIOUS CREWS")
    print("="*120)
    
    suspicious_crews = ['Alex Kemp', 'Carl Cheffers', 'Craig Wrolstad', 'Alan Eck', 'Walt Coleman']
    
    for crew_name in suspicious_crews:
        if crew_name not in crew_encoder.classes_:
            continue
        
        crew_id = crew_encoder.transform([crew_name])[0]
        
        print(f"\n{crew_name} - Predicted margins vs each team:")
        print("-" * 120)
        
        predictions_by_team = []
        for team_name in sorted(team_encoder.classes_):
            team_id = team_encoder.transform([team_name])[0]
            
            # Predict for mid-season (week 9)
            pred = predict_game_margin(model, crew_id, team_id, 9, 2024, crew_encoder, team_encoder)
            predictions_by_team.append((team_name, pred))
        
        # Sort by margin (blowouts first)
        predictions_by_team.sort(key=lambda x: abs(x[1]), reverse=True)
        
        favors = [p for p in predictions_by_team if p[1] > 5]
        crushes = [p for p in predictions_by_team if p[1] < -5]
        neutral = [p for p in predictions_by_team if -5 <= p[1] <= 5]
        
        if favors:
            print(f"  FAVORS: {', '.join([f'{t}({m:+.1f})' for t, m in favors[:4]])}")
        if crushes:
            print(f"  CRUSHES: {', '.join([f'{t}({m:+.1f})' for t, m in crushes[:4]])}")
        if neutral:
            print(f"  NEUTRAL: {', '.join([f'{t}({m:+.1f})' for t, m in neutral[:3]])}")
    
    # Save model
    import pickle
    model_data = {
        'model': model,
        'crew_encoder': crew_encoder,
        'team_encoder': team_encoder,
        'rmse': rmse,
        'mae': mae,
        'r2': train_score
    }
    
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_OUTPUT_PATH.open('wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to crew_prediction_model.pkl")

if __name__ == '__main__':
    main()
