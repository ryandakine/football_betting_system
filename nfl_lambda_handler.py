"""
NFL Analysis Lambda Handler
Runs full prediction system: ensemble models + HRM + weather + referee data
"""
import json
import os
import boto3
import requests
from datetime import datetime
import pickle
import numpy as np

s3 = boto3.client('s3')
BUCKET = os.environ.get('S3_BUCKET', 'football-betting-system-data')

def fetch_odds():
    """Fetch current NFL odds from The Odds API"""
    api_key = os.environ.get('ODDS_API_KEY', '')
    if not api_key:
        print("No odds API key configured")
        return None
    
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'h2h,spreads,totals',
        'oddsFormat': 'american'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"Fetched odds for {len(data)} games")
        return data
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return None

def lambda_handler(event, context):
    """Main handler for NFL game analysis"""
    try:
        # 1. Get today's games from ESPN
        from nfl_live_data_fetcher import NFLLiveDataFetcher
        
        games = []
        async def fetch_games():
            async with NFLLiveDataFetcher() as fetcher:
                return await fetcher.get_live_games()
        
        import asyncio
        games = asyncio.run(fetch_games())
        
        if not games:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'No games found',
                    'games_analyzed': 0
                })
            }
        
        # 2. Fetch odds data
        odds_data = fetch_odds()
        
        # 3. Load trained models from S3
        models = load_models_from_s3()
        
        # 4. Run predictions on each game
        predictions = []
        for game in games:
            try:
                pred = analyze_game(game, models, odds_data)
                predictions.append(pred)
            except Exception as e:
                print(f"Error analyzing {game['id']}: {e}")
        
        # 4. Filter high confidence picks
        high_conf_picks = [p for p in predictions if p['confidence'] > 0.75]
        
        # 5. Store results in S3
        timestamp = datetime.utcnow().isoformat()
        key = f'predictions/{datetime.utcnow().strftime("%Y-%m-%d")}/{timestamp}.json'
        
        results = {
            'timestamp': timestamp,
            'games_analyzed': len(games),
            'total_predictions': len(predictions),
            'high_confidence_picks': high_conf_picks,
            'all_predictions': predictions
        }
        
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=json.dumps(results),
            ContentType='application/json'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Analysis complete',
                'games_analyzed': len(games),
                'predictions': len(predictions),
                'high_confidence_picks': len(high_conf_picks),
                's3_key': key,
                'picks': high_conf_picks
            })
        }
        
    except Exception as e:
        print(f"Lambda error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'type': type(e).__name__
            })
        }

def load_models_from_s3():
    """Load trained ensemble models from S3"""
    models = {}
    model_files = ['spread_ensemble.pkl', 'total_ensemble.pkl', 'moneyline_ensemble.pkl']
    
    for model_file in model_files:
        try:
            response = s3.get_object(Bucket=BUCKET, Key=f'models/{model_file}')
            model_data = response['Body'].read()
            models[model_file.replace('.pkl', '')] = pickle.loads(model_data)
            print(f"Loaded {model_file}")
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
    
    return models

def analyze_game(game, models, odds_data):
    """Run full analysis on a single game"""
    game_id = game['id']
    home_team = game['home_team']
    away_team = game['away_team']
    
    # Extract features for ensemble models
    features = extract_features(game, odds_data)
    
    # Get predictions from each model
    predictions = {}
    
    if 'spread_ensemble' in models:
        spread_pred = models['spread_ensemble'].predict_proba([features])[0]
        predictions['spread'] = {
            'home_covers': float(spread_pred[1]),
            'pick': home_team if spread_pred[1] > 0.5 else away_team
        }
    
    if 'total_ensemble' in models:
        total_pred = models['total_ensemble'].predict_proba([features])[0]
        predictions['total'] = {
            'over_prob': float(total_pred[1]),
            'pick': 'OVER' if total_pred[1] > 0.5 else 'UNDER'
        }
    
    if 'moneyline_ensemble' in models:
        ml_pred = models['moneyline_ensemble'].predict_proba([features])[0]
        predictions['moneyline'] = {
            'home_win_prob': float(ml_pred[1]),
            'pick': home_team if ml_pred[1] > 0.5 else away_team
        }
    
    # Calculate overall confidence
    confidences = [
        abs(predictions.get('spread', {}).get('home_covers', 0.5) - 0.5) * 2,
        abs(predictions.get('total', {}).get('over_prob', 0.5) - 0.5) * 2,
        abs(predictions.get('moneyline', {}).get('home_win_prob', 0.5) - 0.5) * 2
    ]
    avg_confidence = np.mean(confidences)
    
    return {
        'game_id': game_id,
        'home_team': home_team,
        'away_team': away_team,
        'predictions': predictions,
        'confidence': float(avg_confidence),
        'weather': game.get('weather', 'Unknown'),
        'stadium': game.get('stadium', 'Unknown')
    }

def extract_features(game, odds_data):
    """Extract features from game data for model input"""
    home_team = game['home_team']
    away_team = game['away_team']
    
    # Find odds for this game
    spread_line = 0.0
    total_line = 0.0
    home_ml_odds = 0.0
    away_ml_odds = 0.0
    
    if odds_data:
        for game_odds in odds_data:
            h_team = game_odds.get('home_team', '')
            a_team = game_odds.get('away_team', '')
            
            if h_team == home_team or a_team == away_team:
                bookmakers = game_odds.get('bookmakers', [])
                if bookmakers:
                    book = bookmakers[0]  # Use first bookmaker
                    markets = book.get('markets', [])
                    
                    for market in markets:
                        if market['key'] == 'spreads':
                            for outcome in market.get('outcomes', []):
                                if outcome['name'] == home_team:
                                    spread_line = outcome.get('point', 0.0)
                        elif market['key'] == 'totals':
                            outcomes = market.get('outcomes', [])
                            if outcomes:
                                total_line = outcomes[0].get('point', 0.0)
                        elif market['key'] == 'h2h':
                            for outcome in market.get('outcomes', []):
                                if outcome['name'] == home_team:
                                    home_ml_odds = outcome.get('price', 0.0)
                                elif outcome['name'] == away_team:
                                    away_ml_odds = outcome.get('price', 0.0)
                break
    
    # Convert American odds to decimal
    def american_to_decimal(american_odds):
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    home_ml_decimal = american_to_decimal(home_ml_odds) if home_ml_odds != 0 else 2.0
    away_ml_decimal = american_to_decimal(away_ml_odds) if away_ml_odds != 0 else 2.0
    
    # Extract 17 features matching training data
    features = [
        spread_line,
        total_line,
        home_ml_decimal,
        away_ml_decimal,
        1.0,  # is_home
        game.get('home_rank', 16),
        game.get('away_rank', 16),
        0.0,  # recent_form_home
        0.0,  # recent_form_away
        0.0,  # head_to_head_wins
        0.0,  # injuries_home
        0.0,  # injuries_away
        1 if isinstance(game.get('weather'), dict) and game.get('weather', {}).get('condition') == 'Clear' else 0,
        1 if isinstance(game.get('weather'), dict) and 'Rain' in str(game.get('weather', {}).get('condition', '')) else 0,
        1 if isinstance(game.get('weather'), dict) and 'Snow' in str(game.get('weather', {}).get('condition', '')) else 0,
        game.get('weather', {}).get('temperature', 70) if isinstance(game.get('weather'), dict) else 70,
        game.get('weather', {}).get('wind_speed', 0) if isinstance(game.get('weather'), dict) else 0,
    ]
    
    return features
