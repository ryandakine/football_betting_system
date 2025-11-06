"""
NFL Analysis Lambda Handler with Self-Learning
Integrates adaptive learning, anomaly detection, and dynamic strategy optimization
"""
import json
import os
import boto3
import asyncio
from datetime import datetime
import pickle
import numpy as np
import logging

# Import self-learning modules
from adaptive_learning_engine import AdaptiveLearningEngine, PredictionFeedbackCollector
from anomaly_detector import AnomalyDetector
from dynamic_strategy_optimizer import DynamicStrategyOptimizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
BUCKET = os.environ.get('S3_BUCKET', 'football-betting-system-data')

# Initialize self-learning components (these persist across Lambda invocations with proper setup)
learning_engine = None
anomaly_detector = None
strategy_optimizer = None
feedback_collector = None


def initialize_learning_system():
    """Initialize all self-learning components"""
    global learning_engine, anomaly_detector, strategy_optimizer, feedback_collector
    
    learning_engine = AdaptiveLearningEngine(BUCKET)
    anomaly_detector = AnomalyDetector(window_size=100)
    strategy_optimizer = DynamicStrategyOptimizer(initial_bankroll=10000.0)
    feedback_collector = PredictionFeedbackCollector(BUCKET)
    
    logger.info("✅ Self-learning system initialized")


def lambda_handler(event, context):
    """
    Enhanced Lambda handler with self-learning capabilities
    """
    try:
        # Initialize if needed
        if learning_engine is None:
            initialize_learning_system()
        
        logger.info("🚀 Starting NFL Analysis with Self-Learning...")
        
        # 1. Fetch game data
        from nfl_live_data_fetcher import NFLLiveDataFetcher
        
        async def fetch_games():
            async with NFLLiveDataFetcher() as fetcher:
                return await fetcher.get_live_games()
        
        games = asyncio.run(fetch_games())
        
        if not games:
            return create_response(200, {
                'message': 'No games found',
                'games_analyzed': 0
            })
        
        # 2. Load models
        models = load_models_from_s3()
        
        # 3. Process predictions with self-learning
        predictions = []
        anomalies_detected = []
        
        for game in games:
            try:
                pred = analyze_game_with_learning(game, models)
                predictions.append(pred)
                
                # Check for anomalies
                if pred.get('anomaly_analysis'):
                    anomalies_detected.append(pred['anomaly_analysis'])
                    
            except Exception as e:
                logger.error(f"Error analyzing {game['id']}: {e}")
        
        # 4. Apply dynamic strategy filtering
        approved_bets = []
        for pred in predictions:
            evaluation = strategy_optimizer.evaluate_bet_opportunity({
                'market': pred.get('market', 'moneyline'),
                'confidence': pred.get('confidence', 0.5),
                'edge': pred.get('edge', 0)
            })
            
            if evaluation['should_bet']:
                pred['bet_sizing'] = evaluation['bet_sizing']
                approved_bets.append(pred)
        
        # 5. Save results with learning feedback
        timestamp = datetime.utcnow().isoformat()
        key = f'predictions/{datetime.utcnow().strftime("%Y-%m-%d")}/{timestamp}.json'
        
        results = {
            'timestamp': timestamp,
            'games_analyzed': len(games),
            'total_predictions': len(predictions),
            'approved_bets': len(approved_bets),
            'anomalies_detected': len(anomalies_detected),
            'predictions': predictions,
            'approved_picks': approved_bets,
            'learning_insights': {
                'adaptive_weights': learning_engine.calculate_adaptive_weights(),
                'dynamic_kelly_fraction': learning_engine.get_dynamic_kelly_fraction(),
                'concept_drift': learning_engine.detect_concept_drift(),
                'strategy_metrics': strategy_optimizer.get_performance_metrics()
            },
            'anomalies': anomalies_detected[:5]  # Top 5 anomalies
        }
        
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=json.dumps(results),
            ContentType='application/json'
        )
        
        # 6. Save learning state
        learning_engine.save_learning_state()
        strategy_optimizer.save_strategy_state()
        
        logger.info(f"✅ Analysis complete: {len(approved_bets)} approved bets from {len(games)} games")
        
        return create_response(200, {
            'message': 'Analysis complete with self-learning',
            'games_analyzed': len(games),
            'predictions': len(predictions),
            'approved_bets': len(approved_bets),
            'anomalies_detected': len(anomalies_detected),
            's3_key': key,
            'picks': approved_bets[:10]  # Return top 10
        })
        
    except Exception as e:
        logger.error(f"Lambda error: {str(e)}", exc_info=True)
        return create_response(500, {
            'error': str(e),
            'type': type(e).__name__
        })


def analyze_game_with_learning(game, models):
    """
    Analyze game and integrate self-learning feedback
    """
    game_id = game['id']
    home_team = game['home_team']
    away_team = game['away_team']
    
    # Extract features
    features = extract_features(game)
    
    # Get predictions from models
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
    
    # Calculate adaptive confidence using learning engine
    adaptive_weights = learning_engine.calculate_adaptive_weights()
    
    confidences = [
        abs(predictions.get('spread', {}).get('home_covers', 0.5) - 0.5) * 2,
        abs(predictions.get('total', {}).get('over_prob', 0.5) - 0.5) * 2,
        abs(predictions.get('moneyline', {}).get('home_win_prob', 0.5) - 0.5) * 2
    ]
    
    # Weight confidence by adaptive model performance
    weighted_confidence = np.average(
        confidences,
        weights=list(adaptive_weights.values()) if len(adaptive_weights) == 3 else None
    )
    
    # Detect anomalies
    anomaly_scan = anomaly_detector.comprehensive_anomaly_scan(
        prediction={
            'model': 'ensemble',
            'confidence': weighted_confidence,
            'game_id': game_id
        },
        outcome={
            'is_correct': game.get('status') != 'scheduled'
        },
        context={
            'recent_accuracy': learning_engine.model_performance.get('spread_ensemble', {}).get('recent_accuracy', 0.5),
            'historical_accuracy': learning_engine.model_performance.get('spread_ensemble', {}).get('accuracy', 0.5)
        }
    )
    
    return {
        'game_id': game_id,
        'home_team': home_team,
        'away_team': away_team,
        'predictions': predictions,
        'confidence': float(weighted_confidence),
        'adaptive_weights': adaptive_weights,
        'anomaly_analysis': anomaly_scan if anomaly_scan['anomalies_detected'] > 0 else None,
        'weather': game.get('weather', 'Unknown'),
        'stadium': game.get('stadium', 'Unknown'),
        'market': 'moneyline',
        'edge': 0.06  # Placeholder - calculate from odds
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
            logger.info(f"✅ Loaded {model_file}")
        except Exception as e:
            logger.error(f"Error loading {model_file}: {e}")
    
    return models


def extract_features(game):
    """Extract features from game data"""
    features = [
        game.get('home_score', 0),
        game.get('away_score', 0),
        game.get('quarter', 1),
        0,  # spread_line
        0,  # total_line
        0,  # home_ml_odds
        0,  # away_ml_odds
        1.0,  # home_strength
        1.0,  # away_strength
        0,  # rest_diff
        0,  # injury_impact_home
        0,  # injury_impact_away
        0,  # weather_impact
        1 if game.get('game_time', '').startswith(('19', '20')) else 0,  # is_primetime
        0,
        0,
        0
    ]
    return features[:17]


def create_response(status_code, body):
    """Create Lambda response"""
    return {
        'statusCode': status_code,
        'body': json.dumps(body)
    }


if __name__ == "__main__":
    # Test locally
    initialize_learning_system()
    print("Self-learning system ready!")
