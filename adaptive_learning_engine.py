"""
Adaptive Learning Engine - Self-Learning AI System
Continuously improves model performance through feedback and reinforcement
"""
import json
import numpy as np
import boto3
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdaptiveLearningEngine:
    """
    Self-learning system that:
    1. Tracks prediction accuracy per model and market
    2. Adjusts model weights based on performance
    3. Detects and learns from anomalies
    4. Optimizes Kelly fractions dynamically
    5. Performs incremental retraining
    """
    
    def __init__(self, s3_bucket: str = 'football-betting-system-data'):
        self.s3_bucket = s3_bucket
        self.s3 = boto3.client('s3')
        
        # Performance tracking
        self.model_performance = defaultdict(lambda: {
            'correct': 0,
            'total': 0,
            'accuracy': 0.5,
            'recent_accuracy': 0.5,
            'confidence_scores': [],
            'calibration_error': 0
        })
        
        # Market-specific learning
        self.market_insights = defaultdict(lambda: {
            'line_movement_patterns': [],
            'weather_impact': {},
            'time_of_day_bias': {},
            'team_matchup_history': {}
        })
        
        # Learning rates and decay
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.weight_history = []
        
        self.load_performance_history()
    
    def load_performance_history(self):
        """Load historical performance data from S3"""
        try:
            response = self.s3.get_object(
                Bucket=self.s3_bucket,
                Key='learning/model_performance_history.json'
            )
            history = json.loads(response['Body'].read())
            
            for model, stats in history.items():
                self.model_performance[model] = stats
                
            logger.info(f"✅ Loaded performance history for {len(history)} models")
        except:
            logger.info("📝 Starting fresh performance tracking")
    
    def record_prediction(self, prediction: Dict[str, Any], actual_outcome: Dict[str, Any]):
        """
        Record a prediction and its actual outcome for learning
        
        Args:
            prediction: {'model': 'spread_ensemble', 'confidence': 0.8, 'pick': 'home', 'odds': -110}
            actual_outcome: {'result': 'correct', 'margin': 2.5, 'steam_move': True}
        """
        model_name = prediction['model']
        confidence = prediction.get('confidence', 0.5)
        is_correct = actual_outcome['result'] == 'correct'
        
        # Update basic stats
        perf = self.model_performance[model_name]
        perf['total'] += 1
        if is_correct:
            perf['correct'] += 1
        
        perf['accuracy'] = perf['correct'] / max(1, perf['total'])
        perf['confidence_scores'].append(confidence)
        
        # Calibration: compare confidence with actual accuracy
        recent_window = perf['confidence_scores'][-100:]  # Last 100 predictions
        if len(recent_window) > 10:
            avg_confidence = np.mean(recent_window)
            perf['recent_accuracy'] = np.mean([1 if p else 0 for p in recent_window[-20:]])
            perf['calibration_error'] = abs(avg_confidence - perf['recent_accuracy'])
        
        # Market insights
        market = prediction.get('market', 'moneyline')
        self._update_market_insights(market, prediction, actual_outcome)
        
        logger.info(f"📊 {model_name}: {'✅' if is_correct else '❌'} "
                   f"(Accuracy: {perf['accuracy']:.1%}, Calibration: {perf['calibration_error']:.2f})")
    
    def _update_market_insights(self, market: str, prediction: Dict, outcome: Dict):
        """Learn from market-specific patterns"""
        insights = self.market_insights[market]
        
        # Track line movement impact
        if 'line_movement' in outcome:
            insights['line_movement_patterns'].append({
                'movement': outcome['line_movement'],
                'result': outcome['result'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Weather impact learning
        if 'weather_impact' in outcome:
            weather = outcome['weather_impact']
            if weather not in insights['weather_impact']:
                insights['weather_impact'][weather] = {'wins': 0, 'total': 0}
            
            insights['weather_impact'][weather]['total'] += 1
            if outcome['result'] == 'correct':
                insights['weather_impact'][weather]['wins'] += 1
    
    def calculate_adaptive_weights(self) -> Dict[str, float]:
        """
        Calculate optimal model weights based on performance history
        Uses exponential moving average for recent performance emphasis
        """
        weights = {}
        total_accuracy = 0
        
        # Calculate accuracy-based weights
        for model, perf in self.model_performance.items():
            # Emphasize recent accuracy (70%) vs overall (30%)
            accuracy = 0.7 * perf['recent_accuracy'] + 0.3 * perf['accuracy']
            
            # Penalize poorly calibrated models
            calibration_bonus = 1 - perf['calibration_error']
            
            # Apply confidence adjustment
            adjusted_accuracy = accuracy * max(0.5, calibration_bonus)
            weights[model] = max(0.1, adjusted_accuracy)  # Minimum 10% weight
            total_accuracy += weights[model]
        
        # Normalize weights to sum to 1.0
        if total_accuracy > 0:
            weights = {m: w / total_accuracy for m, w in weights.items()}
        
        logger.info(f"🎯 Adaptive Weights: {', '.join(f'{m}:{w:.1%}' for m, w in weights.items())}")
        return weights
    
    def get_dynamic_kelly_fraction(self) -> float:
        """
        Adjust Kelly fraction based on system performance
        More conservative when accuracy is uncertain
        """
        if not self.model_performance:
            return 0.25  # Default quarter Kelly
        
        # Calculate ensemble accuracy
        total_correct = sum(p['correct'] for p in self.model_performance.values())
        total_bets = sum(p['total'] for p in self.model_performance.values())
        
        if total_bets < 100:
            return 0.15  # Very conservative with limited data
        
        ensemble_accuracy = total_correct / total_bets if total_bets > 0 else 0.5
        
        # Dynamic Kelly calculation
        if ensemble_accuracy > 0.60:
            kelly_frac = 0.35  # More aggressive with proven accuracy
        elif ensemble_accuracy > 0.55:
            kelly_frac = 0.25  # Standard quarter Kelly
        else:
            kelly_frac = 0.15  # Conservative with marginal edge
        
        logger.info(f"💰 Dynamic Kelly Fraction: {kelly_frac:.1%} "
                   f"(Ensemble Accuracy: {ensemble_accuracy:.1%})")
        return kelly_frac
    
    def detect_concept_drift(self) -> Dict[str, Any]:
        """
        Detect when market conditions change (concept drift)
        Triggers retraining when drift is detected
        """
        if not self.model_performance:
            return {'drift_detected': False}
        
        drift_signals = []
        
        for model, perf in self.model_performance.items():
            if len(perf['confidence_scores']) < 50:
                continue
            
            # Compare recent vs overall accuracy
            recent_accuracy = perf['recent_accuracy']
            overall_accuracy = perf['accuracy']
            
            drift_magnitude = abs(recent_accuracy - overall_accuracy)
            
            if drift_magnitude > 0.1:  # >10% difference
                drift_signals.append({
                    'model': model,
                    'drift_magnitude': drift_magnitude,
                    'recent_accuracy': recent_accuracy,
                    'overall_accuracy': overall_accuracy
                })
        
        return {
            'drift_detected': len(drift_signals) > 0,
            'signals': drift_signals,
            'recommendation': 'RETRAIN' if len(drift_signals) > 2 else 'MONITOR'
        }
    
    def should_retrain(self) -> bool:
        """
        Determine if model retraining should be triggered
        Based on: concept drift, performance degradation, new data volume
        """
        drift_analysis = self.detect_concept_drift()
        
        if drift_analysis['drift_detected'] and len(drift_analysis['signals']) > 2:
            logger.warning(f"🔄 CONCEPT DRIFT DETECTED - Retraining recommended")
            return True
        
        # Check if we have enough new data
        total_predictions = sum(p['total'] for p in self.model_performance.values())
        if total_predictions > 0 and total_predictions % 100 == 0:
            logger.info(f"📊 {total_predictions} predictions recorded - consider retraining")
            return total_predictions % 200 == 0
        
        return False
    
    def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning analysis report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_statistics': {},
            'market_insights': {},
            'concept_drift': self.detect_concept_drift(),
            'recommendations': []
        }
        
        # Model statistics
        for model, perf in self.model_performance.items():
            report['model_statistics'][model] = {
                'accuracy': f"{perf['accuracy']:.1%}",
                'recent_accuracy': f"{perf['recent_accuracy']:.1%}",
                'calibration_error': f"{perf['calibration_error']:.3f}",
                'predictions': perf['total']
            }
        
        # Market insights summary
        for market, insights in self.market_insights.items():
            weather_summary = {}
            for weather, stats in insights['weather_impact'].items():
                win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
                weather_summary[weather] = f"{win_rate:.1%}"
            
            report['market_insights'][market] = {
                'weather_performance': weather_summary
            }
        
        # Adaptive weights
        report['current_weights'] = self.calculate_adaptive_weights()
        report['dynamic_kelly_fraction'] = self.get_dynamic_kelly_fraction()
        
        # Recommendations
        if report['concept_drift']['drift_detected']:
            report['recommendations'].append("⚠️ Concept drift detected - consider retraining models")
        
        min_accuracy = min((p['accuracy'] for p in self.model_performance.values()), default=0.5)
        if min_accuracy < 0.52:
            report['recommendations'].append("⚠️ Poor model accuracy - urgent retraining needed")
        
        logger.info(f"📈 Learning Report Generated - {len(report['recommendations'])} recommendations")
        return report
    
    def save_learning_state(self):
        """Persist learning state to S3"""
        try:
            # Save performance history
            perf_data = {
                model: {
                    'correct': perf['correct'],
                    'total': perf['total'],
                    'accuracy': perf['accuracy'],
                    'recent_accuracy': perf['recent_accuracy'],
                    'calibration_error': perf['calibration_error']
                }
                for model, perf in self.model_performance.items()
            }
            
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key='learning/model_performance_history.json',
                Body=json.dumps(perf_data),
                ContentType='application/json'
            )
            
            # Save report
            report = self.generate_learning_report()
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=f'learning/reports/{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                Body=json.dumps(report),
                ContentType='application/json'
            )
            
            logger.info("✅ Learning state saved to S3")
        except Exception as e:
            logger.error(f"❌ Failed to save learning state: {e}")


class PredictionFeedbackCollector:
    """Collects actual game outcomes and compares with predictions"""
    
    def __init__(self, s3_bucket: str = 'football-betting-system-data'):
        self.s3_bucket = s3_bucket
        self.s3 = boto3.client('s3')
    
    def match_prediction_to_outcome(self, prediction: Dict, outcome: Dict) -> Dict[str, Any]:
        """Compare prediction with actual outcome"""
        return {
            'game_id': prediction['game_id'],
            'model': prediction['model'],
            'market': prediction['market'],
            'predicted_pick': prediction['pick'],
            'actual_result': outcome['result'],
            'confidence': prediction.get('confidence', 0.5),
            'is_correct': prediction['pick'] == outcome['winner'],
            'margin_error': abs(prediction.get('predicted_margin', 0) - outcome.get('actual_margin', 0)),
            'timestamp': datetime.now().isoformat()
        }
    
    def store_feedback(self, feedback: Dict):
        """Store feedback for analysis"""
        try:
            key = f"feedback/{datetime.now().strftime('%Y-%m-%d')}/{feedback['game_id']}.json"
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=key,
                Body=json.dumps(feedback),
                ContentType='application/json'
            )
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")


if __name__ == "__main__":
    # Test the adaptive learning engine
    engine = AdaptiveLearningEngine()
    
    # Simulate some predictions and outcomes
    for i in range(50):
        prediction = {
            'model': 'spread_ensemble',
            'confidence': np.random.uniform(0.5, 0.95),
            'pick': 'home' if np.random.random() > 0.5 else 'away',
            'market': 'spread'
        }
        
        is_correct = np.random.random() < 0.58  # 58% accuracy
        outcome = {
            'result': 'correct' if is_correct else 'incorrect',
            'weather_impact': 'rain' if np.random.random() > 0.7 else 'clear'
        }
        
        engine.record_prediction(prediction, outcome)
    
    # Generate report
    report = engine.generate_learning_report()
    print(json.dumps(report, indent=2))
