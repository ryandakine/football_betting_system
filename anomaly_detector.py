"""
Anomaly Detection Module - Identifies Market Inefficiencies
Detects unusual patterns that indicate profit opportunities or model failures
"""
import numpy as np
from typing import Dict, List, Any
from collections import deque
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in:
    - Prediction confidence vs actual accuracy
    - Line movements that contradict public opinion
    - Sudden model performance degradation
    - Statistical outliers in odds/outcomes
    - Team performance reversals
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Rolling statistics for anomaly detection
        self.prediction_confidence_history = deque(maxlen=window_size)
        self.actual_accuracy_history = deque(maxlen=window_size)
        self.line_movement_history = deque(maxlen=window_size)
        self.prediction_variance = deque(maxlen=window_size)
        
        # Thresholds
        self.confidence_accuracy_divergence_threshold = 0.15
        self.statistical_outlier_z_threshold = 3.0
        self.performance_degradation_threshold = 0.12
    
    def detect_calibration_anomaly(self, prediction: Dict, outcome: Dict) -> Dict[str, Any]:
        """
        Detect when model confidence doesn't match actual outcomes
        High confidence predictions that lose consistently = miscalibration
        """
        confidence = prediction.get('confidence', 0.5)
        is_correct = outcome.get('is_correct', False)
        
        self.prediction_confidence_history.append(confidence)
        self.actual_accuracy_history.append(1 if is_correct else 0)
        
        if len(self.prediction_confidence_history) < 20:
            return {'detected': False}
        
        avg_confidence = np.mean(self.prediction_confidence_history)
        avg_accuracy = np.mean(self.actual_accuracy_history)
        divergence = abs(avg_confidence - avg_accuracy)
        
        detected = divergence > self.confidence_accuracy_divergence_threshold
        
        if detected:
            logger.warning(f"⚠️ CALIBRATION ANOMALY: "
                          f"Avg Confidence {avg_confidence:.1%} vs Accuracy {avg_accuracy:.1%} "
                          f"(Divergence: {divergence:.1%})")
        
        return {
            'detected': detected,
            'type': 'calibration_anomaly',
            'avg_confidence': avg_confidence,
            'avg_accuracy': avg_accuracy,
            'divergence': divergence,
            'severity': 'critical' if divergence > 0.25 else 'warning' if divergence > 0.15 else 'monitor',
            'recommendation': 'Adjust prediction confidence' if divergence > 0.15 else None
        }
    
    def detect_line_movement_anomaly(self, line_data: Dict) -> Dict[str, Any]:
        """
        Detect unusual line movements that might indicate:
        - Sharp money coming in against consensus
        - Injury news affecting odds
        - Reverse line movement (public bets one way, line goes other)
        """
        opening_line = line_data.get('opening_line', 0)
        current_line = line_data.get('current_line', 0)
        public_bets = line_data.get('public_bet_percentage', 0.5)
        
        movement = current_line - opening_line
        
        self.line_movement_history.append(movement)
        
        # Detect reverse line movement
        is_reverse_movement = (
            (movement > 1.5 and public_bets > 0.6) or  # Line moved UP but public favors favorite
            (movement < -1.5 and public_bets < 0.4)     # Line moved DOWN but public favors underdog
        )
        
        if is_reverse_movement:
            logger.info(f"🔄 REVERSE LINE MOVEMENT: Movement={movement:.1f}, "
                       f"Public Lean={public_bets:.0%}")
        
        return {
            'detected': is_reverse_movement,
            'type': 'reverse_line_movement',
            'movement': movement,
            'public_lean': public_bets,
            'severity': 'high' if abs(movement) > 3 else 'medium',
            'interpretation': 'Sharp money against public' if is_reverse_movement else None
        }
    
    def detect_statistical_outlier(self, values: List[float], new_value: float) -> Dict[str, Any]:
        """
        Use z-score to detect statistical outliers
        Could indicate data quality issues or genuine market anomalies
        """
        if len(values) < 10:
            return {'detected': False}
        
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            return {'detected': False}
        
        z_score = abs((new_value - mean) / std)
        is_outlier = z_score > self.statistical_outlier_z_threshold
        
        if is_outlier:
            logger.warning(f"📊 STATISTICAL OUTLIER: Value={new_value:.2f}, "
                          f"Mean={mean:.2f}, Std={std:.2f}, Z-Score={z_score:.2f}")
        
        return {
            'detected': is_outlier,
            'type': 'statistical_outlier',
            'z_score': z_score,
            'mean': mean,
            'std': std,
            'severity': 'critical' if z_score > 5 else 'high'
        }
    
    def detect_performance_degradation(self, recent_accuracy: float, 
                                      historical_accuracy: float) -> Dict[str, Any]:
        """
        Detect when model performance suddenly degrades
        Indicates possible concept drift or market regime change
        """
        degradation = historical_accuracy - recent_accuracy
        
        detected = degradation > self.performance_degradation_threshold
        
        if detected:
            logger.warning(f"⚠️ PERFORMANCE DEGRADATION: "
                          f"Historical {historical_accuracy:.1%} → Recent {recent_accuracy:.1%} "
                          f"(Drop: {degradation:.1%})")
        
        return {
            'detected': detected,
            'type': 'performance_degradation',
            'degradation': degradation,
            'historical_accuracy': historical_accuracy,
            'recent_accuracy': recent_accuracy,
            'severity': 'critical' if degradation > 0.20 else 'warning',
            'recommendation': 'Reduce bet size or retrain models' if degradation > 0.12 else None
        }
    
    def detect_market_inefficiency(self, odds_data: Dict, outcome: Dict) -> Dict[str, Any]:
        """
        Detect when market is mispricing a game
        Identifies profitable opportunities
        """
        implied_probability = self._calculate_implied_probability(odds_data.get('moneyline', -110))
        actual_probability = self._calculate_actual_probability(outcome)
        
        edge = actual_probability - implied_probability
        
        detected = abs(edge) > 0.05  # >5% edge
        
        if detected:
            direction = "undervalued" if edge > 0 else "overvalued"
            logger.info(f"💎 MARKET INEFFICIENCY: {direction} "
                       f"(Implied: {implied_probability:.1%}, Actual: {actual_probability:.1%}, "
                       f"Edge: {edge:.1%})")
        
        return {
            'detected': detected,
            'type': 'market_inefficiency',
            'implied_probability': implied_probability,
            'actual_probability': actual_probability,
            'edge': edge,
            'opportunity': 'LONG' if edge > 0.05 else 'SHORT' if edge < -0.05 else None,
            'strength': abs(edge)
        }
    
    def detect_team_performance_anomaly(self, team_data: Dict) -> Dict[str, Any]:
        """
        Detect unusual team performance changes
        Could indicate injuries, coaching changes, or momentum shifts
        """
        recent_performance = team_data.get('recent_performance', 0.5)
        season_performance = team_data.get('season_performance', 0.5)
        
        shift = abs(recent_performance - season_performance)
        
        detected = shift > 0.15  # >15% change
        
        if detected:
            direction = "improvement" if recent_performance > season_performance else "decline"
            logger.info(f"📈 TEAM PERFORMANCE ANOMALY: {direction} "
                       f"(Season: {season_performance:.1%}, Recent: {recent_performance:.1%})")
        
        return {
            'detected': detected,
            'type': 'team_performance_anomaly',
            'direction': 'improvement' if recent_performance > season_performance else 'decline',
            'shift': shift,
            'potential_causes': ['Injury', 'Coaching change', 'Momentum', 'Schedule strength']
        }
    
    def detect_consensus_divergence(self, predictions: Dict[str, float], 
                                   public_sentiment: float) -> Dict[str, Any]:
        """
        Detect when expert models disagree with public opinion
        Could indicate either public misjudgment or expert blindspot
        """
        model_consensus = np.mean(list(predictions.values()))
        divergence = abs(model_consensus - public_sentiment)
        
        detected = divergence > 0.20  # >20% divergence
        
        if detected:
            logger.info(f"🤔 CONSENSUS DIVERGENCE: "
                       f"Models {model_consensus:.1%} vs Public {public_sentiment:.1%}")
        
        return {
            'detected': detected,
            'type': 'consensus_divergence',
            'model_consensus': model_consensus,
            'public_sentiment': public_sentiment,
            'divergence': divergence,
            'interpretation': 'Potential mispricing' if detected else None
        }
    
    def comprehensive_anomaly_scan(self, prediction: Dict, outcome: Dict, 
                                   context: Dict) -> Dict[str, Any]:
        """
        Run all anomaly detection modules and consolidate results
        """
        anomalies = []
        
        # Check calibration
        cal_anomaly = self.detect_calibration_anomaly(prediction, outcome)
        if cal_anomaly['detected']:
            anomalies.append(cal_anomaly)
        
        # Check line movement
        if 'line_data' in context:
            line_anomaly = self.detect_line_movement_anomaly(context['line_data'])
            if line_anomaly['detected']:
                anomalies.append(line_anomaly)
        
        # Check performance degradation
        if 'recent_accuracy' in context and 'historical_accuracy' in context:
            perf_anomaly = self.detect_performance_degradation(
                context['recent_accuracy'],
                context['historical_accuracy']
            )
            if perf_anomaly['detected']:
                anomalies.append(perf_anomaly)
        
        # Check market efficiency
        if 'odds_data' in context:
            inefficiency = self.detect_market_inefficiency(
                context['odds_data'],
                outcome
            )
            if inefficiency['detected']:
                anomalies.append(inefficiency)
        
        # Calculate overall risk level
        risk_scores = {'critical': 3, 'high': 2, 'warning': 1, 'monitor': 0}
        total_risk = sum(risk_scores.get(a.get('severity', 'monitor'), 0) for a in anomalies)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'anomalies_detected': len(anomalies),
            'anomaly_list': anomalies,
            'overall_risk_level': 'CRITICAL' if total_risk >= 6 else 
                                 'HIGH' if total_risk >= 3 else
                                 'MEDIUM' if total_risk > 0 else
                                 'LOW',
            'actions_recommended': self._generate_recommendations(anomalies)
        }
    
    def _calculate_implied_probability(self, odds: float) -> float:
        """Convert moneyline odds to implied probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def _calculate_actual_probability(self, outcome: Dict) -> float:
        """Calculate actual probability from outcome data"""
        # Simplified - would use historical data in production
        if outcome.get('is_correct'):
            return 0.5 + np.random.uniform(0.0, 0.5)
        else:
            return np.random.uniform(0.0, 0.5)
    
    def _generate_recommendations(self, anomalies: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on anomalies detected"""
        recommendations = []
        
        critical_anomalies = [a for a in anomalies if a.get('severity') == 'critical']
        if critical_anomalies:
            recommendations.append("🚨 CRITICAL: Reduce bet sizing or halt betting until reviewed")
        
        for anomaly in anomalies:
            if 'recommendation' in anomaly and anomaly['recommendation']:
                recommendations.append(f"➜ {anomaly['recommendation']}")
        
        return recommendations


if __name__ == "__main__":
    detector = AnomalyDetector()
    
    # Test scenarios
    print("Testing Anomaly Detection System...")
    print("=" * 60)
    
    # Scenario 1: Calibration anomaly
    for i in range(30):
        pred = {'confidence': 0.85}
        outcome = {'is_correct': np.random.random() < 0.45}  # Only 45% accuracy with 85% confidence
        anomaly = detector.detect_calibration_anomaly(pred, outcome)
    
    print("\n1. Calibration Anomaly Test:")
    print(json.dumps(anomaly, indent=2))
    
    # Scenario 2: Market inefficiency
    print("\n2. Market Inefficiency Test:")
    odds_data = {'moneyline': 110}
    outcome_data = {'is_correct': True}
    inefficiency = detector.detect_market_inefficiency(odds_data, outcome_data)
    print(json.dumps(inefficiency, indent=2))

import json
