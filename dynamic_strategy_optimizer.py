"""Dynamic Strategy Optimizer - Real-time betting parameter adjustment"""
import numpy as np
import boto3
import json
from datetime import datetime
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class DynamicStrategyOptimizer:
    """Adjusts betting strategy based on real-time performance"""
    
    def __init__(self, initial_bankroll: float = 10000.0, s3_bucket: str = 'football-betting-system-data'):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.s3_bucket = s3_bucket
        self.s3 = boto3.client('s3')
        
        self.confidence_threshold = 0.70
        self.kelly_fraction = 0.25
        self.min_edge = 0.05
        
        self.bets_placed = 0
        self.bets_won = 0
        self.bets_lost = 0
        self.total_stakes = 0.0
        self.total_winnings = 0.0
        
        self.market_thresholds = {
            'moneyline': {'confidence': 0.70, 'edge': 0.05},
            'spread': {'confidence': 0.72, 'edge': 0.06},
            'total': {'confidence': 0.68, 'edge': 0.04}
        }
        
        self.load_strategy_state()
    
    def load_strategy_state(self):
        """Load strategy state from S3"""
        try:
            response = self.s3.get_object(Bucket=self.s3_bucket, Key='strategy/current_state.json')
            state = json.loads(response['Body'].read())
            self.current_bankroll = state.get('bankroll', self.initial_bankroll)
            self.confidence_threshold = state.get('confidence_threshold', 0.70)
            self.kelly_fraction = state.get('kelly_fraction', 0.25)
            logger.info(f"Strategy loaded: Bankroll ${self.current_bankroll:.2f}")
        except:
            logger.info(f"Starting fresh strategy")
    
    def calculate_optimal_bet_size(self, edge: float, confidence: float) -> Dict[str, Any]:
        """Calculate Kelly Criterion bet size"""
        if edge <= 0 or edge < self.min_edge:
            return {'bet_size': 0, 'reason': 'Insufficient edge'}
        
        kelly_percent = edge / 2.0
        fractional_kelly = kelly_percent * self.kelly_fraction
        confidence_adjusted = fractional_kelly * confidence
        
        max_bet = self.current_bankroll * 0.03
        kelly_bet = self.current_bankroll * confidence_adjusted
        final_bet = min(kelly_bet, max_bet)
        
        return {
            'bet_size': max(0, final_bet),
            'kelly_percent': kelly_percent,
            'fractional_kelly': fractional_kelly,
            'confidence_adjusted': confidence_adjusted,
            'max_possible': max_bet
        }
    
    def evaluate_bet_opportunity(self, prediction: Dict) -> Dict[str, Any]:
        """Evaluate if bet meets criteria"""
        market = prediction.get('market', 'moneyline')
        confidence = prediction.get('confidence', 0.5)
        edge = prediction.get('edge', 0)
        
        thresholds = self.market_thresholds.get(market, self.market_thresholds['moneyline'])
        should_bet = confidence >= thresholds['confidence'] and edge >= thresholds['edge']
        
        if should_bet:
            bet_sizing = self.calculate_optimal_bet_size(edge, confidence)
        else:
            bet_sizing = {'bet_size': 0}
        
        return {
            'should_bet': should_bet,
            'market': market,
            'confidence': confidence,
            'edge': edge,
            'bet_sizing': bet_sizing,
            'thresholds': thresholds
        }
    
    def record_bet_result(self, bet: Dict, outcome: Dict):
        """Record bet outcome"""
        self.bets_placed += 1
        self.total_stakes += bet['amount']
        
        if outcome['result'] == 'win':
            self.bets_won += 1
            self.total_winnings += outcome['payout']
            self.current_bankroll += outcome['payout'] - bet['amount']
        else:
            self.bets_lost += 1
            self.current_bankroll -= bet['amount']
        
        self._adaptive_adjustment()
    
    def _adaptive_adjustment(self):
        """Adapt parameters based on performance"""
        if self.bets_placed < 20:
            return
        
        win_rate = self.bets_won / self.bets_placed
        roi = (self.total_winnings - self.total_stakes) / self.total_stakes if self.total_stakes > 0 else 0
        
        if win_rate > 0.60:
            self.confidence_threshold = max(0.60, self.confidence_threshold - 0.01)
        elif win_rate < 0.48:
            self.confidence_threshold = min(0.85, self.confidence_threshold + 0.02)
        
        if roi > 0.15:
            self.kelly_fraction = min(0.35, self.kelly_fraction + 0.01)
        elif roi < -0.10:
            self.kelly_fraction = max(0.10, self.kelly_fraction - 0.02)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance report"""
        win_rate = self.bets_won / self.bets_placed if self.bets_placed > 0 else 0
        roi = (self.total_winnings - self.total_stakes) / self.total_stakes if self.total_stakes > 0 else 0
        avg_bet = self.total_stakes / self.bets_placed if self.bets_placed > 0 else 0
        
        return {
            'bets_placed': self.bets_placed,
            'bets_won': self.bets_won,
            'bets_lost': self.bets_lost,
            'win_rate': f"{win_rate:.1%}",
            'roi': f"{roi:.1%}",
            'avg_bet_size': f"${avg_bet:.2f}",
            'current_bankroll': f"${self.current_bankroll:.2f}",
            'parameters': {
                'confidence_threshold': f"{self.confidence_threshold:.2f}",
                'kelly_fraction': f"{self.kelly_fraction:.2f}"
            }
        }
    
    def should_pause_betting(self) -> Tuple[bool, str]:
        """Check if betting should pause"""
        if self.current_bankroll < self.initial_bankroll * 0.5:
            return True, "Bankroll depleted"
        if self.bets_placed > 10 and self.bets_won / self.bets_placed < 0.45:
            return True, "Losing streak"
        return False, "Normal"
    
    def save_strategy_state(self):
        """Save state to S3"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'bankroll': self.current_bankroll,
                'confidence_threshold': self.confidence_threshold,
                'kelly_fraction': self.kelly_fraction,
                'performance': self.get_performance_metrics()
            }
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key='strategy/current_state.json',
                Body=json.dumps(state),
                ContentType='application/json'
            )
        except Exception as e:
            logger.error(f"Failed to save: {e}")
