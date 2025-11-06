import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis_engine.referee_analyzer import BayesianRefAnalyzer, RefEvent

@dataclass
class BacktestResult:
    strategy_name: str
    total_bets: int
    winning_bets: int
    total_profit: float
    roi: float
    win_rate: float
    avg_bet_size: float
    max_drawdown: float
    sharpe_ratio: float
    period_start: str
    period_end: str

@dataclass
class BettingSignal:
    game_id: str
    bet_type: str  # 'over', 'under', 'home', 'away'
    recommended_bet: float
    confidence: float
    edge: float
    odds: float
    timestamp: datetime
    crew_id: str
    anomaly_score: float

class RefereeBettingBacktest:
    def __init__(self, initial_bankroll: float = 10000):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.analyzer = BayesianRefAnalyzer()
        self.bet_history = []
        self.strategy_configs = {
            'conservative': {
                'min_anomaly_score': 7.0,
                'max_bet_pct': 0.02,  # 2% of bankroll
                'min_edge': 0.05,     # 5% edge required
                'confidence_threshold': 0.70
            },
            'aggressive': {
                'min_anomaly_score': 5.5,
                'max_bet_pct': 0.05,  # 5% of bankroll
                'min_edge': 0.03,     # 3% edge required
                'confidence_threshold': 0.60
            },
            'high_confidence': {
                'min_anomaly_score': 8.5,
                'max_bet_pct': 0.03,  # 3% of bankroll
                'min_edge': 0.08,     # 8% edge required
                'confidence_threshold': 0.80
            }
        }
    
    def load_historical_data(self, file_path: str) -> List[Dict]:
        """Load historical game data with referee info and outcomes"""
        # This would load your actual historical data
        # For now, creating sample structure
        sample_data = [
            {
                'game_id': 'NFL_2023_W1_KC_DET',
                'date': '2023-09-07',
                'home_team': 'DET',
                'away_team': 'KC', 
                'crew_id': 'CREW_7',
                'ref_events': [
                    {'ref_id': 'REF42', 'penalty': 'holding', 'team': 'away', 'quarter': 4, 'time': '3:12'},
                    {'ref_id': 'REF23', 'penalty': 'PI', 'team': 'home', 'quarter': 2, 'time': '8:45'}
                ],
                'spreads': {
                    'opening': -7.0,
                    'closing': -4.5,
                    'movements': [(-7.0, '09:00'), (-6.5, '12:00'), (-5.0, '15:00'), (-4.5, '18:00')]
                },
                'totals': {
                    'opening': 52.5,
                    'closing': 54.0
                },
                'outcomes': {
                    'home_score': 21,
                    'away_score': 20,
                    'total_score': 41,
                    'spread_result': 'home_cover',  # home won by 1, didn't cover -4.5
                    'total_result': 'under'
                },
                'prime_time': True,
                'weather': 'dome'
            }
            # Add more historical games...
        ]
        return sample_data
    
    def analyze_game_for_signals(self, game_data: Dict) -> List[BettingSignal]:
        """Analyze a single game for betting signals based on referee patterns"""
        signals = []
        
        # Convert game data to RefEvent objects
        ref_events = []
        for event in game_data['ref_events']:
            spread_before = game_data['spreads']['opening']
            spread_after = game_data['spreads']['closing']
            
            ref_event = RefEvent(
                ref_id=event['ref_id'],
                crew_id=game_data['crew_id'],
                game_id=game_data['game_id'],
                penalty_type=event['penalty'],
                team_affected=event['team'],
                game_time=f"Q{event['quarter']}-{event['time']}",
                spread_before=spread_before,
                spread_after=spread_after,
                prime_time=game_data['prime_time']
            )
            ref_events.append(ref_event)
        
        # Analyze crew patterns (you'd need historical data for the crew)
        crew_analysis = self.analyzer.analyze_crew_patterns(ref_events)
        
        if game_data['crew_id'] in crew_analysis:
            crew_stats = crew_analysis[game_data['crew_id']]
            anomaly_score = crew_stats['anomaly_score']
            
            # Generate betting signals based on patterns
            if anomaly_score > 6.0:  # Significant anomaly
                # Spread signal
                if crew_stats['observed_home_bias'] > 0.6:
                    # Crew favors home team - bet away spread
                    signals.append(BettingSignal(
                        game_id=game_data['game_id'],
                        bet_type='away_spread',
                        recommended_bet=0,  # Will be calculated later
                        confidence=min(0.85, anomaly_score / 10),
                        edge=max(0.03, (anomaly_score - 5) * 0.01),
                        odds=-110,  # Standard spread odds
                        timestamp=datetime.strptime(game_data['date'], '%Y-%m-%d'),
                        crew_id=game_data['crew_id'],
                        anomaly_score=anomaly_score
                    ))
                elif crew_stats['observed_home_bias'] < 0.4:
                    # Crew favors away team - bet home spread
                    signals.append(BettingSignal(
                        game_id=game_data['game_id'],
                        bet_type='home_spread',
                        recommended_bet=0,
                        confidence=min(0.85, anomaly_score / 10),
                        edge=max(0.03, (anomaly_score - 5) * 0.01),
                        odds=-110,
                        timestamp=datetime.strptime(game_data['date'], '%Y-%m-%d'),
                        crew_id=game_data['crew_id'],
                        anomaly_score=anomaly_score
                    ))
                
                # Total signal based on penalty frequency
                if crew_stats['avg_spread_movement'] > 2.0:
                    # High volatility crew - bet over
                    signals.append(BettingSignal(
                        game_id=game_data['game_id'],
                        bet_type='over',
                        recommended_bet=0,
                        confidence=min(0.75, (anomaly_score - 2) / 10),
                        edge=max(0.02, (anomaly_score - 6) * 0.005),
                        odds=-110,
                        timestamp=datetime.strptime(game_data['date'], '%Y-%m-%d'),
                        crew_id=game_data['crew_id'],
                        anomaly_score=anomaly_score
                    ))
        
        return signals
    
    def calculate_bet_size(self, signal: BettingSignal, strategy: str, current_bankroll: float) -> float:
        """Calculate optimal bet size using Kelly Criterion"""
        config = self.strategy_configs[strategy]
        
        if (signal.anomaly_score < config['min_anomaly_score'] or
            signal.edge < config['min_edge'] or
            signal.confidence < config['confidence_threshold']):
            return 0
        
        # Kelly Criterion: f = (bp - q) / b
        # where b = odds-1, p = win probability, q = 1-p
        american_odds = signal.odds
        if american_odds > 0:
            decimal_odds = (american_odds / 100) + 1
        else:
            decimal_odds = (100 / abs(american_odds)) + 1
        
        win_prob = signal.confidence
        kelly_fraction = ((decimal_odds - 1) * win_prob - (1 - win_prob)) / (decimal_odds - 1)
        
        # Apply fractional Kelly for risk management
        fractional_kelly = kelly_fraction * 0.25  # Quarter Kelly
        
        # Cap at strategy maximum
        max_bet_fraction = config['max_bet_pct']
        final_fraction = min(fractional_kelly, max_bet_fraction)
        
        return max(0, final_fraction * current_bankroll)
    
    def simulate_bet_outcome(self, signal: BettingSignal, game_outcome: Dict, bet_size: float) -> Dict:
        """Simulate the outcome of a bet"""
        won = False
        
        if signal.bet_type == 'home_spread':
            won = game_outcome['spread_result'] == 'home_cover'
        elif signal.bet_type == 'away_spread':
            won = game_outcome['spread_result'] == 'away_cover'
        elif signal.bet_type == 'over':
            won = game_outcome['total_result'] == 'over'
        elif signal.bet_type == 'under':
            won = game_outcome['total_result'] == 'under'
        
        if won:
            profit = bet_size * 0.909  # Standard -110 odds profit
        else:
            profit = -bet_size
        
        return {
            'won': won,
            'profit': profit,
            'bet_size': bet_size,
            'roi': profit / bet_size if bet_size > 0 else 0
        }
    
    def run_backtest(self, historical_data: List[Dict], strategy: str = 'conservative', 
                     start_date: str = None, end_date: str = None) -> BacktestResult:
        """Run complete backtest"""
        
        # Filter data by date range if specified
        if start_date or end_date:
            filtered_data = []
            for game in historical_data:
                game_date = datetime.strptime(game['date'], '%Y-%m-%d')
                if start_date and game_date < datetime.strptime(start_date, '%Y-%m-%d'):
                    continue
                if end_date and game_date > datetime.strptime(end_date, '%Y-%m-%d'):
                    continue
                filtered_data.append(game)
            historical_data = filtered_data
        
        # Reset bankroll
        self.current_bankroll = self.initial_bankroll
        bet_history = []
        bankroll_history = [self.current_bankroll]
        
        for game_data in historical_data:
            # Generate signals
            signals = self.analyze_game_for_signals(game_data)
            
            for signal in signals:
                bet_size = self.calculate_bet_size(signal, strategy, self.current_bankroll)
                
                if bet_size > 0:
                    # Simulate bet outcome
                    outcome = self.simulate_bet_outcome(signal, game_data['outcomes'], bet_size)
                    
                    # Update bankroll
                    self.current_bankroll += outcome['profit']
                    
                    # Record bet
                    bet_record = {
                        'game_id': signal.game_id,
                        'bet_type': signal.bet_type,
                        'bet_size': bet_size,
                        'odds': signal.odds,
                        'confidence': signal.confidence,
                        'edge': signal.edge,
                        'anomaly_score': signal.anomaly_score,
                        'won': outcome['won'],
                        'profit': outcome['profit'],
                        'roi': outcome['roi'],
                        'bankroll_after': self.current_bankroll,
                        'date': game_data['date']
                    }
                    bet_history.append(bet_record)
                    bankroll_history.append(self.current_bankroll)
        
        # Calculate performance metrics
        if not bet_history:
            return BacktestResult(
                strategy_name=strategy,
                total_bets=0,
                winning_bets=0,
                total_profit=0,
                roi=0,
                win_rate=0,
                avg_bet_size=0,
                max_drawdown=0,
                sharpe_ratio=0,
                period_start=historical_data[0]['date'] if historical_data else '',
                period_end=historical_data[-1]['date'] if historical_data else ''
            )
        
        total_bets = len(bet_history)
        winning_bets = sum(1 for bet in bet_history if bet['won'])
        total_profit = self.current_bankroll - self.initial_bankroll
        roi = total_profit / self.initial_bankroll
        win_rate = winning_bets / total_bets
        avg_bet_size = np.mean([bet['bet_size'] for bet in bet_history])
        
        # Calculate max drawdown
        peak = self.initial_bankroll
        max_drawdown = 0
        for balance in bankroll_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = [(bet_history[i]['bankroll_after'] - bet_history[i-1]['bankroll_after']) / bet_history[i-1]['bankroll_after']
                  for i in range(1, len(bet_history))]
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return BacktestResult(
            strategy_name=strategy,
            total_bets=total_bets,
            winning_bets=winning_bets,
            total_profit=total_profit,
            roi=roi,
            win_rate=win_rate,
            avg_bet_size=avg_bet_size,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            period_start=historical_data[0]['date'],
            period_end=historical_data[-1]['date']
        )
    
    def generate_report(self, results: List[BacktestResult]) -> str:
        """Generate formatted backtest report"""
        report = "\n" + "="*80 + "\n"
        report += "REFEREE ANALYSIS BACKTEST REPORT\n"
        report += "="*80 + "\n\n"
        
        for result in results:
            report += f"Strategy: {result.strategy_name.upper()}\n"
            report += f"Period: {result.period_start} to {result.period_end}\n"
            report += f"Total Bets: {result.total_bets}\n"
            report += f"Winning Bets: {result.winning_bets}\n"
            report += f"Win Rate: {result.win_rate:.1%}\n"
            report += f"Total Profit: ${result.total_profit:,.2f}\n"
            report += f"ROI: {result.roi:.1%}\n"
            report += f"Average Bet Size: ${result.avg_bet_size:,.2f}\n"
            report += f"Max Drawdown: {result.max_drawdown:.1%}\n"
            report += f"Sharpe Ratio: {result.sharpe_ratio:.2f}\n"
            report += "-" * 40 + "\n\n"
        
        return report

# Example usage
if __name__ == "__main__":
    backtest = RefereeBettingBacktest(initial_bankroll=10000)
    
    # Load historical data (you'd replace this with real data loading)
    historical_data = backtest.load_historical_data("historical_games.json")
    
    # Run backtests for different strategies
    strategies = ['conservative', 'aggressive', 'high_confidence']
    results = []
    
    for strategy in strategies:
        result = backtest.run_backtest(historical_data, strategy)
        results.append(result)
    
    # Generate and print report
    report = backtest.generate_report(results)
    print(report)
    
    # Save detailed results
    with open('referee_backtest_results.json', 'w') as f:
        json.dump([{
            'strategy': r.strategy_name,
            'total_bets': r.total_bets,
            'winning_bets': r.winning_bets,
            'total_profit': r.total_profit,
            'roi': r.roi,
            'win_rate': r.win_rate,
            'sharpe_ratio': r.sharpe_ratio
        } for r in results], f, indent=2)