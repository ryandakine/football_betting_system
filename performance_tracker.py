#!/usr/bin/env python3
"""
Enhanced Performance Tracking Module for Football Betting System
Tracks betting performance, calculates ROI, and provides detailed analytics.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

logger = logging.getLogger(__name__)


@dataclass
class BetResult:
    """Represents the result of a placed bet."""
    
    bet_id: str
    game_id: str
    sport_type: str
    bet_type: str  # moneyline, spread, total, prop
    selection: str
    odds: float
    stake: float
    expected_value: float
    confidence: float
    placed_at: str
    result: Optional[str] = None  # "won", "lost", "push", "pending"
    profit: Optional[float] = None
    settled_at: Optional[str] = None
    
    def calculate_profit(self) -> float:
        """Calculate profit based on result and odds."""
        if self.result == "won":
            # Convert American odds to decimal for calculation
            if self.odds > 0:
                decimal_odds = 1 + (self.odds / 100)
            else:
                decimal_odds = 1 + (100 / abs(self.odds))
            return (self.stake * decimal_odds) - self.stake
        elif self.result == "lost":
            return -self.stake
        elif self.result == "push":
            return 0
        return 0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for the betting system."""
    
    # Basic metrics
    total_bets: int = 0
    bets_won: int = 0
    bets_lost: int = 0
    bets_pushed: int = 0
    bets_pending: int = 0
    
    # Financial metrics
    total_staked: float = 0.0
    total_profit: float = 0.0
    roi: float = 0.0
    average_stake: float = 0.0
    average_odds: float = 0.0
    
    # Performance metrics
    win_rate: float = 0.0
    expected_value_accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    
    # By market type
    performance_by_market: Dict[str, Dict] = field(default_factory=dict)
    
    # By confidence level
    performance_by_confidence: Dict[str, Dict] = field(default_factory=dict)
    
    # Time-based metrics
    daily_performance: List[Dict] = field(default_factory=list)
    weekly_performance: List[Dict] = field(default_factory=list)
    monthly_performance: List[Dict] = field(default_factory=list)
    
    # Streaks
    current_streak: int = 0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for JSON serialization."""
        return asdict(self)


class PerformanceTracker:
    """Tracks and analyzes betting system performance."""
    
    def __init__(self, initial_bankroll: float = 1000.0, data_dir: str = "performance_data"):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.bet_history: List[BetResult] = []
        self.metrics = PerformanceMetrics()
        self.session_start = datetime.now()
        
        # Load historical data if exists
        self._load_historical_data()
        
        logger.info(f"Performance Tracker initialized with bankroll: ${initial_bankroll:,.2f}")
    
    def track_bet(self, bet_data: Dict) -> str:
        """Track a new bet placement."""
        bet_id = f"{bet_data['game_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        bet_result = BetResult(
            bet_id=bet_id,
            game_id=bet_data.get('game_id', ''),
            sport_type=bet_data.get('sport_type', 'nfl'),
            bet_type=bet_data.get('bet_type', 'moneyline'),
            selection=bet_data.get('selection', ''),
            odds=bet_data.get('odds', 0),
            stake=bet_data.get('stake', 0),
            expected_value=bet_data.get('expected_value', 0),
            confidence=bet_data.get('confidence', 0),
            placed_at=datetime.now().isoformat(),
            result="pending"
        )
        
        self.bet_history.append(bet_result)
        self.metrics.total_bets += 1
        self.metrics.bets_pending += 1
        self.metrics.total_staked += bet_result.stake
        
        logger.info(f"Bet tracked: {bet_id} - ${bet_result.stake:.2f} on {bet_result.selection}")
        
        return bet_id
    
    def update_bet_result(self, bet_id: str, result: str, settled_at: Optional[str] = None):
        """Update the result of a bet."""
        for bet in self.bet_history:
            if bet.bet_id == bet_id:
                if bet.result == "pending":
                    self.metrics.bets_pending -= 1
                
                bet.result = result
                bet.settled_at = settled_at or datetime.now().isoformat()
                bet.profit = bet.calculate_profit()
                
                # Update metrics
                if result == "won":
                    self.metrics.bets_won += 1
                    self.current_bankroll += bet.profit
                    self._update_streak(True)
                elif result == "lost":
                    self.metrics.bets_lost += 1
                    self.current_bankroll += bet.profit  # profit is negative for losses
                    self._update_streak(False)
                elif result == "push":
                    self.metrics.bets_pushed += 1
                
                self.metrics.total_profit += bet.profit
                
                # Update peak bankroll and drawdown
                if self.current_bankroll > self.peak_bankroll:
                    self.peak_bankroll = self.current_bankroll
                
                drawdown = (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll
                if drawdown > self.metrics.max_drawdown:
                    self.metrics.max_drawdown = drawdown
                
                logger.info(f"Bet {bet_id} settled: {result} - Profit: ${bet.profit:.2f}")
                break
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not self.bet_history:
            return self.metrics
        
        settled_bets = [b for b in self.bet_history if b.result != "pending"]
        
        if settled_bets:
            # Basic metrics
            self.metrics.win_rate = self.metrics.bets_won / len(settled_bets) if settled_bets else 0
            
            # ROI
            if self.metrics.total_staked > 0:
                self.metrics.roi = (self.metrics.total_profit / self.metrics.total_staked) * 100
            
            # Average metrics
            self.metrics.average_stake = self.metrics.total_staked / self.metrics.total_bets
            odds_list = [b.odds for b in self.bet_history if b.odds != 0]
            self.metrics.average_odds = statistics.mean(odds_list) if odds_list else 0
            
            # Profit factor
            wins_profit = sum(b.profit for b in settled_bets if b.profit > 0)
            losses = abs(sum(b.profit for b in settled_bets if b.profit < 0))
            if losses > 0:
                self.metrics.profit_factor = wins_profit / losses
            
            # Expected value accuracy
            if settled_bets:
                expected_total = sum(b.expected_value for b in settled_bets)
                actual_total = sum(b.profit for b in settled_bets)
                if expected_total != 0:
                    self.metrics.expected_value_accuracy = (actual_total / expected_total) * 100
            
            # Sharpe ratio (simplified)
            if len(settled_bets) > 1:
                returns = [b.profit / b.stake for b in settled_bets]
                if returns:
                    avg_return = statistics.mean(returns)
                    std_return = statistics.stdev(returns) if len(returns) > 1 else 0
                    if std_return > 0:
                        self.metrics.sharpe_ratio = avg_return / std_return
            
            # Performance by market type
            self._calculate_market_performance()
            
            # Performance by confidence level
            self._calculate_confidence_performance()
            
            # Time-based performance
            self._calculate_time_based_performance()
        
        return self.metrics
    
    def _calculate_market_performance(self):
        """Calculate performance metrics by market type."""
        market_types = set(b.bet_type for b in self.bet_history)
        
        for market in market_types:
            market_bets = [b for b in self.bet_history if b.bet_type == market and b.result != "pending"]
            if market_bets:
                wins = sum(1 for b in market_bets if b.result == "won")
                total = len(market_bets)
                profit = sum(b.profit for b in market_bets)
                staked = sum(b.stake for b in market_bets)
                
                self.metrics.performance_by_market[market] = {
                    "total_bets": total,
                    "wins": wins,
                    "win_rate": wins / total if total > 0 else 0,
                    "profit": profit,
                    "roi": (profit / staked * 100) if staked > 0 else 0
                }
    
    def _calculate_confidence_performance(self):
        """Calculate performance metrics by confidence level."""
        confidence_ranges = [
            ("low", 0.0, 0.6),
            ("medium", 0.6, 0.75),
            ("high", 0.75, 0.85),
            ("very_high", 0.85, 1.0)
        ]
        
        for label, min_conf, max_conf in confidence_ranges:
            range_bets = [
                b for b in self.bet_history 
                if min_conf <= b.confidence < max_conf and b.result != "pending"
            ]
            
            if range_bets:
                wins = sum(1 for b in range_bets if b.result == "won")
                total = len(range_bets)
                profit = sum(b.profit for b in range_bets)
                staked = sum(b.stake for b in range_bets)
                
                self.metrics.performance_by_confidence[label] = {
                    "range": f"{min_conf:.0%}-{max_conf:.0%}",
                    "total_bets": total,
                    "wins": wins,
                    "win_rate": wins / total if total > 0 else 0,
                    "profit": profit,
                    "roi": (profit / staked * 100) if staked > 0 else 0
                }
    
    def _calculate_time_based_performance(self):
        """Calculate daily, weekly, and monthly performance."""
        if not self.bet_history:
            return
        
        # Group bets by date
        from collections import defaultdict
        daily_bets = defaultdict(list)
        
        for bet in self.bet_history:
            if bet.result != "pending":
                bet_date = datetime.fromisoformat(bet.placed_at).date()
                daily_bets[bet_date].append(bet)
        
        # Calculate daily performance
        for date, bets in daily_bets.items():
            daily_profit = sum(b.profit for b in bets)
            daily_staked = sum(b.stake for b in bets)
            wins = sum(1 for b in bets if b.result == "won")
            
            self.metrics.daily_performance.append({
                "date": date.isoformat(),
                "bets": len(bets),
                "wins": wins,
                "profit": daily_profit,
                "roi": (daily_profit / daily_staked * 100) if daily_staked > 0 else 0
            })
    
    def _update_streak(self, won: bool):
        """Update winning/losing streak tracking."""
        if won:
            if self.metrics.current_streak >= 0:
                self.metrics.current_streak += 1
            else:
                self.metrics.current_streak = 1
            
            if self.metrics.current_streak > self.metrics.longest_win_streak:
                self.metrics.longest_win_streak = self.metrics.current_streak
        else:
            if self.metrics.current_streak <= 0:
                self.metrics.current_streak -= 1
            else:
                self.metrics.current_streak = -1
            
            if abs(self.metrics.current_streak) > self.metrics.longest_loss_streak:
                self.metrics.longest_loss_streak = abs(self.metrics.current_streak)
    
    def get_summary(self) -> Dict:
        """Get a summary of current performance."""
        self.calculate_metrics()
        
        return {
            "session_info": {
                "start_time": self.session_start.isoformat(),
                "initial_bankroll": self.initial_bankroll,
                "current_bankroll": self.current_bankroll,
                "profit_loss": self.current_bankroll - self.initial_bankroll,
                "roi": self.metrics.roi
            },
            "betting_stats": {
                "total_bets": self.metrics.total_bets,
                "wins": self.metrics.bets_won,
                "losses": self.metrics.bets_lost,
                "pushes": self.metrics.bets_pushed,
                "pending": self.metrics.bets_pending,
                "win_rate": self.metrics.win_rate,
                "profit_factor": self.metrics.profit_factor
            },
            "risk_metrics": {
                "max_drawdown": self.metrics.max_drawdown,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "current_streak": self.metrics.current_streak,
                "longest_win_streak": self.metrics.longest_win_streak,
                "longest_loss_streak": self.metrics.longest_loss_streak
            },
            "market_performance": self.metrics.performance_by_market,
            "confidence_performance": self.metrics.performance_by_confidence
        }
    
    def save_session(self):
        """Save current session data to file."""
        session_file = self.data_dir / f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": self.current_bankroll,
            "bet_history": [asdict(bet) for bet in self.bet_history],
            "metrics": self.metrics.to_dict(),
            "summary": self.get_summary()
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.info(f"Session data saved to {session_file}")
    
    def _load_historical_data(self):
        """Load historical betting data if available."""
        history_file = self.data_dir / "betting_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    # Load bet history
                    for bet_data in data.get("bet_history", []):
                        self.bet_history.append(BetResult(**bet_data))
                    
                    # Recalculate metrics from history
                    self.calculate_metrics()
                    
                    logger.info(f"Loaded {len(self.bet_history)} historical bets")
            except Exception as e:
                logger.error(f"Failed to load historical data: {e}")
    
    def generate_report(self) -> str:
        """Generate a formatted performance report."""
        summary = self.get_summary()
        
        report = []
        report.append("=" * 60)
        report.append("📊 BETTING PERFORMANCE REPORT")
        report.append("=" * 60)
        
        # Session info
        report.append("\n📈 SESSION OVERVIEW:")
        report.append(f"  Start Time: {summary['session_info']['start_time']}")
        report.append(f"  Initial Bankroll: ${summary['session_info']['initial_bankroll']:,.2f}")
        report.append(f"  Current Bankroll: ${summary['session_info']['current_bankroll']:,.2f}")
        report.append(f"  Profit/Loss: ${summary['session_info']['profit_loss']:,.2f}")
        report.append(f"  ROI: {summary['session_info']['roi']:.2f}%")
        
        # Betting stats
        report.append("\n🎯 BETTING STATISTICS:")
        stats = summary['betting_stats']
        report.append(f"  Total Bets: {stats['total_bets']}")
        report.append(f"  Wins: {stats['wins']} | Losses: {stats['losses']} | Pushes: {stats['pushes']}")
        report.append(f"  Win Rate: {stats['win_rate']:.1%}")
        report.append(f"  Profit Factor: {stats['profit_factor']:.2f}")
        
        # Risk metrics
        report.append("\n⚠️ RISK METRICS:")
        risk = summary['risk_metrics']
        report.append(f"  Max Drawdown: {risk['max_drawdown']:.1%}")
        report.append(f"  Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        report.append(f"  Current Streak: {risk['current_streak']}")
        
        # Market performance
        if summary['market_performance']:
            report.append("\n📊 PERFORMANCE BY MARKET:")
            for market, perf in summary['market_performance'].items():
                report.append(f"  {market.upper()}:")
                report.append(f"    Bets: {perf['total_bets']} | Win Rate: {perf['win_rate']:.1%}")
                report.append(f"    ROI: {perf['roi']:.2f}%")
        
        # Confidence performance
        if summary['confidence_performance']:
            report.append("\n🎯 PERFORMANCE BY CONFIDENCE:")
            for level, perf in summary['confidence_performance'].items():
                report.append(f"  {level.upper()} ({perf['range']}):")
                report.append(f"    Bets: {perf['total_bets']} | Win Rate: {perf['win_rate']:.1%}")
                report.append(f"    ROI: {perf['roi']:.2f}%")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = PerformanceTracker(initial_bankroll=1000.0)
    
    # Track some example bets
    bet_id1 = tracker.track_bet({
        "game_id": "BUF_KC_20250101",
        "sport_type": "nfl",
        "bet_type": "moneyline",
        "selection": "BUF",
        "odds": -110,
        "stake": 50,
        "expected_value": 5.5,
        "confidence": 0.75
    })
    
    bet_id2 = tracker.track_bet({
        "game_id": "GB_DET_20250101",
        "sport_type": "nfl",
        "bet_type": "spread",
        "selection": "GB -3.5",
        "odds": -105,
        "stake": 40,
        "expected_value": 3.8,
        "confidence": 0.68
    })
    
    # Simulate some results
    tracker.update_bet_result(bet_id1, "won")
    tracker.update_bet_result(bet_id2, "lost")
    
    # Generate report
    print(tracker.generate_report())
    
    # Save session
    tracker.save_session()
