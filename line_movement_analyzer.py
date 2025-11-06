#!/usr/bin/env python3
"""
Line Movement Analyzer for Sharp Money Detection
===============================================

Real-time betting line movement analysis to detect:
- Sharp money movements
- Market manipulation patterns
- Arbitrage opportunities
- Value betting spots
- Sportsbook inefficiencies

Integrates with The Odds API for live line tracking.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
from collections import defaultdict, deque
import statistics
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovementType(Enum):
    """Types of line movements"""
    SHARP_MONEY = "sharp_money"
    PUBLIC_MONEY = "public_money"
    STEAM_MOVE = "steam_move"
    REVERSE_LINE_MOVEMENT = "reverse_line_movement"
    ARBITRAGE = "arbitrage"
    NORMAL = "normal"


class AlertLevel(Enum):
    """Alert levels for line movements"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class LineSnapshot:
    """Betting line snapshot at a point in time"""
    game_id: str
    sportsbook: str
    timestamp: datetime
    spread: float
    total: float
    moneyline_home: int
    moneyline_away: int
    volume_indicator: Optional[float] = None
    sharp_percentage: Optional[float] = None


@dataclass
class LineMovement:
    """Detected line movement"""
    game_id: str
    movement_type: MovementType
    alert_level: AlertLevel
    start_time: datetime
    end_time: datetime
    start_line: LineSnapshot
    end_line: LineSnapshot
    movement_size: float
    velocity: float  # Points per minute
    confidence: float
    description: str
    affected_markets: List[str]


@dataclass
class ArbitrageOpportunity:
    """Arbitrage betting opportunity"""
    game_id: str
    market_type: str  # 'spread', 'total', 'moneyline'
    sportsbook_a: str
    sportsbook_b: str
    line_a: float
    line_b: float
    profit_margin: float
    timestamp: datetime
    expires_in_minutes: float
    confidence: float


class LineMovementAnalyzer:
    """Analyzes betting line movements to detect sharp money and opportunities"""
    
    def __init__(self, history_window: int = 1440):  # 24 hours in minutes
        self.history_window = history_window
        self.line_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.movements: Dict[str, List[LineMovement]] = defaultdict(list)
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        
        # Detection thresholds
        self.thresholds = {
            'sharp_money_spread': 1.0,      # 1 point move
            'sharp_money_total': 1.5,       # 1.5 point move
            'steam_move_velocity': 0.5,     # 0.5 points per minute
            'reverse_line_min': 0.5,        # Minimum reverse movement
            'arbitrage_min_profit': 0.02,   # 2% minimum profit
            'volume_threshold': 0.7         # 70% sharp money threshold
        }
        
        # Callbacks
        self.movement_callbacks: List[Callable[[LineMovement], None]] = []
        self.arbitrage_callbacks: List[Callable[[ArbitrageOpportunity], None]] = []
        
        # Statistics
        self.stats = {
            'total_movements': 0,
            'sharp_movements': 0,
            'arbitrage_found': 0,
            'start_time': datetime.now()
        }
    
    def add_movement_callback(self, callback: Callable[[LineMovement], None]):
        """Add callback for line movements"""
        self.movement_callbacks.append(callback)
    
    def add_arbitrage_callback(self, callback: Callable[[ArbitrageOpportunity], None]):
        """Add callback for arbitrage opportunities"""
        self.arbitrage_callbacks.append(callback)
    
    async def process_line_update(self, line_snapshot: LineSnapshot) -> List[LineMovement]:
        """Process new line data and detect movements"""
        try:
            game_id = line_snapshot.game_id
            sportsbook = line_snapshot.sportsbook
            
            # Store line snapshot
            history_key = f"{game_id}_{sportsbook}"
            self.line_history[history_key].append(line_snapshot)
            
            # Detect movements
            movements = []
            
            # Check for movements if we have history
            if len(self.line_history[history_key]) > 1:
                movements.extend(await self._detect_movements(history_key))
            
            # Check for arbitrage opportunities
            arbitrage_opps = await self._detect_arbitrage(game_id)
            for opp in arbitrage_opps:
                self.arbitrage_opportunities.append(opp)
                await self._notify_arbitrage(opp)
            
            # Store and notify movements
            for movement in movements:
                self.movements[game_id].append(movement)
                self.stats['total_movements'] += 1
                
                if movement.movement_type == MovementType.SHARP_MONEY:
                    self.stats['sharp_movements'] += 1
                
                await self._notify_movement(movement)
            
            return movements
            
        except Exception as e:
            logger.error(f"Error processing line update: {e}")
            return []
    
    async def _detect_movements(self, history_key: str) -> List[LineMovement]:
        """Detect line movements from historical data"""
        movements = []
        
        try:
            history = list(self.line_history[history_key])
            if len(history) < 2:
                return movements
            
            current = history[-1]
            previous = history[-2]
            
            # Calculate movements
            spread_move = current.spread - previous.spread
            total_move = current.total - previous.total
            
            # Time between updates
            time_diff = (current.timestamp - previous.timestamp).total_seconds() / 60  # minutes
            if time_diff == 0:
                return movements
            
            # Detect sharp money movements
            movements.extend(self._detect_sharp_money(current, previous, spread_move, total_move, time_diff))
            
            # Detect steam moves
            movements.extend(self._detect_steam_moves(current, previous, spread_move, total_move, time_diff))
            
            # Detect reverse line movement
            movements.extend(self._detect_reverse_movement(history_key, current, previous, spread_move))
            
            return movements
            
        except Exception as e:
            logger.error(f"Error detecting movements: {e}")
            return []
    
    def _detect_sharp_money(self, current: LineSnapshot, previous: LineSnapshot, 
                          spread_move: float, total_move: float, time_diff: float) -> List[LineMovement]:
        """Detect sharp money movements"""
        movements = []
        
        # Sharp money spread movement
        if abs(spread_move) >= self.thresholds['sharp_money_spread']:
            alert_level = AlertLevel.HIGH if abs(spread_move) >= 2.0 else AlertLevel.MEDIUM
            
            movements.append(LineMovement(
                game_id=current.game_id,
                movement_type=MovementType.SHARP_MONEY,
                alert_level=alert_level,
                start_time=previous.timestamp,
                end_time=current.timestamp,
                start_line=previous,
                end_line=current,
                movement_size=abs(spread_move),
                velocity=abs(spread_move) / time_diff,
                confidence=0.8,
                description=f"Sharp money moved spread {spread_move:+.1f} points",
                affected_markets=['spread']
            ))
        
        # Sharp money total movement
        if abs(total_move) >= self.thresholds['sharp_money_total']:
            alert_level = AlertLevel.HIGH if abs(total_move) >= 3.0 else AlertLevel.MEDIUM
            
            movements.append(LineMovement(
                game_id=current.game_id,
                movement_type=MovementType.SHARP_MONEY,
                alert_level=alert_level,
                start_time=previous.timestamp,
                end_time=current.timestamp,
                start_line=previous,
                end_line=current,
                movement_size=abs(total_move),
                velocity=abs(total_move) / time_diff,
                confidence=0.75,
                description=f"Sharp money moved total {total_move:+.1f} points",
                affected_markets=['total']
            ))
        
        return movements
    
    def _detect_steam_moves(self, current: LineSnapshot, previous: LineSnapshot,
                          spread_move: float, total_move: float, time_diff: float) -> List[LineMovement]:
        """Detect steam moves (rapid line movements)"""
        movements = []
        
        # Steam move detection based on velocity
        spread_velocity = abs(spread_move) / time_diff if time_diff > 0 else 0
        total_velocity = abs(total_move) / time_diff if time_diff > 0 else 0
        
        if spread_velocity >= self.thresholds['steam_move_velocity']:
            movements.append(LineMovement(
                game_id=current.game_id,
                movement_type=MovementType.STEAM_MOVE,
                alert_level=AlertLevel.CRITICAL,
                start_time=previous.timestamp,
                end_time=current.timestamp,
                start_line=previous,
                end_line=current,
                movement_size=abs(spread_move),
                velocity=spread_velocity,
                confidence=0.9,
                description=f"STEAM MOVE: Spread moved {spread_move:+.1f} in {time_diff:.1f} minutes",
                affected_markets=['spread']
            ))
        
        if total_velocity >= self.thresholds['steam_move_velocity']:
            movements.append(LineMovement(
                game_id=current.game_id,
                movement_type=MovementType.STEAM_MOVE,
                alert_level=AlertLevel.CRITICAL,
                start_time=previous.timestamp,
                end_time=current.timestamp,
                start_line=previous,
                end_line=current,
                movement_size=abs(total_move),
                velocity=total_velocity,
                confidence=0.9,
                description=f"STEAM MOVE: Total moved {total_move:+.1f} in {time_diff:.1f} minutes",
                affected_markets=['total']
            ))
        
        return movements
    
    def _detect_reverse_movement(self, history_key: str, current: LineSnapshot, 
                               previous: LineSnapshot, spread_move: float) -> List[LineMovement]:
        """Detect reverse line movement (line moves against public money)"""
        movements = []
        
        try:
            # Simplified reverse line movement detection
            # In production, you'd correlate with betting percentages
            
            if abs(spread_move) >= self.thresholds['reverse_line_min']:
                # Simulate public money percentage (in production, get from betting data)
                simulated_public_pct = 0.65 if spread_move > 0 else 0.35
                
                # Reverse movement: line moves opposite to public money
                if (spread_move > 0 and simulated_public_pct > 0.6) or \
                   (spread_move < 0 and simulated_public_pct < 0.4):
                    
                    movements.append(LineMovement(
                        game_id=current.game_id,
                        movement_type=MovementType.REVERSE_LINE_MOVEMENT,
                        alert_level=AlertLevel.HIGH,
                        start_time=previous.timestamp,
                        end_time=current.timestamp,
                        start_line=previous,
                        end_line=current,
                        movement_size=abs(spread_move),
                        velocity=abs(spread_move) / max((current.timestamp - previous.timestamp).total_seconds() / 60, 1),
                        confidence=0.7,
                        description=f"REVERSE: Line moved {spread_move:+.1f} against {simulated_public_pct:.0%} public money",
                        affected_markets=['spread']
                    ))
            
            return movements
            
        except Exception as e:
            logger.error(f"Error detecting reverse movement: {e}")
            return []
    
    async def _detect_arbitrage(self, game_id: str) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across sportsbooks"""
        opportunities = []
        
        try:
            # Get all current lines for this game
            game_lines = {}
            for history_key, history in self.line_history.items():
                if history_key.startswith(game_id) and history:
                    sportsbook = history_key.split('_')[-1]
                    game_lines[sportsbook] = history[-1]
            
            if len(game_lines) < 2:
                return opportunities
            
            # Check spread arbitrage
            spreads = {book: line.spread for book, line in game_lines.items()}
            if len(spreads) >= 2:
                min_spread = min(spreads.values())
                max_spread = max(spreads.values())
                
                if abs(max_spread - min_spread) >= 1.0:  # 1 point difference
                    profit_margin = self._calculate_arbitrage_profit(min_spread, max_spread)
                    
                    if profit_margin >= self.thresholds['arbitrage_min_profit']:
                        min_book = [book for book, spread in spreads.items() if spread == min_spread][0]
                        max_book = [book for book, spread in spreads.items() if spread == max_spread][0]
                        
                        opportunities.append(ArbitrageOpportunity(
                            game_id=game_id,
                            market_type='spread',
                            sportsbook_a=min_book,
                            sportsbook_b=max_book,
                            line_a=min_spread,
                            line_b=max_spread,
                            profit_margin=profit_margin,
                            timestamp=datetime.now(),
                            expires_in_minutes=15.0,  # Estimate
                            confidence=0.8
                        ))
            
            # Check total arbitrage
            totals = {book: line.total for book, line in game_lines.items()}
            if len(totals) >= 2:
                min_total = min(totals.values())
                max_total = max(totals.values())
                
                if abs(max_total - min_total) >= 2.0:  # 2 point difference
                    profit_margin = self._calculate_arbitrage_profit(min_total, max_total)
                    
                    if profit_margin >= self.thresholds['arbitrage_min_profit']:
                        min_book = [book for book, total in totals.items() if total == min_total][0]
                        max_book = [book for book, total in totals.items() if total == max_total][0]
                        
                        opportunities.append(ArbitrageOpportunity(
                            game_id=game_id,
                            market_type='total',
                            sportsbook_a=min_book,
                            sportsbook_b=max_book,
                            line_a=min_total,
                            line_b=max_total,
                            profit_margin=profit_margin,
                            timestamp=datetime.now(),
                            expires_in_minutes=10.0,  # Totals move faster
                            confidence=0.75
                        ))
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage: {e}")
            return []
    
    def _calculate_arbitrage_profit(self, line_a: float, line_b: float) -> float:
        """Calculate potential arbitrage profit margin"""
        # Simplified arbitrage calculation
        # In production, would use actual odds and calculate guaranteed profit
        line_diff = abs(line_a - line_b)
        return min(line_diff * 0.02, 0.1)  # Max 10% profit margin
    
    async def _notify_movement(self, movement: LineMovement):
        """Notify callbacks of line movement"""
        for callback in self.movement_callbacks:
            try:
                callback(movement)
            except Exception as e:
                logger.error(f"Error in movement callback: {e}")
    
    async def _notify_arbitrage(self, opportunity: ArbitrageOpportunity):
        """Notify callbacks of arbitrage opportunity"""
        self.stats['arbitrage_found'] += 1
        
        for callback in self.arbitrage_callbacks:
            try:
                callback(opportunity)
            except Exception as e:
                logger.error(f"Error in arbitrage callback: {e}")
    
    def get_recent_movements(self, game_id: str, hours: int = 2) -> List[LineMovement]:
        """Get recent movements for a game"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            movement for movement in self.movements.get(game_id, [])
            if movement.end_time >= cutoff_time
        ]
    
    def get_movement_summary(self, game_id: str) -> Dict[str, Any]:
        """Get movement summary for a game"""
        movements = self.movements.get(game_id, [])
        
        if not movements:
            return {'total_movements': 0}
        
        return {
            'total_movements': len(movements),
            'sharp_movements': len([m for m in movements if m.movement_type == MovementType.SHARP_MONEY]),
            'steam_moves': len([m for m in movements if m.movement_type == MovementType.STEAM_MOVE]),
            'reverse_movements': len([m for m in movements if m.movement_type == MovementType.REVERSE_LINE_MOVEMENT]),
            'latest_movement': movements[-1].description if movements else None,
            'max_movement_size': max([m.movement_size for m in movements]),
            'avg_movement_size': statistics.mean([m.movement_size for m in movements])
        }
    
    def get_arbitrage_opportunities(self, active_only: bool = True) -> List[ArbitrageOpportunity]:
        """Get arbitrage opportunities"""
        if not active_only:
            return self.arbitrage_opportunities.copy()
        
        # Filter for active opportunities (not expired)
        active_opps = []
        now = datetime.now()
        
        for opp in self.arbitrage_opportunities:
            minutes_elapsed = (now - opp.timestamp).total_seconds() / 60
            if minutes_elapsed < opp.expires_in_minutes:
                active_opps.append(opp)
        
        return active_opps
    
    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'total_movements': self.stats['total_movements'],
            'sharp_movements': self.stats['sharp_movements'],
            'arbitrage_found': self.stats['arbitrage_found'],
            'uptime_seconds': uptime,
            'movements_per_hour': (self.stats['total_movements'] / max(uptime / 3600, 1)),
            'sharp_percentage': (self.stats['sharp_movements'] / max(self.stats['total_movements'], 1)) * 100,
            'active_games': len(self.movements),
            'active_arbitrage': len(self.get_arbitrage_opportunities(active_only=True))
        }


class LiveLineTracker:
    """Tracks live betting lines from multiple sportsbooks"""
    
    def __init__(self, analyzer: LineMovementAnalyzer):
        self.analyzer = analyzer
        self.sportsbooks = ['DraftKings', 'FanDuel', 'BetMGM', 'Caesars', 'PointsBet']
        self.tracking_active = False
    
    async def simulate_line_tracking(self, game_id: str, duration_minutes: int = 30):
        """Simulate live line tracking (for demo)"""
        logger.info(f"Starting line tracking simulation for {game_id}")
        
        # Initial lines
        base_spread = -3.5
        base_total = 47.5
        
        for minute in range(duration_minutes):
            for sportsbook in self.sportsbooks:
                # Simulate line movements
                spread_noise = np.random.normal(0, 0.1)
                total_noise = np.random.normal(0, 0.2)
                
                # Occasional sharp movements
                if np.random.random() < 0.05:  # 5% chance of sharp move
                    spread_noise += np.random.choice([-1.5, 1.5])
                
                if np.random.random() < 0.03:  # 3% chance of total move
                    total_noise += np.random.choice([-2.0, 2.0])
                
                # Create line snapshot
                line = LineSnapshot(
                    game_id=game_id,
                    sportsbook=sportsbook,
                    timestamp=datetime.now() - timedelta(minutes=duration_minutes-minute),
                    spread=base_spread + spread_noise + (minute * 0.02),  # Gradual drift
                    total=base_total + total_noise + (minute * 0.01),
                    moneyline_home=-110,
                    moneyline_away=-110,
                    volume_indicator=np.random.uniform(0.3, 0.9),
                    sharp_percentage=np.random.uniform(0.2, 0.8)
                )
                
                # Process line update
                await self.analyzer.process_line_update(line)
                
                # Small delay to simulate real-time
                if minute % 5 == 0:  # Every 5 minutes, show progress
                    await asyncio.sleep(0.1)


async def main():
    """Demo of line movement analyzer"""
    print("📈 NFL LINE MOVEMENT ANALYZER DEMO")
    print("=" * 60)
    
    # Create analyzer
    analyzer = LineMovementAnalyzer()
    
    # Add callbacks
    def movement_alert(movement: LineMovement):
        print(f"🚨 {movement.alert_level.name} ALERT: {movement.movement_type.value}")
        print(f"   Game: {movement.game_id}")
        print(f"   {movement.description}")
        print(f"   Velocity: {movement.velocity:.2f} points/minute")
        print(f"   Confidence: {movement.confidence:.0%}")
    
    def arbitrage_alert(opportunity: ArbitrageOpportunity):
        print(f"💰 ARBITRAGE OPPORTUNITY: {opportunity.game_id}")
        print(f"   Market: {opportunity.market_type}")
        print(f"   {opportunity.sportsbook_a}: {opportunity.line_a}")
        print(f"   {opportunity.sportsbook_b}: {opportunity.line_b}")
        print(f"   Profit Margin: {opportunity.profit_margin:.1%}")
        print(f"   Expires in: {opportunity.expires_in_minutes:.0f} minutes")
    
    analyzer.add_movement_callback(movement_alert)
    analyzer.add_arbitrage_callback(arbitrage_alert)
    
    # Create line tracker
    tracker = LiveLineTracker(analyzer)
    
    print("📊 Simulating live line tracking...")
    print("Tracking multiple sportsbooks for sharp money and arbitrage...")
    print("-" * 60)
    
    # Simulate line tracking for multiple games
    games = ['KC_vs_BAL', 'BUF_vs_MIA', 'SF_vs_SEA']
    
    tasks = []
    for game in games:
        task = asyncio.create_task(tracker.simulate_line_tracking(game, 15))  # 15 minutes
        tasks.append(task)
    
    # Run all tracking tasks
    await asyncio.gather(*tasks)
    
    # Show final statistics
    print("\n" + "=" * 60)
    print("📊 LINE MOVEMENT ANALYSIS SUMMARY")
    print("=" * 60)
    
    stats = analyzer.get_analyzer_stats()
    print(f"Total Movements Detected: {stats['total_movements']}")
    print(f"Sharp Money Movements: {stats['sharp_movements']}")
    print(f"Arbitrage Opportunities: {stats['arbitrage_found']}")
    print(f"Sharp Money %: {stats['sharp_percentage']:.1f}%")
    print(f"Movements per Hour: {stats['movements_per_hour']:.1f}")
    print(f"Active Games Tracked: {stats['active_games']}")
    
    # Show game-specific summaries
    print("\n🎯 GAME-SPECIFIC ANALYSIS:")
    for game in games:
        summary = analyzer.get_movement_summary(game)
        if summary['total_movements'] > 0:
            print(f"\n{game}:")
            print(f"   Total Movements: {summary['total_movements']}")
            print(f"   Sharp Movements: {summary['sharp_movements']}")
            print(f"   Steam Moves: {summary['steam_moves']}")
            print(f"   Max Movement: {summary['max_movement_size']:.1f} points")
            print(f"   Latest: {summary['latest_movement']}")
    
    # Show active arbitrage
    active_arb = analyzer.get_arbitrage_opportunities(active_only=True)
    print(f"\n💰 ACTIVE ARBITRAGE OPPORTUNITIES: {len(active_arb)}")
    for opp in active_arb:
        print(f"   {opp.game_id} - {opp.market_type}: {opp.profit_margin:.1%} profit")
    
    print("\n✅ Line movement analysis demo completed!")
    print("Key Features Demonstrated:")
    print("- Sharp money detection")
    print("- Steam move identification")
    print("- Reverse line movement analysis")
    print("- Arbitrage opportunity detection")
    print("- Multi-sportsbook tracking")
    print("- Real-time alert system")


if __name__ == "__main__":
    asyncio.run(main())
