#!/usr/bin/env python3
"""
Line Movement Tracker
=====================
Tracks odds changes over time to detect sharp money movements, steam moves,
and public vs sharp divergence.

Sharp Money Indicators:
- Line moves AGAINST public money (reverse line movement)
- Large line moves (>1 point spread, >1.5 total)
- Steam moves (rapid movement across multiple books)
- Early sharp action vs late public action
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/referee_conspiracy")
LINE_HISTORY_DIR = DATA_DIR / "line_history"
LINE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LineSnapshot:
    """Single point-in-time odds snapshot"""
    timestamp: str
    game_id: str
    bookmaker: str
    
    # Lines
    spread: Optional[float] = None
    spread_odds: Optional[int] = None
    total: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    moneyline_home: Optional[int] = None
    moneyline_away: Optional[int] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LineMovement:
    """Detected line movement with analysis"""
    game_id: str
    market: str  # 'spread', 'total', 'moneyline'
    
    # Movement details
    opening_line: float
    current_line: float
    line_move: float
    move_direction: str  # 'up', 'down'
    
    # Timing
    first_seen: str
    last_update: str
    move_velocity: float  # Points per hour
    
    # Sharp indicators
    is_steam_move: bool = False  # Rapid move across multiple books
    is_reverse_move: bool = False  # Line moves against public %
    sharp_confidence: float = 0.0  # 0-1 sharp money confidence
    
    # Public data (if available)
    public_bet_pct: Optional[float] = None  # % of bets on favored side
    public_money_pct: Optional[float] = None  # % of money on favored side
    
    # Analysis
    edge_detected: bool = False
    recommendation: str = ""  # 'FADE_PUBLIC', 'FOLLOW_SHARP', 'WAIT', 'PASS'
    
    def to_dict(self) -> dict:
        return asdict(self)


class LineMovementTracker:
    """Track and analyze line movements"""
    
    def __init__(self):
        self.history: Dict[str, List[LineSnapshot]] = {}
        self.load_history()
    
    def load_history(self):
        """Load line history from disk"""
        for history_file in LINE_HISTORY_DIR.glob("*.json"):
            try:
                data = json.loads(history_file.read_text())
                game_id = history_file.stem.replace("line_history_", "")
                self.history[game_id] = [LineSnapshot(**snap) for snap in data]
                logger.info(f"Loaded {len(self.history[game_id])} snapshots for {game_id}")
            except Exception as e:
                logger.warning(f"Could not load {history_file}: {e}")
    
    def save_history(self, game_id: str):
        """Save line history for a game"""
        if game_id not in self.history:
            return
        
        history_file = LINE_HISTORY_DIR / f"line_history_{game_id}.json"
        snapshots = [snap.to_dict() for snap in self.history[game_id]]
        history_file.write_text(json.dumps(snapshots, indent=2))
        logger.debug(f"Saved {len(snapshots)} snapshots for {game_id}")
    
    def add_snapshot(self, snapshot: LineSnapshot):
        """Add new odds snapshot"""
        if snapshot.game_id not in self.history:
            self.history[snapshot.game_id] = []
        
        self.history[snapshot.game_id].append(snapshot)
        self.save_history(snapshot.game_id)
    
    def ingest_odds_file(self, odds_file: Path):
        """Ingest odds from nfl_odds_*.json file"""
        try:
            odds_data = json.loads(odds_file.read_text())
            timestamp = datetime.now().isoformat()
            
            for game_odds in odds_data:
                snapshot = LineSnapshot(
                    timestamp=timestamp,
                    game_id=game_odds.get("game_id", ""),
                    bookmaker=game_odds.get("bookmaker", ""),
                    spread=game_odds.get("spread_home"),
                    spread_odds=game_odds.get("spread_home_odds"),
                    total=game_odds.get("total"),
                    over_odds=game_odds.get("over_odds"),
                    under_odds=game_odds.get("under_odds"),
                    moneyline_home=game_odds.get("moneyline_home"),
                    moneyline_away=game_odds.get("moneyline_away"),
                )
                
                self.add_snapshot(snapshot)
            
            logger.info(f"Ingested {len(odds_data)} odds snapshots from {odds_file.name}")
            return len(odds_data)
        
        except Exception as e:
            logger.error(f"Failed to ingest {odds_file}: {e}")
            return 0
    
    def detect_movements(self, game_id: str) -> List[LineMovement]:
        """Detect significant line movements for a game"""
        if game_id not in self.history or len(self.history[game_id]) < 2:
            return []
        
        movements = []
        snapshots = sorted(self.history[game_id], key=lambda s: s.timestamp)
        
        # Group by bookmaker
        by_bookmaker: Dict[str, List[LineSnapshot]] = {}
        for snap in snapshots:
            if snap.bookmaker not in by_bookmaker:
                by_bookmaker[snap.bookmaker] = []
            by_bookmaker[snap.bookmaker].append(snap)
        
        # Analyze each bookmaker's line movement
        for bookmaker, snaps in by_bookmaker.items():
            if len(snaps) < 2:
                continue
            
            opening = snaps[0]
            current = snaps[-1]
            
            # Check spread movement
            if opening.spread is not None and current.spread is not None:
                spread_move = current.spread - opening.spread
                
                if abs(spread_move) >= 0.5:  # Half point or more
                    movement = self._analyze_spread_movement(
                        game_id, opening, current, spread_move, snaps
                    )
                    movements.append(movement)
            
            # Check total movement
            if opening.total is not None and current.total is not None:
                total_move = current.total - opening.total
                
                if abs(total_move) >= 1.0:  # 1 point or more
                    movement = self._analyze_total_movement(
                        game_id, opening, current, total_move, snaps
                    )
                    movements.append(movement)
        
        # Detect steam moves (same direction across multiple books)
        movements = self._detect_steam_moves(movements)
        
        return movements
    
    def _analyze_spread_movement(
        self,
        game_id: str,
        opening: LineSnapshot,
        current: LineSnapshot,
        move: float,
        snapshots: List[LineSnapshot]
    ) -> LineMovement:
        """Analyze spread movement"""
        
        # Calculate velocity
        time_diff = self._time_diff_hours(opening.timestamp, current.timestamp)
        velocity = abs(move) / max(time_diff, 0.1)
        
        # Sharp indicators
        is_steam = velocity > 1.0  # >1 point per hour = steam
        sharp_confidence = min(1.0, velocity * 0.3 + abs(move) * 0.2)
        
        # Recommendation
        if is_steam and abs(move) >= 1.5:
            recommendation = "FOLLOW_SHARP"
            edge_detected = True
        elif abs(move) >= 2.0:
            recommendation = "FOLLOW_SHARP"
            edge_detected = True
        elif abs(move) < 1.0:
            recommendation = "WAIT"
            edge_detected = False
        else:
            recommendation = "MONITOR"
            edge_detected = False
        
        return LineMovement(
            game_id=game_id,
            market="spread",
            opening_line=opening.spread,
            current_line=current.spread,
            line_move=move,
            move_direction="up" if move > 0 else "down",
            first_seen=opening.timestamp,
            last_update=current.timestamp,
            move_velocity=velocity,
            is_steam_move=is_steam,
            sharp_confidence=sharp_confidence,
            edge_detected=edge_detected,
            recommendation=recommendation,
        )
    
    def _analyze_total_movement(
        self,
        game_id: str,
        opening: LineSnapshot,
        current: LineSnapshot,
        move: float,
        snapshots: List[LineSnapshot]
    ) -> LineMovement:
        """Analyze total movement"""
        
        # Calculate velocity
        time_diff = self._time_diff_hours(opening.timestamp, current.timestamp)
        velocity = abs(move) / max(time_diff, 0.1)
        
        # Sharp indicators
        is_steam = velocity > 1.5  # >1.5 points per hour = steam
        sharp_confidence = min(1.0, velocity * 0.25 + abs(move) * 0.15)
        
        # Recommendation
        if is_steam and abs(move) >= 2.0:
            recommendation = "FOLLOW_SHARP"
            edge_detected = True
        elif abs(move) >= 2.5:
            recommendation = "FOLLOW_SHARP"
            edge_detected = True
        elif abs(move) < 1.5:
            recommendation = "WAIT"
            edge_detected = False
        else:
            recommendation = "MONITOR"
            edge_detected = False
        
        return LineMovement(
            game_id=game_id,
            market="total",
            opening_line=opening.total,
            current_line=current.total,
            line_move=move,
            move_direction="up" if move > 0 else "down",
            first_seen=opening.timestamp,
            last_update=current.timestamp,
            move_velocity=velocity,
            is_steam_move=is_steam,
            sharp_confidence=sharp_confidence,
            edge_detected=edge_detected,
            recommendation=recommendation,
        )
    
    def _detect_steam_moves(self, movements: List[LineMovement]) -> List[LineMovement]:
        """Detect steam moves (same direction across multiple bookmakers)"""
        
        # Group by game and market
        by_game_market: Dict[str, List[LineMovement]] = {}
        for mov in movements:
            key = f"{mov.game_id}_{mov.market}"
            if key not in by_game_market:
                by_game_market[key] = []
            by_game_market[key].append(mov)
        
        # Check for consensus movement
        for key, movs in by_game_market.items():
            if len(movs) < 2:
                continue
            
            # All moving in same direction?
            directions = [m.move_direction for m in movs]
            if len(set(directions)) == 1:  # All same direction
                # Mark as steam move
                for mov in movs:
                    mov.is_steam_move = True
                    mov.sharp_confidence = min(1.0, mov.sharp_confidence + 0.2)
                    
                    if mov.recommendation == "MONITOR":
                        mov.recommendation = "FOLLOW_SHARP"
                        mov.edge_detected = True
        
        return movements
    
    def _time_diff_hours(self, start: str, end: str) -> float:
        """Calculate time difference in hours"""
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            diff = (end_dt - start_dt).total_seconds() / 3600
            return max(0.1, diff)
        except Exception:
            return 1.0
    
    def get_sharp_plays(self, min_confidence: float = 0.6) -> List[LineMovement]:
        """Get all high-confidence sharp plays"""
        sharp_plays = []
        
        for game_id in self.history.keys():
            movements = self.detect_movements(game_id)
            for mov in movements:
                if mov.sharp_confidence >= min_confidence and mov.edge_detected:
                    sharp_plays.append(mov)
        
        # Sort by confidence
        sharp_plays.sort(key=lambda m: m.sharp_confidence, reverse=True)
        return sharp_plays
    
    def export_analysis(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Export line movement analysis"""
        output_file = output_file or DATA_DIR / "line_movement_analysis.json"
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "games_tracked": len(self.history),
            "sharp_plays": [],
            "steam_moves": [],
            "all_movements": [],
        }
        
        for game_id in self.history.keys():
            movements = self.detect_movements(game_id)
            
            for mov in movements:
                mov_dict = mov.to_dict()
                analysis["all_movements"].append(mov_dict)
                
                if mov.is_steam_move:
                    analysis["steam_moves"].append(mov_dict)
                
                if mov.sharp_confidence >= 0.6:
                    analysis["sharp_plays"].append(mov_dict)
        
        # Sort
        analysis["sharp_plays"].sort(key=lambda x: x["sharp_confidence"], reverse=True)
        analysis["steam_moves"].sort(key=lambda x: x["move_velocity"], reverse=True)
        
        output_file.write_text(json.dumps(analysis, indent=2))
        logger.info(f"📊 Exported line movement analysis to {output_file}")
        logger.info(f"   Sharp plays: {len(analysis['sharp_plays'])}")
        logger.info(f"   Steam moves: {len(analysis['steam_moves'])}")
        
        return analysis


async def track_line_movements():
    """Main function to track line movements"""
    tracker = LineMovementTracker()
    
    # Ingest all available odds files
    odds_files = sorted(DATA_DIR.glob("nfl_odds_*.json"))
    
    for odds_file in odds_files:
        tracker.ingest_odds_file(odds_file)
    
    # Analyze and export
    analysis = tracker.export_analysis()
    
    # Print sharp plays
    if analysis["sharp_plays"]:
        print("\n" + "=" * 80)
        print("🎯 SHARP PLAYS DETECTED")
        print("=" * 80)
        
        for play in analysis["sharp_plays"][:5]:
            print(f"\n🔥 {play['game_id']} - {play['market'].upper()}")
            print(f"   Line Move: {play['opening_line']} → {play['current_line']} ({play['line_move']:+.1f})")
            print(f"   Velocity: {play['move_velocity']:.2f} points/hour")
            print(f"   Sharp Confidence: {play['sharp_confidence']:.0%}")
            print(f"   Recommendation: {play['recommendation']}")
            if play['is_steam_move']:
                print(f"   🚨 STEAM MOVE DETECTED")
    else:
        print("\n⏳ No sharp plays detected yet. Keep monitoring...")
    
    return analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(track_line_movements())
