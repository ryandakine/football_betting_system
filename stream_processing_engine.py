#!/usr/bin/env python3
"""
Stream Processing Engine for NFL Real-Time Intelligence
======================================================

Continuous model updates and real-time prediction engine:
- Live game state tracking
- Dynamic model updates
- Prediction recalculation
- Significant change detection
- Sub-second response times

Integrates with WebSocket client and message queue system.
"""

import asyncio
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import pickle
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameState(Enum):
    """Game state enumeration"""
    PREGAME = "pregame"
    ACTIVE = "active"
    HALFTIME = "halftime"
    FINAL = "final"
    OVERTIME = "overtime"


@dataclass
class GameSnapshot:
    """Complete game state snapshot"""
    game_id: str
    timestamp: datetime
    quarter: int
    time_remaining: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    down: Optional[int] = None
    yards_to_go: Optional[int] = None
    field_position: Optional[int] = None
    possession: Optional[str] = None
    game_state: GameState = GameState.ACTIVE
    
    # Advanced metrics
    home_win_probability: float = 0.5
    total_points_projection: float = 0.0
    spread_movement: float = 0.0
    
    # Performance stats
    home_rushing_yards: int = 0
    home_passing_yards: int = 0
    away_rushing_yards: int = 0
    away_passing_yards: int = 0
    home_turnovers: int = 0
    away_turnovers: int = 0


@dataclass
class PredictionUpdate:
    """Model prediction update"""
    game_id: str
    timestamp: datetime
    previous_prediction: Dict[str, float]
    new_prediction: Dict[str, float]
    confidence_change: float
    significant_change: bool
    trigger_event: str
    model_version: str = "1.0"


class LiveGameModel:
    """Real-time game prediction model"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.version = "1.0"
        self.last_update = datetime.now()
        
        # Model parameters (simplified for demo)
        self.weights = {
            'score_differential': 0.3,
            'time_remaining': 0.2,
            'field_position': 0.15,
            'down_distance': 0.1,
            'turnovers': 0.25
        }
        
        # Performance tracking
        self.predictions_made = 0
        self.accuracy_score = 0.0
        self.processing_times = deque(maxlen=100)
    
    def predict_win_probability(self, game_snapshot: GameSnapshot) -> Dict[str, float]:
        """Predict win probability for both teams"""
        start_time = time.time()
        
        try:
            # Calculate base probability from score differential
            score_diff = game_snapshot.home_score - game_snapshot.away_score
            base_prob = 0.5 + (score_diff * 0.02)  # 2% per point
            
            # Time adjustment
            quarter_weight = {1: 0.7, 2: 0.8, 3: 0.9, 4: 1.0}.get(game_snapshot.quarter, 1.0)
            time_factor = self._parse_time_remaining(game_snapshot.time_remaining)
            time_adjustment = (1 - time_factor) * 0.1 * quarter_weight
            
            # Field position adjustment
            field_adj = 0.0
            if game_snapshot.field_position:
                if game_snapshot.field_position > 50:  # In opponent territory
                    field_adj = 0.05
                elif game_snapshot.field_position < 20:  # In own red zone
                    field_adj = -0.03
            
            # Turnover impact
            turnover_diff = game_snapshot.away_turnovers - game_snapshot.home_turnovers
            turnover_adj = turnover_diff * 0.08
            
            # Calculate final probability
            home_prob = np.clip(
                base_prob + time_adjustment + field_adj + turnover_adj,
                0.01, 0.99
            )
            away_prob = 1.0 - home_prob
            
            result = {
                'home_win_prob': home_prob,
                'away_win_prob': away_prob,
                'confidence': abs(home_prob - 0.5) * 2,  # 0-1 scale
                'model_version': self.version
            }
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.predictions_made += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction model: {e}")
            return {
                'home_win_prob': 0.5,
                'away_win_prob': 0.5,
                'confidence': 0.0,
                'model_version': self.version
            }
    
    def predict_total_points(self, game_snapshot: GameSnapshot) -> float:
        """Predict final total points"""
        current_total = game_snapshot.home_score + game_snapshot.away_score
        time_factor = self._parse_time_remaining(game_snapshot.time_remaining)
        quarter_multiplier = {1: 4.0, 2: 2.0, 3: 1.5, 4: 1.0}.get(game_snapshot.quarter, 1.0)
        
        # Estimate remaining points based on current pace
        if game_snapshot.quarter > 0:
            points_per_quarter = current_total / game_snapshot.quarter
            remaining_points = points_per_quarter * time_factor * quarter_multiplier
            return current_total + remaining_points
        
        return current_total + 21  # Default assumption
    
    def _parse_time_remaining(self, time_str: str) -> float:
        """Parse time remaining into fraction of quarter"""
        try:
            if ':' in time_str:
                minutes, seconds = map(int, time_str.split(':'))
                total_seconds = minutes * 60 + seconds
                return total_seconds / 900.0  # 15 minutes per quarter
            return 0.0
        except:
            return 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            'model_name': self.model_name,
            'version': self.version,
            'predictions_made': self.predictions_made,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'accuracy_score': self.accuracy_score,
            'last_update': self.last_update.isoformat()
        }


class StreamProcessor:
    """Main stream processing engine"""
    
    def __init__(self):
        self.game_states: Dict[str, GameSnapshot] = {}
        self.models: Dict[str, LiveGameModel] = {}
        self.prediction_history: Dict[str, List[PredictionUpdate]] = defaultdict(list)
        
        # Change detection thresholds
        self.significance_thresholds = {
            'win_probability': 0.05,  # 5% change
            'total_points': 3.0,      # 3 point change
            'spread': 1.0             # 1 point spread change
        }
        
        # Performance tracking
        self.processing_stats = {
            'events_processed': 0,
            'predictions_generated': 0,
            'significant_changes': 0,
            'start_time': datetime.now()
        }
        
        # Notification callbacks
        self.change_callbacks: List[Callable[[PredictionUpdate], None]] = []
        
        # Initialize default model
        self.add_model('default', LiveGameModel('DefaultNFLModel'))
    
    def add_model(self, model_id: str, model: LiveGameModel):
        """Add a prediction model"""
        self.models[model_id] = model
        logger.info(f"Added model: {model_id}")
    
    def add_change_callback(self, callback: Callable[[PredictionUpdate], None]):
        """Add callback for significant prediction changes"""
        self.change_callbacks.append(callback)
    
    async def process_event(self, event_data: Dict[str, Any]) -> bool:
        """Process incoming event and update predictions"""
        try:
            start_time = time.time()
            
            game_id = event_data.get('game_id', 'unknown')
            event_type = event_data.get('event_type', 'unknown')
            
            # Update game state
            updated = self._update_game_state(game_id, event_data)
            
            if updated:
                # Generate new predictions
                await self._generate_predictions(game_id, event_type)
                
                self.processing_stats['events_processed'] += 1
                
                # Log processing time
                processing_time = time.time() - start_time
                if processing_time > 0.5:  # Log slow processing
                    logger.warning(f"Slow event processing: {processing_time:.3f}s for {event_type}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing event: {e}")
            return False
    
    def _update_game_state(self, game_id: str, event_data: Dict[str, Any]) -> bool:
        """Update game state from event data"""
        try:
            # Get or create game snapshot
            if game_id not in self.game_states:
                self.game_states[game_id] = GameSnapshot(
                    game_id=game_id,
                    timestamp=datetime.now(),
                    quarter=1,
                    time_remaining="15:00",
                    home_team=event_data.get('home_team', 'HOME'),
                    away_team=event_data.get('away_team', 'AWAY'),
                    home_score=0,
                    away_score=0
                )
            
            snapshot = self.game_states[game_id]
            snapshot.timestamp = datetime.now()
            updated = False
            
            # Update based on event type
            event_type = event_data.get('event_type', '')
            data = event_data.get('data', {})
            
            if event_type == 'score':
                if 'home_score' in data:
                    snapshot.home_score = data['home_score']
                    updated = True
                if 'away_score' in data:
                    snapshot.away_score = data['away_score']
                    updated = True
            
            elif event_type == 'play':
                if 'down' in data:
                    snapshot.down = data['down']
                if 'yards_to_go' in data:
                    snapshot.yards_to_go = data['yards_to_go']
                if 'field_position' in data:
                    snapshot.field_position = data['field_position']
                updated = True
            
            elif event_type == 'quarter_change':
                if 'new_quarter' in data:
                    snapshot.quarter = data['new_quarter']
                    snapshot.time_remaining = "15:00"
                    updated = True
            
            elif event_type == 'timeout' or event_type == 'injury':
                # These don't change core game state but trigger prediction updates
                updated = True
            
            # Update game state enum
            if snapshot.quarter > 4:
                snapshot.game_state = GameState.OVERTIME
            elif event_type == 'halftime':
                snapshot.game_state = GameState.HALFTIME
            elif event_type == 'final':
                snapshot.game_state = GameState.FINAL
            else:
                snapshot.game_state = GameState.ACTIVE
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating game state: {e}")
            return False
    
    async def _generate_predictions(self, game_id: str, trigger_event: str):
        """Generate new predictions for a game"""
        try:
            if game_id not in self.game_states:
                return
            
            snapshot = self.game_states[game_id]
            model = self.models.get('default')
            
            if not model:
                return
            
            # Get previous prediction
            previous_predictions = {}
            if game_id in self.prediction_history and self.prediction_history[game_id]:
                previous_predictions = self.prediction_history[game_id][-1].new_prediction
            
            # Generate new predictions
            win_prob_prediction = model.predict_win_probability(snapshot)
            total_points_prediction = model.predict_total_points(snapshot)
            
            new_prediction = {
                'home_win_prob': win_prob_prediction['home_win_prob'],
                'away_win_prob': win_prob_prediction['away_win_prob'],
                'total_points': total_points_prediction,
                'confidence': win_prob_prediction['confidence']
            }
            
            # Check for significant changes
            significant_change = self._is_significant_change(previous_predictions, new_prediction)
            
            # Calculate confidence change
            prev_confidence = previous_predictions.get('confidence', 0.5)
            confidence_change = new_prediction['confidence'] - prev_confidence
            
            # Create prediction update
            prediction_update = PredictionUpdate(
                game_id=game_id,
                timestamp=datetime.now(),
                previous_prediction=previous_predictions,
                new_prediction=new_prediction,
                confidence_change=confidence_change,
                significant_change=significant_change,
                trigger_event=trigger_event,
                model_version=model.version
            )
            
            # Store prediction
            self.prediction_history[game_id].append(prediction_update)
            
            # Update stats
            self.processing_stats['predictions_generated'] += 1
            if significant_change:
                self.processing_stats['significant_changes'] += 1
            
            # Notify callbacks for significant changes
            if significant_change:
                for callback in self.change_callbacks:
                    try:
                        callback(prediction_update)
                    except Exception as e:
                        logger.error(f"Error in change callback: {e}")
            
            logger.debug(f"Generated prediction for {game_id}: {new_prediction}")
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
    
    def _is_significant_change(self, previous: Dict[str, float], new: Dict[str, float]) -> bool:
        """Determine if prediction change is significant"""
        if not previous:
            return True  # First prediction is always significant
        
        # Check win probability change
        prev_home_prob = previous.get('home_win_prob', 0.5)
        new_home_prob = new.get('home_win_prob', 0.5)
        prob_change = abs(new_home_prob - prev_home_prob)
        
        if prob_change >= self.significance_thresholds['win_probability']:
            return True
        
        # Check total points change
        prev_total = previous.get('total_points', 0)
        new_total = new.get('total_points', 0)
        total_change = abs(new_total - prev_total)
        
        if total_change >= self.significance_thresholds['total_points']:
            return True
        
        return False
    
    def get_game_state(self, game_id: str) -> Optional[GameSnapshot]:
        """Get current game state"""
        return self.game_states.get(game_id)
    
    def get_latest_prediction(self, game_id: str) -> Optional[Dict[str, float]]:
        """Get latest prediction for a game"""
        if game_id in self.prediction_history and self.prediction_history[game_id]:
            return self.prediction_history[game_id][-1].new_prediction
        return None
    
    def get_prediction_history(self, game_id: str) -> List[PredictionUpdate]:
        """Get prediction history for a game"""
        return self.prediction_history.get(game_id, [])
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        uptime = (datetime.now() - self.processing_stats['start_time']).total_seconds()
        
        stats = self.processing_stats.copy()
        stats['uptime_seconds'] = uptime
        stats['events_per_second'] = stats['events_processed'] / max(uptime, 1)
        stats['predictions_per_second'] = stats['predictions_generated'] / max(uptime, 1)
        stats['active_games'] = len(self.game_states)
        
        # Add model stats
        model_stats = {}
        for model_id, model in self.models.items():
            model_stats[model_id] = model.get_performance_stats()
        stats['model_stats'] = model_stats
        
        return stats


# Integration with message queue
class StreamProcessingIntegration:
    """Integrates stream processor with message queue system"""
    
    def __init__(self, stream_processor: StreamProcessor):
        self.stream_processor = stream_processor
        self.notification_callbacks: List[Callable[[PredictionUpdate], None]] = []
        
        # Add stream processor callback
        self.stream_processor.add_change_callback(self._handle_significant_change)
    
    def add_notification_callback(self, callback: Callable[[PredictionUpdate], None]):
        """Add callback for prediction notifications"""
        self.notification_callbacks.append(callback)
    
    def _handle_significant_change(self, prediction_update: PredictionUpdate):
        """Handle significant prediction changes"""
        logger.info(f"🚨 SIGNIFICANT CHANGE: {prediction_update.game_id}")
        logger.info(f"   Trigger: {prediction_update.trigger_event}")
        logger.info(f"   Home Win Prob: {prediction_update.previous_prediction.get('home_win_prob', 0):.1%} → {prediction_update.new_prediction['home_win_prob']:.1%}")
        
        # Notify all callbacks
        for callback in self.notification_callbacks:
            try:
                callback(prediction_update)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
    
    async def process_queue_event(self, queue_event) -> bool:
        """Process event from message queue"""
        try:
            # Convert queue event to format expected by stream processor
            event_data = {
                'game_id': queue_event.game_id,
                'event_type': queue_event.event_type,
                'data': queue_event.data,
                'timestamp': queue_event.timestamp.isoformat(),
                'source': queue_event.source
            }
            
            return await self.stream_processor.process_event(event_data)
            
        except Exception as e:
            logger.error(f"Error processing queue event: {e}")
            return False


async def main():
    """Demo of stream processing engine"""
    print("⚡ NFL STREAM PROCESSING ENGINE DEMO")
    print("=" * 50)
    
    # Create stream processor
    processor = StreamProcessor()
    
    # Add notification callback
    def notification_handler(update: PredictionUpdate):
        print(f"📢 ALERT: {update.game_id} - {update.trigger_event}")
        print(f"   Win Probability Change: {update.confidence_change:+.1%}")
        print(f"   New Prediction: {update.new_prediction}")
    
    processor.add_change_callback(notification_handler)
    
    # Simulate game events
    game_events = [
        {'game_id': 'KC_vs_BAL', 'event_type': 'score', 'data': {'home_score': 7, 'away_score': 0}},
        {'game_id': 'KC_vs_BAL', 'event_type': 'play', 'data': {'down': 1, 'yards_to_go': 10, 'field_position': 25}},
        {'game_id': 'KC_vs_BAL', 'event_type': 'score', 'data': {'home_score': 7, 'away_score': 7}},
        {'game_id': 'KC_vs_BAL', 'event_type': 'injury', 'data': {'player': 'Patrick Mahomes', 'severity': 'questionable'}},
        {'game_id': 'KC_vs_BAL', 'event_type': 'score', 'data': {'home_score': 14, 'away_score': 7}},
        {'game_id': 'KC_vs_BAL', 'event_type': 'quarter_change', 'data': {'new_quarter': 2}},
    ]
    
    print("🏈 Processing live game events...")
    
    for i, event in enumerate(game_events):
        print(f"\n--- Event {i+1}: {event['event_type']} ---")
        await processor.process_event(event)
        
        # Show current prediction
        prediction = processor.get_latest_prediction('KC_vs_BAL')
        if prediction:
            print(f"Current Prediction:")
            print(f"  KC Win Prob: {prediction['home_win_prob']:.1%}")
            print(f"  BAL Win Prob: {prediction['away_win_prob']:.1%}")
            print(f"  Total Points: {prediction['total_points']:.1f}")
            print(f"  Confidence: {prediction['confidence']:.1%}")
        
        await asyncio.sleep(0.5)  # Simulate real-time delay
    
    # Show final statistics
    print("\n" + "=" * 50)
    print("📊 PROCESSING STATISTICS")
    print("=" * 50)
    
    stats = processor.get_processing_stats()
    print(f"Events Processed: {stats['events_processed']}")
    print(f"Predictions Generated: {stats['predictions_generated']}")
    print(f"Significant Changes: {stats['significant_changes']}")
    print(f"Processing Rate: {stats['events_per_second']:.1f} events/sec")
    print(f"Active Games: {stats['active_games']}")
    
    # Show model performance
    for model_id, model_stats in stats['model_stats'].items():
        print(f"\nModel '{model_id}':")
        print(f"  Predictions: {model_stats['predictions_made']}")
        print(f"  Avg Processing Time: {model_stats['avg_processing_time_ms']:.1f}ms")
    
    print("\n✅ Stream processing demo completed!")
    print("Key Features Demonstrated:")
    print("- Real-time game state tracking")
    print("- Dynamic prediction updates")
    print("- Significant change detection")
    print("- Sub-second processing times")
    print("- Performance monitoring")


if __name__ == "__main__":
    asyncio.run(main())
