#!/usr/bin/env python3
"""
Event-Driven Message Queue System for NFL Real-Time Intelligence Engine
======================================================================

Production-ready message queue system with:
- Redis Streams for high-performance pub/sub
- Event routing and processing
- Fault tolerance and recovery
- Integration with WebSocket client
- Scalable architecture

Uses Redis instead of RabbitMQ/Kafka for simplicity while maintaining enterprise features.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
import uuid
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class EventStatus(Enum):
    """Event processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class QueueEvent:
    """Standardized queue event structure"""
    event_id: str
    event_type: str
    source: str
    game_id: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.MEDIUM
    retry_count: int = 0
    max_retries: int = 3
    status: EventStatus = EventStatus.PENDING
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'source': self.source,
            'game_id': self.game_id,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'priority': self.priority.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'status': self.status.value,
            'processing_time': self.processing_time,
            'error_message': self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueEvent':
        """Create from dictionary"""
        return cls(
            event_id=data['event_id'],
            event_type=data['event_type'],
            source=data['source'],
            game_id=data['game_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data['data'],
            priority=EventPriority(data.get('priority', EventPriority.MEDIUM.value)),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            status=EventStatus(data.get('status', EventStatus.PENDING.value)),
            processing_time=data.get('processing_time'),
            error_message=data.get('error_message')
        )


class EventProcessor:
    """Base class for event processors"""
    
    def __init__(self, processor_name: str):
        self.processor_name = processor_name
        self.processed_count = 0
        self.error_count = 0
        self.start_time = datetime.now()
    
    async def process(self, event: QueueEvent) -> bool:
        """Process an event. Return True for success, False for failure."""
        raise NotImplementedError("Subclasses must implement process method")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            'processor_name': self.processor_name,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'success_rate': (self.processed_count - self.error_count) / max(self.processed_count, 1),
            'uptime_seconds': uptime,
            'events_per_second': self.processed_count / max(uptime, 1)
        }


class ScoreUpdateProcessor(EventProcessor):
    """Processor for score update events"""
    
    def __init__(self):
        super().__init__("ScoreUpdateProcessor")
        self.game_scores = {}
    
    async def process(self, event: QueueEvent) -> bool:
        """Process score update events"""
        try:
            game_id = event.game_id
            score_data = event.data
            
            # Update game scores
            if game_id not in self.game_scores:
                self.game_scores[game_id] = {'home': 0, 'away': 0}
            
            if 'home_score' in score_data:
                self.game_scores[game_id]['home'] = score_data['home_score']
            if 'away_score' in score_data:
                self.game_scores[game_id]['away'] = score_data['away_score']
            
            logger.info(f"Score Update: {game_id} - {self.game_scores[game_id]}")
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error processing score update: {e}")
            self.error_count += 1
            return False


class InjuryProcessor(EventProcessor):
    """Processor for injury events"""
    
    def __init__(self):
        super().__init__("InjuryProcessor")
        self.injuries = {}
    
    async def process(self, event: QueueEvent) -> bool:
        """Process injury events"""
        try:
            injury_data = event.data
            player = injury_data.get('player', 'Unknown')
            team = injury_data.get('team', 'Unknown')
            severity = injury_data.get('severity', 'Unknown')
            
            injury_key = f"{team}_{player}"
            self.injuries[injury_key] = {
                'player': player,
                'team': team,
                'severity': severity,
                'timestamp': event.timestamp,
                'game_id': event.game_id
            }
            
            logger.warning(f"🚑 INJURY: {player} ({team}) - {severity} severity in {event.game_id}")
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error processing injury: {e}")
            self.error_count += 1
            return False


class PlayProcessor(EventProcessor):
    """Processor for play events"""
    
    def __init__(self):
        super().__init__("PlayProcessor")
        self.play_stats = {}
    
    async def process(self, event: QueueEvent) -> bool:
        """Process play events"""
        try:
            play_data = event.data
            game_id = event.game_id
            
            if game_id not in self.play_stats:
                self.play_stats[game_id] = {
                    'total_plays': 0,
                    'passing_plays': 0,
                    'rushing_plays': 0,
                    'total_yards': 0
                }
            
            stats = self.play_stats[game_id]
            stats['total_plays'] += 1
            
            play_type = play_data.get('play_type', 'unknown')
            yards = play_data.get('yards', 0)
            
            if play_type == 'pass':
                stats['passing_plays'] += 1
            elif play_type == 'run':
                stats['rushing_plays'] += 1
            
            stats['total_yards'] += yards
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            logger.error(f"Error processing play: {e}")
            self.error_count += 1
            return False


class EventQueue:
    """Redis-based event queue with pub/sub capabilities"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_name: str = "nfl_events",
        consumer_group: str = "intelligence_engine",
        max_retries: int = 3
    ):
        self.redis_url = redis_url
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.max_retries = max_retries
        self.redis_client = None
        self.is_running = False
        
        # Event processors
        self.processors: Dict[str, EventProcessor] = {}
        self.default_processor = None
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'processed_events': 0,
            'failed_events': 0,
            'start_time': None
        }
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            
            # Test connection
            await self.redis_client.ping()
            logger.info(f"Connected to Redis: {self.redis_url}")
            
            # Create consumer group
            try:
                await self.redis_client.xgroup_create(
                    self.stream_name, 
                    self.consumer_group, 
                    id='0', 
                    mkstream=True
                )
                logger.info(f"Created consumer group: {self.consumer_group}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    logger.info(f"Consumer group {self.consumer_group} already exists")
                else:
                    raise
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")
    
    def add_processor(self, event_type: str, processor: EventProcessor):
        """Add event processor for specific event type"""
        self.processors[event_type] = processor
        logger.info(f"Added processor for {event_type}: {processor.processor_name}")
    
    def set_default_processor(self, processor: EventProcessor):
        """Set default processor for unhandled event types"""
        self.default_processor = processor
        logger.info(f"Set default processor: {processor.processor_name}")
    
    async def publish_event(self, event: QueueEvent) -> bool:
        """Publish event to the queue"""
        try:
            if not self.redis_client:
                logger.error("Redis client not connected")
                return False
            
            # Add to stream
            event_data = event.to_dict()
            message_id = await self.redis_client.xadd(
                self.stream_name,
                event_data,
                maxlen=10000  # Keep last 10k events
            )
            
            self.stats['total_events'] += 1
            logger.debug(f"Published event {event.event_id} with message ID {message_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return False
    
    async def process_events(self, consumer_name: str = None):
        """Process events from the queue"""
        if not consumer_name:
            consumer_name = f"consumer_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting event processing with consumer: {consumer_name}")
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        while self.is_running:
            try:
                # Read from stream
                messages = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    consumer_name,
                    {self.stream_name: '>'},
                    count=10,
                    block=1000  # 1 second timeout
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        await self._process_message(message_id, fields, consumer_name)
                
            except redis.ConnectionError:
                logger.error("Lost Redis connection, attempting to reconnect...")
                await asyncio.sleep(5)
                await self.connect()
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message_id: str, fields: Dict, consumer_name: str):
        """Process a single message"""
        async with _processing_lock:  # Fix race conditions
            try:
                # Convert Redis fields to QueueEvent
                event = QueueEvent.from_dict(fields)
                
                start_time = time.time()
                success = False
                
                # Find appropriate processor
                processor = self.processors.get(event.event_type, self.default_processor)
                
                if processor:
                    success = await processor.process(event)
                    processing_time = time.time() - start_time
                    
                    if success:
                        self.stats['processed_events'] += 1
                        logger.debug(f"Processed {event.event_type} event in {processing_time:.3f}s")
                    else:
                        self.stats['failed_events'] += 1
                        logger.error(f"Failed to process {event.event_type} event")
                else:
                    logger.warning(f"No processor found for event type: {event.event_type}")
                
                # Acknowledge message
                await self.redis_client.xack(self.stream_name, self.consumer_group, message_id)
                
            except Exception as e:
                logger.error(f"Error processing message {message_id}: {e}")
                self.stats['failed_events'] += 1
    
    def stop_processing(self):
        """Stop event processing"""
        self.is_running = False
        logger.info("Stopping event processing")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        uptime = 0
        if self.stats['start_time']:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        queue_stats = {
            'total_events': self.stats['total_events'],
            'processed_events': self.stats['processed_events'],
            'failed_events': self.stats['failed_events'],
            'success_rate': self.stats['processed_events'] / max(self.stats['total_events'], 1),
            'uptime_seconds': uptime,
            'events_per_second': self.stats['processed_events'] / max(uptime, 1)
        }
        
        # Add processor stats
        processor_stats = {}
        for event_type, processor in self.processors.items():
            processor_stats[event_type] = processor.get_stats()
        
        return {
            'queue_stats': queue_stats,
            'processor_stats': processor_stats
        }


class EventRouter:
    """Routes events from WebSocket to message queue"""
    
    def __init__(self, event_queue: EventQueue):
        self.event_queue = event_queue
        self.routing_stats = {
            'total_routed': 0,
            'routing_errors': 0
        }
    
    async def route_websocket_event(self, websocket_event) -> bool:
        """Convert WebSocket event to queue event and publish"""
        try:
            # Convert WebSocket GameEvent to QueueEvent
            queue_event = QueueEvent(
                event_id=websocket_event.event_id,
                event_type=websocket_event.event_type,
                source=websocket_event.source,
                game_id=websocket_event.game_id,
                timestamp=websocket_event.timestamp,
                data=websocket_event.data,
                priority=self._determine_priority(websocket_event.event_type)
            )
            
            # Publish to queue
            success = await self.event_queue.publish_event(queue_event)
            
            if success:
                self.routing_stats['total_routed'] += 1
                logger.debug(f"Routed {websocket_event.event_type} event to queue")
            else:
                self.routing_stats['routing_errors'] += 1
                logger.error(f"Failed to route {websocket_event.event_type} event")
            
            return success
            
        except Exception as e:
            logger.error(f"Error routing WebSocket event: {e}")
            self.routing_stats['routing_errors'] += 1
            return False
    
    def _determine_priority(self, event_type: str) -> EventPriority:
        """Determine event priority based on type"""
        high_priority_events = ['score', 'touchdown', 'injury']
        critical_events = ['game_end', 'emergency']
        
        if event_type in critical_events:
            return EventPriority.CRITICAL
        elif event_type in high_priority_events:
            return EventPriority.HIGH
        elif event_type in ['play', 'penalty']:
            return EventPriority.MEDIUM
        else:
            return EventPriority.LOW
    
    def get_routing_stats(self) -> Dict[str, int]:
        """Get routing statistics"""
        return self.routing_stats.copy()


# Integration with WebSocket client
async def create_integrated_system():
    """Create integrated WebSocket + Message Queue system"""
    
    # Import WebSocket components
    try:
        from realtime_websocket_client import MultiProviderWebSocketManager, GameEvent
    except ImportError:
        logger.error("WebSocket client not available")
        return None
    
    # Create message queue
    event_queue = EventQueue()
    await event_queue.connect()
    
    # Add processors
    event_queue.add_processor('score', ScoreUpdateProcessor())
    event_queue.add_processor('injury', InjuryProcessor())
    event_queue.add_processor('play', PlayProcessor())
    
    # Create router
    router = EventRouter(event_queue)
    
    # Create WebSocket manager
    ws_manager = MultiProviderWebSocketManager()
    
    # Connect WebSocket events to message queue
    async def websocket_to_queue_handler(event: GameEvent):
        await router.route_websocket_event(event)
    
    ws_manager.add_event_handler(websocket_to_queue_handler)
    
    return {
        'websocket_manager': ws_manager,
        'event_queue': event_queue,
        'router': router
    }


async def main():
    """Demo of event-driven message queue system"""
    print("🔄 NFL EVENT-DRIVEN MESSAGE QUEUE DEMO")
    print("=" * 50)
    
    # Create event queue
    event_queue = EventQueue()
    
    # Try to connect to Redis (will fail if Redis not running, but that's OK for demo)
    try:
        connected = await event_queue.connect()
        if not connected:
            print("⚠️ Redis not available - running in demo mode")
            return
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        print("Install Redis with: sudo apt-get install redis-server")
        return
    
    # Add processors
    event_queue.add_processor('score', ScoreUpdateProcessor())
    event_queue.add_processor('injury', InjuryProcessor())
    event_queue.add_processor('play', PlayProcessor())
    
    # Create some test events
    test_events = [
        QueueEvent(
            event_id="test_001",
            event_type="score",
            source="TestProvider",
            game_id="KC_vs_BAL",
            timestamp=datetime.now(),
            data={'home_score': 7, 'away_score': 0, 'score_type': 'touchdown'}
        ),
        QueueEvent(
            event_id="test_002", 
            event_type="injury",
            source="TestProvider",
            game_id="KC_vs_BAL",
            timestamp=datetime.now(),
            data={'player': 'Patrick Mahomes', 'team': 'KC', 'severity': 'minor'}
        ),
        QueueEvent(
            event_id="test_003",
            event_type="play",
            source="TestProvider", 
            game_id="KC_vs_BAL",
            timestamp=datetime.now(),
            data={'play_type': 'pass', 'yards': 15, 'player': 'Travis Kelce'}
        )
    ]
    
    # Publish test events
    print("📤 Publishing test events...")
    for event in test_events:
        await event_queue.publish_event(event)
    
    # Process events
    print("⚙️ Processing events...")
    
    # Create a consumer task
    consumer_task = asyncio.create_task(event_queue.process_events("demo_consumer"))
    
    # Let it run for a few seconds
    await asyncio.sleep(3)
    
    # Stop processing
    event_queue.stop_processing()
    consumer_task.cancel()
    
    # Show stats
    stats = event_queue.get_stats()
    print("\n📊 PROCESSING STATISTICS:")
    print(f"Total Events: {stats['queue_stats']['total_events']}")
    print(f"Processed: {stats['queue_stats']['processed_events']}")
    print(f"Success Rate: {stats['queue_stats']['success_rate']:.1%}")
    
    print("\nProcessor Details:")
    for event_type, proc_stats in stats['processor_stats'].items():
        print(f"  {event_type}: {proc_stats['processed_count']} processed, "
              f"{proc_stats['success_rate']:.1%} success rate")
    
    # Cleanup
    await event_queue.disconnect()
    
    print("\n✅ Event-driven message queue demo completed!")
    print("Key Features Demonstrated:")
    print("- Redis Streams for high-performance messaging")
    print("- Event processors with specialized handling")
    print("- Fault tolerance and error handling")
    print("- Real-time statistics and monitoring")
    print("- Integration-ready architecture")


if __name__ == "__main__":
    asyncio.run(main())
