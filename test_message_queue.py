#!/usr/bin/env python3
"""
Test Suite for Event-Driven Message Queue System
===============================================

Comprehensive tests for the message queue including:
- Event processing and routing
- Redis integration (with mocking)
- Processor functionality
- Performance and scalability
- Error handling and recovery
"""

import unittest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from event_driven_message_queue import (
    EventQueue, QueueEvent, EventProcessor, EventRouter,
    ScoreUpdateProcessor, InjuryProcessor, PlayProcessor,
    EventPriority, EventStatus
)


class MockRedisClient:
    """Mock Redis client for testing"""
    
    def __init__(self):
        self.streams = {}
        self.consumer_groups = {}
        self.messages = []
        self.connected = True
    
    async def ping(self):
        if not self.connected:
            raise Exception("Connection lost")
        return True
    
    async def xadd(self, stream, fields, maxlen=None):
        if stream not in self.streams:
            self.streams[stream] = []
        
        message_id = f"{int(datetime.now().timestamp() * 1000)}-0"
        self.streams[stream].append((message_id, fields))
        return message_id
    
    async def xgroup_create(self, stream, group, id='0', mkstream=False):
        if stream not in self.consumer_groups:
            self.consumer_groups[stream] = {}
        
        if group in self.consumer_groups[stream]:
            raise Exception("BUSYGROUP Consumer Group name already exists")
        
        self.consumer_groups[stream][group] = {'id': id}
        return True
    
    async def xreadgroup(self, group, consumer, streams, count=1, block=0):
        # Return mock messages
        if self.messages:
            message = self.messages.pop(0)
            return [(list(streams.keys())[0], [message])]
        return []
    
    async def xack(self, stream, group, message_id):
        return 1
    
    async def close(self):
        self.connected = False


class TestQueueEvent(unittest.TestCase):
    """Test QueueEvent data structure"""
    
    def test_queue_event_creation(self):
        """Test QueueEvent creation and serialization"""
        event = QueueEvent(
            event_id="test_123",
            event_type="score",
            source="TestProvider",
            game_id="KC_vs_BAL",
            timestamp=datetime.now(),
            data={'home_score': 14, 'away_score': 7},
            priority=EventPriority.HIGH
        )
        
        self.assertEqual(event.event_id, "test_123")
        self.assertEqual(event.event_type, "score")
        self.assertEqual(event.priority, EventPriority.HIGH)
        self.assertEqual(event.status, EventStatus.PENDING)
    
    def test_event_serialization(self):
        """Test event to/from dictionary conversion"""
        original_event = QueueEvent(
            event_id="test_456",
            event_type="injury",
            source="TestProvider",
            game_id="BUF_vs_MIA",
            timestamp=datetime.now(),
            data={'player': 'Josh Allen', 'severity': 'minor'}
        )
        
        # Convert to dict and back
        event_dict = original_event.to_dict()
        restored_event = QueueEvent.from_dict(event_dict)
        
        self.assertEqual(original_event.event_id, restored_event.event_id)
        self.assertEqual(original_event.event_type, restored_event.event_type)
        self.assertEqual(original_event.data, restored_event.data)
        self.assertEqual(original_event.priority, restored_event.priority)


class TestEventProcessors(unittest.TestCase):
    """Test event processor functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.score_processor = ScoreUpdateProcessor()
        self.injury_processor = InjuryProcessor()
        self.play_processor = PlayProcessor()
    
    async def test_score_processor(self):
        """Test score update processor"""
        event = QueueEvent(
            event_id="score_test",
            event_type="score",
            source="TestProvider",
            game_id="KC_vs_BAL",
            timestamp=datetime.now(),
            data={'home_score': 21, 'away_score': 14}
        )
        
        result = await self.score_processor.process(event)
        
        self.assertTrue(result)
        self.assertEqual(self.score_processor.processed_count, 1)
        self.assertIn("KC_vs_BAL", self.score_processor.game_scores)
        self.assertEqual(self.score_processor.game_scores["KC_vs_BAL"]["home"], 21)
        self.assertEqual(self.score_processor.game_scores["KC_vs_BAL"]["away"], 14)
    
    async def test_injury_processor(self):
        """Test injury processor"""
        event = QueueEvent(
            event_id="injury_test",
            event_type="injury",
            source="TestProvider", 
            game_id="KC_vs_BAL",
            timestamp=datetime.now(),
            data={'player': 'Patrick Mahomes', 'team': 'KC', 'severity': 'questionable'}
        )
        
        result = await self.injury_processor.process(event)
        
        self.assertTrue(result)
        self.assertEqual(self.injury_processor.processed_count, 1)
        self.assertIn("KC_Patrick Mahomes", self.injury_processor.injuries)
        injury_info = self.injury_processor.injuries["KC_Patrick Mahomes"]
        self.assertEqual(injury_info['severity'], 'questionable')
    
    async def test_play_processor(self):
        """Test play processor"""
        event = QueueEvent(
            event_id="play_test",
            event_type="play",
            source="TestProvider",
            game_id="KC_vs_BAL", 
            timestamp=datetime.now(),
            data={'play_type': 'pass', 'yards': 25, 'player': 'Travis Kelce'}
        )
        
        result = await self.play_processor.process(event)
        
        self.assertTrue(result)
        self.assertEqual(self.play_processor.processed_count, 1)
        self.assertIn("KC_vs_BAL", self.play_processor.play_stats)
        stats = self.play_processor.play_stats["KC_vs_BAL"]
        self.assertEqual(stats['total_plays'], 1)
        self.assertEqual(stats['passing_plays'], 1)
        self.assertEqual(stats['total_yards'], 25)
    
    def test_processor_stats(self):
        """Test processor statistics"""
        stats = self.score_processor.get_stats()
        
        self.assertIn('processor_name', stats)
        self.assertIn('processed_count', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('uptime_seconds', stats)
        self.assertEqual(stats['processor_name'], 'ScoreUpdateProcessor')


class TestEventQueue(unittest.TestCase):
    """Test EventQueue functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_redis = MockRedisClient()
        self.event_queue = EventQueue()
        
    @patch('redis.asyncio.from_url')
    async def test_queue_connection(self, mock_redis_from_url):
        """Test Redis connection"""
        mock_redis_from_url.return_value = self.mock_redis
        
        connected = await self.event_queue.connect()
        
        self.assertTrue(connected)
        self.assertIsNotNone(self.event_queue.redis_client)
    
    @patch('redis.asyncio.from_url')
    async def test_event_publishing(self, mock_redis_from_url):
        """Test event publishing to queue"""
        mock_redis_from_url.return_value = self.mock_redis
        await self.event_queue.connect()
        
        event = QueueEvent(
            event_id="pub_test",
            event_type="test_event",
            source="TestProvider",
            game_id="test_game",
            timestamp=datetime.now(),
            data={'test': 'data'}
        )
        
        result = await self.event_queue.publish_event(event)
        
        self.assertTrue(result)
        self.assertEqual(self.event_queue.stats['total_events'], 1)
    
    def test_processor_registration(self):
        """Test processor registration"""
        processor = ScoreUpdateProcessor()
        
        self.event_queue.add_processor('score', processor)
        
        self.assertIn('score', self.event_queue.processors)
        self.assertEqual(self.event_queue.processors['score'], processor)
    
    def test_queue_stats(self):
        """Test queue statistics"""
        stats = self.event_queue.get_stats()
        
        self.assertIn('queue_stats', stats)
        self.assertIn('processor_stats', stats)
        self.assertIn('total_events', stats['queue_stats'])
        self.assertIn('success_rate', stats['queue_stats'])


class TestEventRouter(unittest.TestCase):
    """Test EventRouter functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_queue = Mock()
        self.mock_queue.publish_event = AsyncMock(return_value=True)
        self.router = EventRouter(self.mock_queue)
    
    async def test_websocket_event_routing(self):
        """Test routing WebSocket events to queue"""
        # Mock WebSocket event
        mock_websocket_event = Mock()
        mock_websocket_event.event_id = "ws_test_123"
        mock_websocket_event.event_type = "score"
        mock_websocket_event.source = "ESPN"
        mock_websocket_event.game_id = "KC_vs_BAL"
        mock_websocket_event.timestamp = datetime.now()
        mock_websocket_event.data = {'home_score': 14}
        
        result = await self.router.route_websocket_event(mock_websocket_event)
        
        self.assertTrue(result)
        self.mock_queue.publish_event.assert_called_once()
        self.assertEqual(self.router.routing_stats['total_routed'], 1)
    
    def test_priority_determination(self):
        """Test event priority determination"""
        # Test high priority events
        high_priority = self.router._determine_priority('score')
        self.assertEqual(high_priority, EventPriority.HIGH)
        
        # Test medium priority events
        medium_priority = self.router._determine_priority('play')
        self.assertEqual(medium_priority, EventPriority.MEDIUM)
        
        # Test low priority events
        low_priority = self.router._determine_priority('timeout')
        self.assertEqual(low_priority, EventPriority.LOW)
    
    def test_routing_stats(self):
        """Test routing statistics"""
        stats = self.router.get_routing_stats()
        
        self.assertIn('total_routed', stats)
        self.assertIn('routing_errors', stats)
        self.assertIsInstance(stats['total_routed'], int)
        self.assertIsInstance(stats['routing_errors'], int)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete message queue system"""
    
    @patch('redis.asyncio.from_url')
    async def test_end_to_end_processing(self, mock_redis_from_url):
        """Test complete end-to-end event processing"""
        mock_redis = MockRedisClient()
        mock_redis_from_url.return_value = mock_redis
        
        # Create queue and connect
        event_queue = EventQueue()
        await event_queue.connect()
        
        # Add processors
        score_processor = ScoreUpdateProcessor()
        event_queue.add_processor('score', score_processor)
        
        # Create test event
        event = QueueEvent(
            event_id="integration_test",
            event_type="score",
            source="TestProvider",
            game_id="integration_game",
            timestamp=datetime.now(),
            data={'home_score': 28, 'away_score': 21}
        )
        
        # Publish event
        await event_queue.publish_event(event)
        
        # Simulate message processing
        mock_redis.messages = [("test_id", event.to_dict())]
        
        # Process one cycle
        consumer_task = asyncio.create_task(event_queue.process_events("test_consumer"))
        await asyncio.sleep(0.1)  # Let it process
        event_queue.stop_processing()
        consumer_task.cancel()
        
        # Verify processing
        self.assertEqual(score_processor.processed_count, 1)
        self.assertIn("integration_game", score_processor.game_scores)


class TestPerformanceAndScalability(unittest.TestCase):
    """Performance and scalability tests"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.event_queue = EventQueue()
        self.router = EventRouter(self.event_queue)
    
    async def test_high_volume_event_creation(self):
        """Test creating large numbers of events"""
        import time
        
        start_time = time.time()
        events = []
        
        for i in range(1000):
            event = QueueEvent(
                event_id=f"perf_test_{i}",
                event_type="play",
                source="PerfTest",
                game_id=f"game_{i % 10}",
                timestamp=datetime.now(),
                data={'play_id': i, 'yards': i % 20}
            )
            events.append(event)
        
        end_time = time.time()
        creation_time = end_time - start_time
        
        self.assertEqual(len(events), 1000)
        self.assertLess(creation_time, 1.0)  # Should create 1000 events in < 1 second
        
        events_per_second = 1000 / creation_time
        self.assertGreater(events_per_second, 500)  # At least 500 events/sec
    
    async def test_processor_performance(self):
        """Test processor performance with multiple events"""
        processor = PlayProcessor()
        
        # Create test events
        events = []
        for i in range(100):
            event = QueueEvent(
                event_id=f"proc_test_{i}",
                event_type="play",
                source="PerfTest",
                game_id="perf_game",
                timestamp=datetime.now(),
                data={'play_type': 'run', 'yards': i % 10}
            )
            events.append(event)
        
        # Process all events
        import time
        start_time = time.time()
        
        for event in events:
            await processor.process(event)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        self.assertEqual(processor.processed_count, 100)
        self.assertLess(processing_time, 0.5)  # Should process 100 events in < 0.5s
        
        stats = processor.get_stats()
        self.assertGreater(stats['events_per_second'], 100)


def run_message_queue_tests():
    """Run all message queue tests"""
    print("🔄 MESSAGE QUEUE TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQueueEvent))
    suite.addTests(loader.loadTestsFromTestCase(TestEventProcessors))
    suite.addTests(loader.loadTestsFromTestCase(TestEventQueue))
    suite.addTests(loader.loadTestsFromTestCase(TestEventRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceAndScalability))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MESSAGE QUEUE TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                   result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSUCCESS RATE: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_message_queue_tests()
    
    if success:
        print("\n✅ ALL MESSAGE QUEUE TESTS PASSED!")
        print("Event-driven message queue system is ready for production.")
    else:
        print("\n❌ SOME MESSAGE QUEUE TESTS FAILED!")
        print("Please review and fix issues before proceeding.")
    
    exit(0 if success else 1)
