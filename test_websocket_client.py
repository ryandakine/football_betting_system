#!/usr/bin/env python3
"""
Test Suite for Real-Time WebSocket Client
=========================================

Comprehensive tests for the WebSocket client including:
- Connection management and reconnection
- Message handling and validation
- Error handling and recovery
- Performance and latency testing
- Multi-provider coordination
"""

import unittest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_websocket_client import (
    WebSocketGameClient, MultiProviderWebSocketManager,
    GameEvent, ConnectionState, ConnectionStats
)


class MockWebSocket:
    """Mock WebSocket for testing"""
    
    def __init__(self, messages=None, should_fail=False):
        self.messages = messages or []
        self.should_fail = should_fail
        self.sent_messages = []
        self.closed = False
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def __aiter__(self):
        for message in self.messages:
            if self.should_fail:
                raise Exception("Mock WebSocket failure")
            yield message
    
    async def send(self, message):
        self.sent_messages.append(message)
    
    async def close(self):
        self.closed = True


class TestWebSocketGameClient(unittest.TestCase):
    """Test cases for WebSocketGameClient"""
    
    def setUp(self):
        """Set up test environment"""
        self.client = WebSocketGameClient(
            provider_name="TestProvider",
            websocket_url="wss://test.example.com/ws",
            api_key="test_api_key",
            reconnect_delay=0.1,
            max_reconnect_delay=1.0,
            reconnect_attempts=3
        )
        
        # Track callback calls
        self.events_received = []
        self.connection_states = []
        self.errors_received = []
        
        self.client.add_event_callback(self.events_received.append)
        self.client.add_connection_callback(self.connection_states.append)
        self.client.add_error_callback(self.errors_received.append)
    
    def test_initialization(self):
        """Test client initialization"""
        self.assertEqual(self.client.provider_name, "TestProvider")
        self.assertEqual(self.client.websocket_url, "wss://test.example.com/ws")
        self.assertEqual(self.client.api_key, "test_api_key")
        self.assertEqual(self.client.state, ConnectionState.DISCONNECTED)
        self.assertIsInstance(self.client.stats, ConnectionStats)
    
    def test_message_validation(self):
        """Test message validation logic"""
        # Valid message
        valid_message = {
            'type': 'score',
            'timestamp': '2024-09-19T12:00:00Z',
            'game_id': 'test_game_1',
            'data': {'home_score': 14, 'away_score': 7}
        }
        self.assertTrue(self.client._validate_message(valid_message))
        
        # Missing required field
        invalid_message = {
            'timestamp': '2024-09-19T12:00:00Z',
            'game_id': 'test_game_1'
        }
        self.assertFalse(self.client._validate_message(invalid_message))
        
        # Invalid timestamp
        invalid_timestamp = {
            'type': 'score',
            'timestamp': 'invalid_timestamp',
            'game_id': 'test_game_1'
        }
        self.assertFalse(self.client._validate_message(invalid_timestamp))
    
    def test_game_event_parsing(self):
        """Test parsing of WebSocket messages into GameEvent objects"""
        message = {
            'id': 'event_123',
            'type': 'touchdown',
            'timestamp': '2024-09-19T12:00:00Z',
            'game_id': 'nfl_game_456',
            'data': {
                'team': 'KC',
                'player': 'Patrick Mahomes',
                'yards': 15
            },
            'confidence': 0.95
        }
        
        event = self.client._parse_game_event(message)
        
        self.assertIsInstance(event, GameEvent)
        self.assertEqual(event.event_id, 'event_123')
        self.assertEqual(event.event_type, 'touchdown')
        self.assertEqual(event.game_id, 'nfl_game_456')
        self.assertEqual(event.source, 'TestProvider')
        self.assertEqual(event.confidence, 0.95)
        self.assertIn('team', event.data)
    
    async def test_message_handling(self):
        """Test message handling and callback triggering"""
        test_message = json.dumps({
            'type': 'score_update',
            'timestamp': datetime.now().isoformat(),
            'game_id': 'test_game',
            'data': {'home_score': 21, 'away_score': 14}
        })
        
        await self.client._handle_message(test_message)
        
        # Check that event was processed
        self.assertEqual(len(self.events_received), 1)
        event = self.events_received[0]
        self.assertEqual(event.event_type, 'score_update')
        self.assertEqual(event.game_id, 'test_game')
        
        # Check stats were updated
        self.assertEqual(self.client.stats.total_messages, 1)
        self.assertEqual(self.client.stats.failed_messages, 0)
        self.assertIsNotNone(self.client.stats.last_message_time)
    
    async def test_invalid_message_handling(self):
        """Test handling of invalid messages"""
        # Invalid JSON
        await self.client._handle_message("invalid json")
        
        # Missing required fields
        invalid_message = json.dumps({'data': 'incomplete'})
        await self.client._handle_message(invalid_message)
        
        # Check that failed messages were tracked
        self.assertEqual(self.client.stats.failed_messages, 2)
        self.assertEqual(len(self.events_received), 0)
    
    def test_connection_headers(self):
        """Test WebSocket connection header generation"""
        async def run_test():
            headers = await self.client._create_connection_headers()
            
            self.assertIn('User-Agent', headers)
            self.assertIn('Authorization', headers)
            self.assertEqual(headers['Authorization'], 'Bearer test_api_key')
        
        asyncio.run(run_test())
    
    def test_stats_calculation(self):
        """Test connection statistics calculations"""
        stats = ConnectionStats()
        stats.connected_since = datetime.now() - timedelta(seconds=60)
        stats.total_messages = 100
        stats.failed_messages = 5
        
        self.assertEqual(stats.success_rate, 0.95)
        self.assertGreater(stats.uptime_seconds, 50)
    
    @patch('websockets.connect')
    async def test_connection_success(self, mock_connect):
        """Test successful WebSocket connection"""
        mock_websocket = AsyncMock()
        mock_connect.return_value = mock_websocket
        
        result = await self.client._connect()
        
        self.assertTrue(result)
        self.assertEqual(self.client.state, ConnectionState.CONNECTED)
        self.assertIsNotNone(self.client.stats.connected_since)
        
        # Check connection state callbacks
        self.assertIn(ConnectionState.CONNECTING, self.connection_states)
        self.assertIn(ConnectionState.CONNECTED, self.connection_states)
    
    @patch('websockets.connect')
    async def test_connection_failure(self, mock_connect):
        """Test WebSocket connection failure"""
        mock_connect.side_effect = Exception("Connection failed")
        
        result = await self.client._connect()
        
        self.assertFalse(result)
        self.assertEqual(self.client.state, ConnectionState.FAILED)
        self.assertEqual(len(self.errors_received), 1)
    
    async def test_send_message(self):
        """Test sending messages through WebSocket"""
        # Mock connected state
        self.client.state = ConnectionState.CONNECTED
        self.client.websocket = AsyncMock()
        
        test_message = {'type': 'subscribe', 'game_id': 'test_game'}
        await self.client.send_message(test_message)
        
        self.client.websocket.send.assert_called_once()
        sent_data = json.loads(self.client.websocket.send.call_args[0][0])
        self.assertEqual(sent_data['type'], 'subscribe')
    
    async def test_disconnect(self):
        """Test graceful disconnection"""
        self.client.websocket = AsyncMock()
        self.client.state = ConnectionState.CONNECTED
        
        await self.client.disconnect()
        
        self.client.websocket.close.assert_called_once()
        self.assertEqual(self.client.state, ConnectionState.DISCONNECTED)


class TestMultiProviderWebSocketManager(unittest.TestCase):
    """Test cases for MultiProviderWebSocketManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.manager = MultiProviderWebSocketManager()
        self.events_received = []
        
        self.manager.add_event_handler(self.events_received.append)
    
    def test_add_provider(self):
        """Test adding WebSocket providers"""
        client = self.manager.add_provider(
            "TestProvider1",
            "wss://test1.example.com/ws",
            api_key="key1"
        )
        
        self.assertIn("TestProvider1", self.manager.clients)
        self.assertIsInstance(client, WebSocketGameClient)
        self.assertEqual(client.provider_name, "TestProvider1")
        self.assertEqual(client.api_key, "key1")
    
    def test_event_forwarding(self):
        """Test that events from providers are forwarded to global handlers"""
        # Add a provider
        client = self.manager.add_provider(
            "TestProvider",
            "wss://test.example.com/ws"
        )
        
        # Simulate an event from the client
        test_event = GameEvent(
            event_id="test_123",
            game_id="game_456",
            timestamp=datetime.now(),
            event_type="test_event",
            data={"test": "data"},
            source="TestProvider"
        )
        
        self.manager._handle_event(test_event)
        
        # Check that the event was forwarded
        self.assertEqual(len(self.events_received), 1)
        self.assertEqual(self.events_received[0].event_id, "test_123")
    
    def test_multiple_providers(self):
        """Test managing multiple providers"""
        # Add multiple providers
        self.manager.add_provider("Provider1", "wss://provider1.com/ws")
        self.manager.add_provider("Provider2", "wss://provider2.com/ws")
        self.manager.add_provider("Provider3", "wss://provider3.com/ws")
        
        self.assertEqual(len(self.manager.clients), 3)
        
        # Test getting stats from all providers
        all_stats = self.manager.get_all_stats()
        self.assertEqual(len(all_stats), 3)
        self.assertIn("Provider1", all_stats)
        self.assertIn("Provider2", all_stats)
        self.assertIn("Provider3", all_stats)


class TestPerformanceAndLoad(unittest.TestCase):
    """Performance and load testing"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.client = WebSocketGameClient(
            provider_name="PerfTestProvider",
            websocket_url="wss://test.example.com/ws"
        )
        
        self.events_processed = []
        self.client.add_event_callback(self.events_processed.append)
    
    async def test_message_processing_performance(self):
        """Test message processing performance"""
        # Generate test messages
        test_messages = []
        for i in range(1000):
            message = json.dumps({
                'id': f'event_{i}',
                'type': 'performance_test',
                'timestamp': datetime.now().isoformat(),
                'game_id': f'game_{i % 10}',
                'data': {'value': i, 'test_data': 'x' * 100}
            })
            test_messages.append(message)
        
        # Process messages and measure time
        start_time = time.time()
        
        for message in test_messages:
            await self.client._handle_message(message)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        self.assertEqual(len(self.events_processed), 1000)
        self.assertLess(processing_time, 1.0)  # Should process 1000 messages in < 1 second
        
        messages_per_second = 1000 / processing_time
        self.assertGreater(messages_per_second, 500)  # At least 500 messages/sec
        
        print(f"Performance: {messages_per_second:.0f} messages/second")
        print(f"Average latency: {self.client.stats.average_latency * 1000:.2f}ms")
    
    async def test_concurrent_message_handling(self):
        """Test handling multiple concurrent messages"""
        # Create multiple clients
        clients = []
        all_events = []
        
        for i in range(5):
            client = WebSocketGameClient(
                provider_name=f"ConcurrentProvider{i}",
                websocket_url=f"wss://test{i}.example.com/ws"
            )
            client.add_event_callback(all_events.append)
            clients.append(client)
        
        # Process messages concurrently
        async def process_messages(client, message_count):
            for j in range(message_count):
                message = json.dumps({
                    'type': 'concurrent_test',
                    'timestamp': datetime.now().isoformat(),
                    'game_id': f'{client.provider_name}_game',
                    'data': {'message_id': j}
                })
                await client._handle_message(message)
        
        # Run concurrent processing
        start_time = time.time()
        tasks = [process_messages(client, 200) for client in clients]
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify results
        self.assertEqual(len(all_events), 1000)  # 5 clients × 200 messages
        self.assertLess(end_time - start_time, 2.0)  # Should complete in < 2 seconds


class TestIntegrationScenarios(unittest.TestCase):
    """Integration testing with realistic scenarios"""
    
    async def test_live_game_simulation(self):
        """Simulate a live game with realistic event flow"""
        manager = MultiProviderWebSocketManager()
        
        # Track different event types
        events_by_type = {}
        
        def categorize_events(event):
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = []
            events_by_type[event.event_type].append(event)
        
        manager.add_event_handler(categorize_events)
        
        # Add mock provider
        client = manager.add_provider("MockGameProvider", "wss://mock.example.com/ws")
        
        # Simulate game events
        game_events = [
            ('kickoff', {'quarter': 1, 'time': '15:00'}),
            ('play', {'type': 'run', 'yards': 5, 'down': 1}),
            ('play', {'type': 'pass', 'yards': 12, 'down': 2}),
            ('touchdown', {'team': 'KC', 'player': 'Travis Kelce'}),
            ('extra_point', {'successful': True}),
            ('score_update', {'home_score': 7, 'away_score': 0}),
            ('injury', {'player': 'Player X', 'severity': 'minor'}),
            ('timeout', {'team': 'BAL', 'type': 'official'}),
            ('quarter_end', {'quarter': 1}),
            ('halftime', {'stats': {'total_yards': {'KC': 245, 'BAL': 178}}})
        ]
        
        # Process events
        for event_type, data in game_events:
            message = json.dumps({
                'type': event_type,
                'timestamp': datetime.now().isoformat(),
                'game_id': 'KC_vs_BAL_2024',
                'data': data
            })
            await client._handle_message(message)
        
        # Verify realistic game flow was processed
        self.assertIn('touchdown', events_by_type)
        self.assertIn('score_update', events_by_type)
        self.assertIn('injury', events_by_type)
        
        # Check that game is tracked
        active_games = client.get_active_games()
        self.assertIn('KC_vs_BAL_2024', active_games)
        self.assertEqual(active_games['KC_vs_BAL_2024']['event_count'], len(game_events))


def run_websocket_tests():
    """Run all WebSocket tests"""
    print("🔌 WEBSOCKET CLIENT TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWebSocketGameClient))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiProviderWebSocketManager))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceAndLoad))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("WEBSOCKET TEST SUMMARY")
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
    success = run_websocket_tests()
    
    if success:
        print("\n✅ ALL WEBSOCKET TESTS PASSED!")
        print("WebSocket client is ready for production use.")
    else:
        print("\n❌ SOME WEBSOCKET TESTS FAILED!")
        print("Please review and fix issues before proceeding.")
    
    exit(0 if success else 1)
