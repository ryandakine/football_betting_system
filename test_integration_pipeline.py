#!/usr/bin/env python3
"""
NFL Data Ingestion Pipeline Integration Tests
============================================

Comprehensive integration tests for the complete NFL data ingestion system.
Tests the coordination between:
- nfl_live_data_fetcher.py
- advanced_data_ingestion.py  
- data_collection.py
"""

import asyncio
import unittest
import tempfile
import os
import sqlite3
import pandas as pd
from datetime import datetime
from unittest.mock import patch, Mock
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nfl_live_data_fetcher import NFLLiveDataFetcher
from advanced_data_ingestion import AdvancedDataIngestion
from data_collection import NFLDataCollector, run_nfl_data_collection


class TestDataIngestionPipeline(unittest.TestCase):
    """Test the complete data ingestion pipeline integration"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = os.path.join(self.temp_dir, "integration_test.db")
        
    def tearDown(self):
        """Clean up integration test environment"""
        # Clean up any created files
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)
    
    async def test_live_data_fetcher_integration(self):
        """Test NFL live data fetcher with real API structure"""
        async with NFLLiveDataFetcher() as fetcher:
            # Test with mock data that matches real ESPN API structure
            with patch.object(fetcher, '_make_request') as mock_request:
                mock_request.return_value = {
                    'events': [
                        {
                            'id': '401671745',
                            'date': '2024-09-05T20:20Z',
                            'competitions': [{
                                'competitors': [
                                    {
                                        'team': {'displayName': 'Kansas City Chiefs'},
                                        'homeAway': 'home',
                                        'score': '27'
                                    },
                                    {
                                        'team': {'displayName': 'Baltimore Ravens'},
                                        'homeAway': 'away',
                                        'score': '20'
                                    }
                                ],
                                'venue': {'fullName': 'GEHA Field at Arrowhead Stadium'}
                            }],
                            'status': {
                                'type': {
                                    'state': 'post',
                                    'completed': True,
                                    'shortDetail': 'Final'
                                }
                            }
                        }
                    ]
                }
                
                # Test live games fetch
                games = await fetcher.get_live_games()
                
                # Verify structure
                self.assertIsInstance(games, list)
                if games:
                    game = games[0]
                    required_fields = [
                        'id', 'home_team', 'away_team', 'home_score', 
                        'away_score', 'status', 'stadium', 'data_source'
                    ]
                    for field in required_fields:
                        self.assertIn(field, game)
                    
                    # Verify data quality
                    self.assertEqual(game['data_source'], 'ESPN_API')
                    self.assertIsInstance(game['home_score'], int)
                    self.assertIsInstance(game['away_score'], int)
    
    def test_advanced_data_ingestion_integration(self):
        """Test advanced data ingestion system"""
        # Create advanced data ingestion instance
        ingestion = AdvancedDataIngestion(
            cache_dir=os.path.join(self.temp_dir, "cache"),
            db_path=self.temp_db
        )
        
        # Verify database initialization
        self.assertTrue(os.path.exists(self.temp_db))
        
        # Check database schema
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['player_stats', 'team_stats', 'games']
            for table in expected_tables:
                self.assertIn(table, tables)
    
    def test_data_collector_integration(self):
        """Test NFL data collector integration"""
        collector = NFLDataCollector(db_path=self.temp_db)
        
        # Test with mock ESPN API response
        with patch.object(collector.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                'events': [
                    {
                        'id': '401671745',
                        'date': '2024-09-05T20:20Z',
                        'competitions': [{
                            'competitors': [
                                {
                                    'team': {'abbreviation': 'KC'},
                                    'homeAway': 'home',
                                    'score': '27'
                                },
                                {
                                    'team': {'abbreviation': 'BAL'},
                                    'homeAway': 'away',
                                    'score': '20'
                                }
                            ],
                            'venue': {'fullName': 'GEHA Field at Arrowhead Stadium'}
                        }],
                        'status': {'type': {'name': 'Final'}}
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            # Test schedule fetching
            schedule_df = collector.fetch_nfl_schedule(2024, 1)
            
            # Verify results
            self.assertFalse(schedule_df.empty)
            self.assertEqual(len(schedule_df), 1)
            self.assertEqual(schedule_df.iloc[0]['home_team'], 'KC')
            self.assertEqual(schedule_df.iloc[0]['away_team'], 'BAL')
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end data pipeline"""
        
        # Mock all external API calls
        with patch('data_collection.NFLDataCollector.fetch_nfl_schedule') as mock_schedule, \
             patch('data_collection.NFLDataCollector.fetch_team_stats') as mock_team_stats, \
             patch('data_collection.NFLDataCollector.fetch_player_stats') as mock_player_stats:
            
            # Set up mock return values
            mock_schedule.return_value = pd.DataFrame([
                {
                    'game_id': '401671745',
                    'season': 2024,
                    'week': 1,
                    'home_team': 'KC',
                    'away_team': 'BAL',
                    'home_score': 27,
                    'away_score': 20,
                    'venue': 'GEHA Field at Arrowhead Stadium',
                    'status': 'Final',
                    'data_source': 'ESPN_API'
                }
            ])
            
            mock_team_stats.return_value = pd.DataFrame([
                {
                    'Tm': 'Kansas City Chiefs',
                    'G': '17',
                    'W': '14',
                    'L': '3',
                    'PF': '456',
                    'PA': '329',
                    'season': 2024,
                    'data_source': 'PRO_FOOTBALL_REF'
                }
            ])
            
            mock_player_stats.return_value = pd.DataFrame([
                {
                    'Player': 'Patrick Mahomes',
                    'Tm': 'KC',
                    'Att': '597',
                    'Cmp': '401',
                    'Yds': '4183',
                    'TD': '27',
                    'position': 'QB',
                    'season': 2024,
                    'data_source': 'PRO_FOOTBALL_REF'
                }
            ])
            
            # Run the complete data collection pipeline
            collector = NFLDataCollector(db_path=self.temp_db)
            results = collector.collect_all_data(2024, 1)
            
            # Verify all data types were collected
            expected_data_types = [
                'schedule', 'team_stats', 'player_stats', 'schedule_with_weather'
            ]
            for data_type in expected_data_types:
                self.assertIn(data_type, results)
                self.assertFalse(results[data_type].empty)
            
            # Verify data was saved to database
            with sqlite3.connect(self.temp_db) as conn:
                cursor = conn.cursor()
                
                # Check that tables were created and populated
                cursor.execute("SELECT COUNT(*) FROM game_schedule")
                schedule_count = cursor.fetchone()[0]
                self.assertGreater(schedule_count, 0)
                
                cursor.execute("SELECT COUNT(*) FROM team_statistics")
                team_count = cursor.fetchone()[0]
                self.assertGreater(team_count, 0)
                
                cursor.execute("SELECT COUNT(*) FROM player_statistics")
                player_count = cursor.fetchone()[0]
                self.assertGreater(player_count, 0)
    
    def test_data_quality_validation(self):
        """Test data quality validation across the pipeline"""
        
        # Create sample data with quality issues
        test_games = [
            {
                'id': '401671745',
                'home_team': 'Kansas City Chiefs',
                'away_team': 'Baltimore Ravens',
                'home_score': 27,
                'away_score': 20,
                'quarter': 4,
                'time_remaining': '00:00',
                'status': 'Final',
                'stadium': 'GEHA Field at Arrowhead Stadium',
                'data_source': 'ESPN_API'
            },
            {
                'id': '401671746',
                'home_team': 'Buffalo Bills',
                'away_team': 'Miami Dolphins', 
                'home_score': -1,  # Invalid negative score
                'away_score': 21,
                'quarter': 6,  # Invalid quarter
                'time_remaining': '15:00',
                'status': 'in_progress',
                'stadium': 'Highmark Stadium',
                'data_source': 'ESPN_API'
            }
        ]
        
        # Test data validation with live data fetcher
        async def run_validation_test():
            async with NFLLiveDataFetcher() as fetcher:
                validated_games = await fetcher.validate_game_data(test_games)
                
                # Should filter out invalid game
                self.assertEqual(len(validated_games), 1)
                
                # Remaining game should be valid
                valid_game = validated_games[0]
                self.assertEqual(valid_game['id'], '401671745')
                self.assertGreaterEqual(valid_game['home_score'], 0)
                self.assertGreaterEqual(valid_game['away_score'], 0)
                self.assertTrue(1 <= valid_game['quarter'] <= 5)
                self.assertIn('data_quality', valid_game)
        
        # Run the async test
        asyncio.run(run_validation_test())
    
    def test_error_handling_pipeline(self):
        """Test error handling across the entire pipeline"""
        
        collector = NFLDataCollector(db_path=self.temp_db)
        
        # Test with network errors
        with patch.object(collector.session, 'get') as mock_get:
            mock_get.side_effect = Exception("Network Error")
            
            # Should handle gracefully
            schedule_df = collector.fetch_nfl_schedule(2024, 1)
            self.assertTrue(schedule_df.empty)
            
            team_stats_df = collector.fetch_team_stats(2024)
            self.assertTrue(team_stats_df.empty)
            
            player_stats_df = collector.fetch_player_stats(2024, 'QB')
            self.assertTrue(player_stats_df.empty)
        
        # Test database errors
        invalid_collector = NFLDataCollector(db_path="/invalid/path/test.db")
        test_df = pd.DataFrame([{'test': 'data'}])
        
        # Should not raise exception
        try:
            invalid_collector.save_to_database(test_df, 'test_table')
        except Exception as e:
            self.fail(f"save_to_database raised {type(e).__name__} unexpectedly!")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for the pipeline"""
        import time
        
        collector = NFLDataCollector(db_path=self.temp_db)
        
        # Test rate limiting performance
        start_time = time.time()
        for i in range(3):
            collector._rate_limit('test_source')
        end_time = time.time()
        
        # Should respect rate limits (approximately 2+ seconds for 3 calls)
        self.assertGreater(end_time - start_time, 1.5)
        
        # Test DataFrame operations performance
        large_df = pd.DataFrame({
            'game_id': range(1000),
            'home_team': ['KC'] * 1000,
            'away_team': ['BAL'] * 1000,
            'venue': ['Arrowhead Stadium'] * 1000
        })
        
        start_time = time.time()
        weather_df = collector.fetch_weather_data(large_df)
        end_time = time.time()
        
        # Should process quickly (under 1 second for 1000 games)
        self.assertLess(end_time - start_time, 1.0)
        self.assertEqual(len(weather_df), 1000)
        self.assertIn('temperature', weather_df.columns)


def run_integration_tests():
    """Run all integration tests"""
    print("🏈 NFL DATA INGESTION PIPELINE INTEGRATION TESTS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataIngestionPipeline)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                   result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSUCCESS RATE: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    
    if success:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
        print("The NFL data ingestion pipeline is ready for production.")
    else:
        print("\n❌ SOME INTEGRATION TESTS FAILED!")
        print("Please review and fix issues before proceeding.")
    
    exit(0 if success else 1)
