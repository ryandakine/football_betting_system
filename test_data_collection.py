#!/usr/bin/env python3
"""
Test Suite for NFL Data Collection Module
========================================

Comprehensive tests for data_collection.py to ensure all functionality works correctly.
Tests both individual components and integration scenarios.
"""

import unittest
import sqlite3
import os
import tempfile
import pandas as pd
from datetime import datetime
from unittest.mock import patch, Mock, MagicMock
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collection import NFLDataCollector, run_nfl_data_collection

class TestNFLDataCollector(unittest.TestCase):
    """Test cases for NFLDataCollector class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = os.path.join(self.temp_dir, "test_nfl.db")
        self.collector = NFLDataCollector(db_path=self.temp_db)
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        import time
        
        start_time = time.time()
        self.collector._rate_limit('test_source')
        self.collector._rate_limit('test_source')
        end_time = time.time()
        
        # Should have some delay
        self.assertGreater(end_time - start_time, 0.5)
    
    @patch('requests.Session.get')
    def test_fetch_nfl_schedule_success(self, mock_get):
        """Test successful NFL schedule fetching"""
        # Mock ESPN API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            'events': [
                {
                    'id': '12345',
                    'date': '2024-09-08T20:20Z',
                    'competitions': [{
                        'competitors': [
                            {
                                'team': {'abbreviation': 'KC'},
                                'homeAway': 'home',
                                'score': '21'
                            },
                            {
                                'team': {'abbreviation': 'BAL'},
                                'homeAway': 'away', 
                                'score': '17'
                            }
                        ],
                        'venue': {'fullName': 'Arrowhead Stadium'}
                    }],
                    'status': {'type': {'name': 'Final'}}
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Test the function
        df = self.collector.fetch_nfl_schedule(2024, 1)
        
        # Verify results
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['home_team'], 'KC')
        self.assertEqual(df.iloc[0]['away_team'], 'BAL')
        self.assertEqual(df.iloc[0]['home_score'], 21)
        self.assertEqual(df.iloc[0]['away_score'], 17)
        self.assertEqual(df.iloc[0]['venue'], 'Arrowhead Stadium')
    
    @patch('requests.Session.get')
    def test_fetch_nfl_schedule_api_error(self, mock_get):
        """Test NFL schedule fetching with API error"""
        mock_get.side_effect = Exception("API Error")
        
        df = self.collector.fetch_nfl_schedule(2024, 1)
        
        # Should return empty DataFrame on error
        self.assertTrue(df.empty)
    
    @patch('requests.Session.get')
    def test_fetch_team_stats_success(self, mock_get):
        """Test successful team stats fetching"""
        # Mock Pro Football Reference response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.content = """
        <html>
            <table id="team_stats">
                <thead>
                    <tr>
                        <th>Tm</th>
                        <th>G</th>
                        <th>W</th>
                        <th>L</th>
                        <th>PF</th>
                        <th>PA</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th>Kansas City Chiefs</th>
                        <td>17</td>
                        <td>14</td>
                        <td>3</td>
                        <td>456</td>
                        <td>329</td>
                    </tr>
                </tbody>
            </table>
        </html>
        """
        mock_get.return_value = mock_response
        
        df = self.collector.fetch_team_stats(2024)
        
        # Verify results
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['Tm'], 'Kansas City Chiefs')
        self.assertEqual(df.iloc[0]['W'], 14)
        self.assertEqual(df.iloc[0]['PF'], 456)
    
    @patch('requests.Session.get')
    def test_fetch_player_stats_success(self, mock_get):
        """Test successful player stats fetching"""
        # Mock Pro Football Reference response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.content = """
        <html>
            <table id="stats">
                <thead>
                    <tr>
                        <th>Player</th>
                        <th>Tm</th>
                        <th>Att</th>
                        <th>Cmp</th>
                        <th>Yds</th>
                        <th>TD</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <th>Patrick Mahomes</th>
                        <td>KC</td>
                        <td>597</td>
                        <td>401</td>
                        <td>4183</td>
                        <td>27</td>
                    </tr>
                </tbody>
            </table>
        </html>
        """
        mock_get.return_value = mock_response
        
        df = self.collector.fetch_player_stats(2024, 'QB')
        
        # Verify results
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['Player'], 'Patrick Mahomes')
        self.assertEqual(df.iloc[0]['Tm'], 'KC')
        self.assertEqual(df.iloc[0]['position'], 'QB')
        self.assertEqual(df.iloc[0]['Yds'], '4183')
    
    def test_fetch_weather_data(self):
        """Test weather data fetching (placeholder implementation)"""
        # Create sample games DataFrame
        games_df = pd.DataFrame([
            {
                'game_id': '12345',
                'home_team': 'KC',
                'away_team': 'BAL',
                'venue': 'Arrowhead Stadium'
            }
        ])
        
        result_df = self.collector.fetch_weather_data(games_df)
        
        # Should add weather columns
        self.assertIn('temperature', result_df.columns)
        self.assertIn('weather_conditions', result_df.columns)
        self.assertIn('wind_speed', result_df.columns)
        self.assertIn('humidity', result_df.columns)
    
    def test_save_to_database(self):
        """Test database saving functionality"""
        # Create sample DataFrame
        test_df = pd.DataFrame([
            {'team': 'KC', 'wins': 14, 'losses': 3},
            {'team': 'BAL', 'wins': 13, 'losses': 4}
        ])
        
        # Save to database
        self.collector.save_to_database(test_df, 'test_table')
        
        # Verify data was saved
        with sqlite3.connect(self.temp_db) as conn:
            saved_df = pd.read_sql_query("SELECT * FROM test_table", conn)
        
        self.assertEqual(len(saved_df), 2)
        self.assertEqual(saved_df.iloc[0]['team'], 'KC')
        self.assertEqual(saved_df.iloc[0]['wins'], 14)
    
    def test_save_empty_dataframe(self):
        """Test saving empty DataFrame"""
        empty_df = pd.DataFrame()
        
        # Should not raise error
        self.collector.save_to_database(empty_df, 'empty_table')
        
        # Verify table was not created
        with sqlite3.connect(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='empty_table'")
            result = cursor.fetchone()
        
        self.assertIsNone(result)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete data collection system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test environment"""
        # Clean up any created files
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.temp_dir)
    
    @patch('data_collection.NFLDataCollector.fetch_nfl_schedule')
    @patch('data_collection.NFLDataCollector.fetch_team_stats')
    @patch('data_collection.NFLDataCollector.fetch_player_stats')
    def test_collect_all_data_integration(self, mock_player_stats, mock_team_stats, mock_schedule):
        """Test complete data collection integration"""
        # Mock return values
        mock_schedule.return_value = pd.DataFrame([
            {'game_id': '12345', 'home_team': 'KC', 'away_team': 'BAL'}
        ])
        mock_team_stats.return_value = pd.DataFrame([
            {'team': 'KC', 'wins': 14}
        ])
        mock_player_stats.return_value = pd.DataFrame([
            {'player': 'Patrick Mahomes', 'team': 'KC'}
        ])
        
        # Create collector with temp database
        temp_db = os.path.join(self.temp_dir, "test_integration.db")
        collector = NFLDataCollector(db_path=temp_db)
        
        # Run collection
        results = collector.collect_all_data(2024, 1)
        
        # Verify all data types were collected
        self.assertIn('schedule', results)
        self.assertIn('team_stats', results)
        self.assertIn('player_stats', results)
        self.assertIn('schedule_with_weather', results)
        
        # Verify data was saved to database
        with sqlite3.connect(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['game_schedule', 'team_statistics', 'player_statistics', 'games_with_weather']
        for table in expected_tables:
            self.assertIn(table, tables)
    
    @patch('data_collection.NFLDataCollector.collect_all_data')
    def test_run_nfl_data_collection(self, mock_collect):
        """Test main data collection function"""
        # Mock return value
        mock_collect.return_value = {
            'schedule': pd.DataFrame([{'game_id': '12345'}]),
            'team_stats': pd.DataFrame([{'team': 'KC'}])
        }
        
        # Run collection
        with patch('os.path.join') as mock_join:
            mock_join.return_value = os.path.join(self.temp_dir, "test.parquet")
            results = run_nfl_data_collection(2024, save_to_files=False)
        
        # Verify results
        self.assertIn('schedule', results)
        self.assertIn('team_stats', results)
        mock_collect.assert_called_once_with(2024, None)

class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""
    
    def setUp(self):
        """Set up error handling tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_db = os.path.join(self.temp_dir, "test_error.db")
        self.collector = NFLDataCollector(db_path=self.temp_db)
    
    def tearDown(self):
        """Clean up error handling tests"""
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    @patch('requests.Session.get')
    def test_network_timeout(self, mock_get):
        """Test handling of network timeouts"""
        mock_get.side_effect = Exception("Timeout")
        
        df = self.collector.fetch_nfl_schedule(2024, 1)
        
        # Should handle gracefully and return empty DataFrame
        self.assertTrue(df.empty)
    
    @patch('requests.Session.get')
    def test_invalid_json_response(self, mock_get):
        """Test handling of invalid JSON responses"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        df = self.collector.fetch_nfl_schedule(2024, 1)
        
        # Should handle gracefully
        self.assertTrue(df.empty)
    
    def test_database_permission_error(self):
        """Test handling of database permission errors"""
        # Try to save to a read-only location
        readonly_collector = NFLDataCollector(db_path="/root/readonly.db")
        test_df = pd.DataFrame([{'test': 'data'}])
        
        # Should not raise exception
        try:
            readonly_collector.save_to_database(test_df, 'test_table')
        except Exception as e:
            self.fail(f"save_to_database raised {type(e).__name__} unexpectedly!")

def run_performance_tests():
    """Run basic performance tests"""
    print("\n" + "="*50)
    print("PERFORMANCE TESTS")
    print("="*50)
    
    import time
    
    # Test rate limiting performance
    collector = NFLDataCollector()
    
    print("Testing rate limiting performance...")
    start_time = time.time()
    for i in range(3):
        collector._rate_limit('test_source')
    end_time = time.time()
    
    print(f"Rate limiting for 3 requests took: {end_time - start_time:.2f} seconds")
    
    # Test DataFrame operations
    print("Testing DataFrame operations...")
    large_df = pd.DataFrame({
        'game_id': range(1000),
        'home_team': ['KC'] * 1000,
        'away_team': ['BAL'] * 1000
    })
    
    start_time = time.time()
    weather_df = collector.fetch_weather_data(large_df)
    end_time = time.time()
    
    print(f"Weather data processing for 1000 games took: {end_time - start_time:.2f} seconds")
    print(f"Result DataFrame shape: {weather_df.shape}")

if __name__ == '__main__':
    print("NFL Data Collection Test Suite")
    print("="*50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_tests()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
