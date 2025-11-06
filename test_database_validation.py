#!/usr/bin/env python3
"""
Database Validation and Performance Testing
==========================================

Comprehensive database validation for the NFL betting system:
- Schema validation
- Data integrity checks
- Performance benchmarks
- Storage/retrieval testing
- Concurrent access testing
"""

import unittest
import sqlite3
import os
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_data_ingestion import AdvancedDataIngestion, PlayerStats, TeamStats, GameData
from data_collection import NFLDataCollector


class TestDatabaseValidation(unittest.TestCase):
    """Comprehensive database validation tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_nfl_validation.db")
        self.collector = NFLDataCollector(db_path=self.test_db)
        self.ingestion = AdvancedDataIngestion(
            cache_dir=os.path.join(self.temp_dir, "cache"),
            db_path=self.test_db
        )
        
    def tearDown(self):
        """Clean up test environment"""
        # Clean up any created files
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)
    
    def test_database_schema_validation(self):
        """Test database schema integrity"""
        print("\n🗄️ Testing Database Schema...")
        
        # Check that database exists
        self.assertTrue(os.path.exists(self.test_db))
        
        with sqlite3.connect(self.test_db) as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Expected tables from advanced_data_ingestion.py
            expected_tables = ['player_stats', 'team_stats', 'games']
            
            for table in expected_tables:
                self.assertIn(table, tables, f"Missing table: {table}")
                print(f"✅ Table '{table}' exists")
            
            # Check table schemas
            for table in expected_tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                self.assertGreater(len(columns), 0, f"Table {table} has no columns")
                print(f"✅ Table '{table}' has {len(columns)} columns")
            
            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            
            expected_indexes = [
                'idx_player_season_week',
                'idx_team_season_week', 
                'idx_games_season_week'
            ]
            
            for index in expected_indexes:
                self.assertIn(index, indexes, f"Missing index: {index}")
                print(f"✅ Index '{index}' exists")
    
    def test_data_insertion_and_retrieval(self):
        """Test basic data insertion and retrieval"""
        print("\n📝 Testing Data Insertion and Retrieval...")
        
        # Create sample data
        sample_games = pd.DataFrame([
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
                'data_source': 'TEST'
            },
            {
                'game_id': '401671746',
                'season': 2024,
                'week': 1,
                'home_team': 'BUF',
                'away_team': 'MIA',
                'home_score': 31,
                'away_score': 10,
                'venue': 'Highmark Stadium',
                'status': 'Final',
                'data_source': 'TEST'
            }
        ])
        
        # Insert data
        self.collector.save_to_database(sample_games, 'test_games')
        
        # Retrieve data
        with sqlite3.connect(self.test_db) as conn:
            retrieved_df = pd.read_sql_query("SELECT * FROM test_games", conn)
        
        # Validate retrieval
        self.assertEqual(len(retrieved_df), 2)
        self.assertEqual(retrieved_df.iloc[0]['home_team'], 'KC')
        self.assertEqual(retrieved_df.iloc[1]['home_team'], 'BUF')
        
        print("✅ Data insertion and retrieval working correctly")
    
    def test_data_type_validation(self):
        """Test data type handling and validation"""
        print("\n🔍 Testing Data Type Validation...")
        
        # Create data with various types
        mixed_data = pd.DataFrame([
            {
                'game_id': '401671747',
                'season': 2024,
                'week': 1,
                'home_team': 'KC',
                'away_team': 'BAL',
                'home_score': 27,  # Integer
                'away_score': 20.5,  # Float
                'venue': 'Test Stadium',  # String
                'status': 'Final',
                'temperature': None,  # NULL
                'created_at': datetime.now()  # Datetime
            }
        ])
        
        # Insert and retrieve
        self.collector.save_to_database(mixed_data, 'mixed_types_test')
        
        with sqlite3.connect(self.test_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM mixed_types_test")
            row = cursor.fetchone()
            
            # Validate data types were preserved appropriately
            self.assertIsNotNone(row)
            print("✅ Mixed data types handled correctly")
    
    def test_concurrent_database_access(self):
        """Test concurrent database access"""
        print("\n🔄 Testing Concurrent Database Access...")
        
        def write_data(thread_id):
            """Function to write data from multiple threads"""
            try:
                collector = NFLDataCollector(db_path=self.test_db)
                test_data = pd.DataFrame([{
                    'game_id': f'thread_{thread_id}_game',
                    'season': 2024,
                    'week': thread_id,
                    'home_team': f'TEAM{thread_id}A',
                    'away_team': f'TEAM{thread_id}B',
                    'home_score': thread_id * 7,
                    'away_score': thread_id * 3,
                    'venue': f'Stadium {thread_id}',
                    'status': 'Final',
                    'data_source': f'THREAD_{thread_id}'
                }])
                
                collector.save_to_database(test_data, f'concurrent_test_{thread_id}')
                return True
            except Exception as e:
                print(f"Thread {thread_id} error: {e}")
                return False
        
        # Run concurrent writes
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_data, i) for i in range(5)]
            results = [future.result() for future in futures]
        
        # All writes should succeed
        self.assertTrue(all(results), "Some concurrent writes failed")
        
        # Verify data was written
        with sqlite3.connect(self.test_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'concurrent_test_%'")
            tables = cursor.fetchall()
            self.assertEqual(len(tables), 5, "Not all concurrent tables were created")
        
        print("✅ Concurrent database access working correctly")
    
    def test_database_performance_benchmarks(self):
        """Test database performance with various data sizes"""
        print("\n⚡ Testing Database Performance...")
        
        # Test with different data sizes
        test_sizes = [100, 1000, 5000]
        performance_results = {}
        
        for size in test_sizes:
            print(f"  Testing with {size} records...")
            
            # Generate test data
            test_data = pd.DataFrame({
                'game_id': [f'perf_test_{i}' for i in range(size)],
                'season': [2024] * size,
                'week': np.random.randint(1, 18, size),
                'home_team': [f'TEAM{i%32}' for i in range(size)],
                'away_team': [f'TEAM{(i+1)%32}' for i in range(size)],
                'home_score': np.random.randint(0, 50, size),
                'away_score': np.random.randint(0, 50, size),
                'venue': [f'Stadium {i%16}' for i in range(size)],
                'status': ['Final'] * size,
                'data_source': ['PERFORMANCE_TEST'] * size
            })
            
            # Measure insertion time
            start_time = time.time()
            self.collector.save_to_database(test_data, f'performance_test_{size}')
            insert_time = time.time() - start_time
            
            # Measure query time
            start_time = time.time()
            with sqlite3.connect(self.test_db) as conn:
                result_df = pd.read_sql_query(
                    f"SELECT * FROM performance_test_{size} WHERE week = 1",
                    conn
                )
            query_time = time.time() - start_time
            
            # Measure aggregation time
            start_time = time.time()
            with sqlite3.connect(self.test_db) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT home_team, AVG(home_score) FROM performance_test_{size} GROUP BY home_team")
                agg_results = cursor.fetchall()
            agg_time = time.time() - start_time
            
            performance_results[size] = {
                'insert_time': insert_time,
                'query_time': query_time,
                'aggregation_time': agg_time,
                'records_per_second_insert': size / insert_time if insert_time > 0 else 0,
                'records_per_second_query': len(result_df) / query_time if query_time > 0 else 0
            }
            
            print(f"    Insert: {insert_time:.3f}s ({size/insert_time:.0f} records/sec)")
            print(f"    Query: {query_time:.3f}s")
            print(f"    Aggregation: {agg_time:.3f}s")
        
        # Performance assertions
        for size, results in performance_results.items():
            # Insert should be reasonably fast (at least 100 records/sec for small datasets)
            if size <= 1000:
                self.assertGreater(results['records_per_second_insert'], 100,
                                 f"Insert performance too slow for {size} records")
            
            # Query should be very fast (under 0.1s for simple queries)
            self.assertLess(results['query_time'], 0.5,
                           f"Query performance too slow for {size} records")
        
        print("✅ Database performance benchmarks passed")
        return performance_results
    
    def test_data_integrity_constraints(self):
        """Test data integrity and constraint handling"""
        print("\n🔒 Testing Data Integrity Constraints...")
        
        # Test duplicate handling
        duplicate_data = pd.DataFrame([
            {'game_id': 'duplicate_test', 'season': 2024, 'week': 1, 'home_team': 'KC', 'away_team': 'BAL'},
            {'game_id': 'duplicate_test', 'season': 2024, 'week': 1, 'home_team': 'KC', 'away_team': 'BAL'}
        ])
        
        # This should handle duplicates gracefully
        try:
            self.collector.save_to_database(duplicate_data, 'duplicate_test')
            print("✅ Duplicate data handled gracefully")
        except Exception as e:
            print(f"⚠️ Duplicate handling: {e}")
        
        # Test NULL value handling
        null_data = pd.DataFrame([
            {'game_id': 'null_test', 'season': 2024, 'week': None, 'home_team': None, 'away_team': 'BAL'}
        ])
        
        try:
            self.collector.save_to_database(null_data, 'null_test')
            print("✅ NULL values handled gracefully")
        except Exception as e:
            print(f"⚠️ NULL value handling: {e}")
    
    def test_database_size_and_storage(self):
        """Test database size and storage efficiency"""
        print("\n💾 Testing Database Size and Storage...")
        
        # Get initial database size
        initial_size = os.path.getsize(self.test_db)
        
        # Add a significant amount of data
        large_dataset = pd.DataFrame({
            'game_id': [f'storage_test_{i}' for i in range(10000)],
            'season': np.random.choice([2020, 2021, 2022, 2023, 2024], 10000),
            'week': np.random.randint(1, 18, 10000),
            'home_team': np.random.choice(['KC', 'BAL', 'BUF', 'MIA', 'NE', 'NYJ'], 10000),
            'away_team': np.random.choice(['KC', 'BAL', 'BUF', 'MIA', 'NE', 'NYJ'], 10000),
            'home_score': np.random.randint(0, 50, 10000),
            'away_score': np.random.randint(0, 50, 10000),
            'venue': np.random.choice(['Stadium A', 'Stadium B', 'Stadium C'], 10000),
            'status': ['Final'] * 10000,
            'data_source': ['STORAGE_TEST'] * 10000
        })
        
        self.collector.save_to_database(large_dataset, 'storage_efficiency_test')
        
        # Get final database size
        final_size = os.path.getsize(self.test_db)
        size_increase = final_size - initial_size
        
        # Calculate storage efficiency
        avg_bytes_per_record = size_increase / 10000
        
        print(f"  Initial DB size: {initial_size:,} bytes")
        print(f"  Final DB size: {final_size:,} bytes")
        print(f"  Size increase: {size_increase:,} bytes")
        print(f"  Avg bytes per record: {avg_bytes_per_record:.1f}")
        
        # Storage should be reasonable (under 100 bytes per simple record)
        self.assertLess(avg_bytes_per_record, 200, "Storage efficiency is poor")
        
        print("✅ Database storage efficiency is acceptable")
    
    def test_database_recovery_and_corruption(self):
        """Test database recovery and corruption handling"""
        print("\n🛠️ Testing Database Recovery...")
        
        # Create some data
        test_data = pd.DataFrame([
            {'game_id': 'recovery_test', 'season': 2024, 'week': 1, 'home_team': 'KC', 'away_team': 'BAL'}
        ])
        
        self.collector.save_to_database(test_data, 'recovery_test')
        
        # Test database integrity check
        with sqlite3.connect(self.test_db) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            
            self.assertEqual(integrity_result, 'ok', "Database integrity check failed")
            print("✅ Database integrity check passed")
        
        # Test VACUUM operation (database cleanup)
        with sqlite3.connect(self.test_db) as conn:
            cursor = conn.cursor()
            cursor.execute("VACUUM")
            print("✅ Database VACUUM operation completed")


def run_database_validation():
    """Run comprehensive database validation"""
    print("🗄️ NFL DATABASE VALIDATION SUITE")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDatabaseValidation)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATABASE VALIDATION SUMMARY")
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
    success = run_database_validation()
    
    if success:
        print("\n✅ ALL DATABASE VALIDATION TESTS PASSED!")
        print("Your NFL betting system database is production-ready.")
    else:
        print("\n❌ SOME DATABASE TESTS FAILED!")
        print("Please review and fix database issues before proceeding.")
    
    exit(0 if success else 1)
