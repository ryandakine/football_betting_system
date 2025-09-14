#!/usr/bin/env python3
"""
Fix database schema for the learning system
"""

import os
import sqlite3
from pathlib import Path


def fix_database_schema():
    """Fix the database schema issues."""
    print("🔧 Fixing Database Schema")
    print("=" * 30)

    # Database path
    db_path = "data/learning_system.db"

    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if predictions table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        )
        table_exists = cursor.fetchone() is not None

        if table_exists:
            print("✅ Predictions table exists")

            # Check if 'date' column exists
            cursor.execute("PRAGMA table_info(predictions)")
            columns = [column[1] for column in cursor.fetchall()]

            if "date" not in columns:
                print("⚠️  Adding missing 'date' column...")
                cursor.execute("ALTER TABLE predictions ADD COLUMN date TEXT")
                print("✅ 'date' column added")
            else:
                print("✅ 'date' column exists")

            # Check if 'timestamp' column exists
            if "timestamp" not in columns:
                print("⚠️  Adding missing 'timestamp' column...")
                cursor.execute("ALTER TABLE predictions ADD COLUMN timestamp TEXT")
                print("✅ 'timestamp' column added")
            else:
                print("✅ 'timestamp' column exists")
        else:
            print("⚠️  Predictions table doesn't exist, creating it...")
            create_predictions_table(cursor)

        # Check if learning_metrics table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='learning_metrics'"
        )
        metrics_exists = cursor.fetchone() is not None

        if not metrics_exists:
            print("⚠️  Learning metrics table doesn't exist, creating it...")
            create_learning_metrics_table(cursor)

        # Check if historical_data table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='historical_data'"
        )
        historical_exists = cursor.fetchone() is not None

        if not historical_exists:
            print("⚠️  Historical data table doesn't exist, creating it...")
            create_historical_data_table(cursor)

        conn.commit()
        print("✅ Database schema fixed successfully!")

        # Show table structure
        print("\n📊 Database Tables:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")

    except Exception as e:
        print(f"❌ Error fixing database schema: {e}")
        conn.rollback()
    finally:
        conn.close()


def create_predictions_table(cursor):
    """Create the predictions table with correct schema."""
    cursor.execute(
        """
        CREATE TABLE predictions (
            prediction_id TEXT PRIMARY KEY,
            timestamp TEXT,
            date TEXT,
            game_id TEXT,
            home_team TEXT,
            away_team TEXT,
            predicted_winner TEXT,
            confidence REAL,
            stake_amount REAL,
            odds REAL,
            expected_value REAL,
            features TEXT,
            model_name TEXT,
            strategy_type TEXT,
            actual_winner TEXT,
            was_correct INTEGER,
            actual_profit REAL,
            actual_roi REAL,
            learning_weight REAL,
            pattern_strength REAL
        )
    """
    )
    print("✅ Predictions table created")


def create_learning_metrics_table(cursor):
    """Create the learning_metrics table."""
    cursor.execute(
        """
        CREATE TABLE learning_metrics (
            date TEXT PRIMARY KEY,
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy REAL,
            total_profit REAL,
            total_invested REAL,
            roi REAL,
            model_performance TEXT,
            feature_importance TEXT,
            pattern_insights TEXT
        )
    """
    )
    print("✅ Learning metrics table created")


def create_historical_data_table(cursor):
    """Create the historical_data table."""
    cursor.execute(
        """
        CREATE TABLE historical_data (
            date TEXT,
            game_id TEXT,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            winner TEXT,
            features TEXT,
            PRIMARY KEY (date, game_id)
        )
    """
    )
    print("✅ Historical data table created")


def test_database():
    """Test the database after fixing."""
    print("\n🧪 Testing Database")
    print("=" * 20)

    db_path = "data/learning_system.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Test predictions table
        cursor.execute("SELECT COUNT(*) FROM predictions")
        prediction_count = cursor.fetchone()[0]
        print(f"✅ Predictions table: {prediction_count} records")

        # Test learning_metrics table
        cursor.execute("SELECT COUNT(*) FROM learning_metrics")
        metrics_count = cursor.fetchone()[0]
        print(f"✅ Learning metrics table: {metrics_count} records")

        # Test historical_data table
        cursor.execute("SELECT COUNT(*) FROM historical_data")
        historical_count = cursor.fetchone()[0]
        print(f"✅ Historical data table: {historical_count} records")

        print("✅ Database test completed successfully!")

    except Exception as e:
        print(f"❌ Database test failed: {e}")
    finally:
        conn.close()


def main():
    """Main function."""
    fix_database_schema()
    test_database()

    print("\n" + "=" * 50)
    print("Database setup complete!")
    print("\nNext steps:")
    print("1. Run: .\\run_test.bat")
    print("2. Start the learning API server: python learning_api_server.py")


if __name__ == "__main__":
    main()
