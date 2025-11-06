#!/usr/bin/env python3
"""
Gold Standard MLB Database Readiness Checker
Checks if all required columns exist before running the betting system
"""
import os
import sqlite3
import sys


def check_database_readiness():
    """Check if database has all required columns"""
    db_path = "gold_standard_betting.db"

    # Check if database exists
    if not os.path.exists(db_path):
        print("❌ Database not found!")
        return False

    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if recommendations table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recommendations';"
        )
        if not cursor.fetchone():
            print("❌ No 'recommendations' table found!")
            return False

        # Get current table structure
        cursor.execute("PRAGMA table_info(recommendations);")
        columns_info = cursor.fetchall()
        existing_columns = {column[1] for column in columns_info}

        # Define all possible required columns (comprehensive list)
        required_columns = {
            # Core essentials that the system definitely needs
            "timestamp",
            "id",
            "game_id",
            "bet_type",
            "bet_size",  # bet_size was missing!
            "market",
            "team",
            "opponent",
            "player",
            "selection",
            "line",
            "point",
            "odds",
            "units",
            "stake",
            "ai_confidence",
            "expected_value",
            "confidence_tier",
            "claude_analysis",
            "openai_analysis",
            "grok_analysis",
            "claude_confidence",
            "openai_confidence",
            "grok_confidence",
            "claude_selection",
            "openai_selection",
            "grok_selection",
            "bookmaker",
            "created_at",
            "game_time",
            "actual_result",
            "actual_profit",
            # Additional columns the system might use
            "win_probability",
            "confidence_score",
            "confidence",
            "kelly_fraction",
            "recommended_stake",
            "risk_level",
            "analysis_consensus",
            "sportsbook",
        }

        # Check which columns are missing
        missing_columns = required_columns - existing_columns

        print("🔍 Database Readiness Check")
        print("=" * 50)
        print(f"📊 Total columns found: {len(existing_columns)}")
        print(f"📋 Required columns: {len(required_columns)}")

        if missing_columns:
            print(f"❌ Missing {len(missing_columns)} columns:")
            for col in sorted(missing_columns):
                if col == "bet_size":
                    print(f"   🔥 {col} ← THIS WAS CAUSING THE ERROR!")
                else:
                    print(f"   📌 {col}")
            print("\n🔧 Run: python fix_database.py")
            return False
        else:
            print("✅ All required columns present!")
            print("🎯 Database is ready for your betting system!")
            print("🔥 bet_size column confirmed ✅")
            print("🚀 You can safely run: python gold_standard_mlb_launcher.py")
            return True

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        if conn:
            conn.close()


def show_current_structure():
    """Show the current database structure"""
    db_path = "gold_standard_betting.db"

    if not os.path.exists(db_path):
        print("❌ Database not found!")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(recommendations);")
        columns_info = cursor.fetchall()

        print("\n📋 Current Database Structure:")
        print("-" * 40)
        for i, col_info in enumerate(columns_info, 1):
            print(f"{i:2d}. {col_info[1]} ({col_info[2]})")

    except Exception as e:
        print(f"❌ Error reading structure: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    print("🏆 Gold Standard MLB Database Checker")
    print("=" * 50)

    is_ready = check_database_readiness()

    if not is_ready:
        show_current_structure()
        print("\n" + "=" * 50)
        print("🔧 Database needs fixing - run fix_database.py first!")
    else:
        print("\n" + "=" * 50)
        print("🎉 DATABASE READY! Go get those 23+ profitable bets!")

    sys.exit(0 if is_ready else 1)
