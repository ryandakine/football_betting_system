#!/usr/bin/env python3
"""
Simple fix to add the missing bet_size column
"""
import sqlite3
import sys


def add_bet_size_column():
    """Add the missing bet_size column"""
    try:
        # Connect to database
        print("🔌 Connecting to database...")
        conn = sqlite3.connect("gold_standard_betting.db")
        cursor = conn.cursor()

        # Check current columns
        cursor.execute("PRAGMA table_info(recommendations);")
        columns = [col[1] for col in cursor.fetchall()]

        print(f"📊 Current columns: {len(columns)}")

        # Check if bet_size exists
        if "bet_size" in columns:
            print("✅ bet_size column already exists!")
            return True

        print("➕ Adding bet_size column...")

        # Add the missing column
        cursor.execute("ALTER TABLE recommendations ADD COLUMN bet_size REAL")
        conn.commit()

        # Verify it was added
        cursor.execute("PRAGMA table_info(recommendations);")
        new_columns = [col[1] for col in cursor.fetchall()]

        if "bet_size" in new_columns:
            print("✅ bet_size column successfully added!")
            print(f"📊 Total columns now: {len(new_columns)}")
            return True
        else:
            print("❌ Failed to add bet_size column")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    print("🔧 Adding Missing bet_size Column")
    print("=" * 40)

    success = add_bet_size_column()

    print("=" * 40)
    if success:
        print("🎉 SUCCESS! bet_size column added!")
        print("🚀 Now run: python gold_standard_mlb_launcher.py")
    else:
        print("❌ Failed to add column")
        sys.exit(1)
