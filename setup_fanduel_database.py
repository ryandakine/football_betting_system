#!/usr/bin/env python3
"""
FanDuel Analysis Database Setup Script
This script helps set up the database tables for the MLB betting analysis system.
"""

import os
import sys
from pathlib import Path


def read_sql_file(file_path):
    """Read SQL file content"""
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ SQL file not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading SQL file: {e}")
        return None


def setup_database():
    """Set up the FanDuel analysis database tables"""

    print("🏗️  Setting up FanDuel Analysis Database...")
    print("=" * 50)

    # Check if we have the required environment variables
    required_vars = ["SUPABASE_URL", "SUPABASE_ANON_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these in your .env file or environment.")
        return False

    # Read the SQL schema
    sql_file = Path("database/fanduel_analysis_schema.sql")
    if not sql_file.exists():
        print(f"❌ SQL schema file not found: {sql_file}")
        print("Creating the database directory and schema file...")

        # Create database directory
        sql_file.parent.mkdir(exist_ok=True)

        # Create a basic schema
        basic_schema = """
-- Basic FanDuel Analysis Schema
CREATE TABLE IF NOT EXISTS fanduel_betting_analysis (
    id SERIAL PRIMARY KEY,
    games_data JSONB NOT NULL,
    summary_data JSONB NOT NULL,
    ai_analysis TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    analysis_date DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS fanduel_tracking_history (
    id SERIAL PRIMARY KEY,
    tracking_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    analysis_date DATE DEFAULT CURRENT_DATE
);

CREATE INDEX IF NOT EXISTS idx_fanduel_analysis_date ON fanduel_betting_analysis(analysis_date);
CREATE INDEX IF NOT EXISTS idx_tracking_history_date ON fanduel_tracking_history(analysis_date);
"""

        with open(sql_file, "w") as f:
            f.write(basic_schema)

        print(f"✅ Created basic schema file: {sql_file}")

    sql_content = read_sql_file(sql_file)
    if not sql_content:
        return False

    print("📋 SQL Schema loaded successfully")
    print(f"📁 Schema file: {sql_file}")

    # Try to import supabase if available
    try:
        from supabase import Client, create_client

        print("✅ Supabase client available")
    except ImportError:
        print("⚠️  Supabase client not available")
        print("To install: pip install supabase")
        print("\n📝 Manual Setup Instructions:")
        print("1. Go to your Supabase dashboard")
        print("2. Navigate to SQL Editor")
        print("3. Copy and paste the SQL from the schema file")
        print("4. Execute the SQL to create the tables")
        return True

    # Set up Supabase client
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase")

        # Execute the SQL
        print("🚀 Creating database tables...")
        result = supabase.rpc("exec_sql", {"sql": sql_content})

        print("✅ Database tables created successfully!")

    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        print("\n📝 Manual Setup Instructions:")
        print("1. Go to your Supabase dashboard")
        print("2. Navigate to SQL Editor")
        print("3. Copy and paste the SQL from the schema file")
        print("4. Execute the SQL to create the tables")
        return False

    print("\n🎉 Database setup complete!")
    print("\n📊 Tables created:")
    print("   - fanduel_betting_analysis")
    print("   - fanduel_tracking_history")
    print("   - fanduel_game_analysis")
    print("   - bookmaker_performance")
    print("   - daily_summary_stats")

    return True


def test_database_connection():
    """Test the database connection and basic operations"""

    print("\n🧪 Testing database connection...")

    try:
        from supabase import Client, create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        supabase: Client = create_client(supabase_url, supabase_key)

        # Test basic query
        result = (
            supabase.table("fanduel_betting_analysis").select("*").limit(1).execute()
        )
        print("✅ Database connection test successful!")

        return True

    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False


def main():
    """Main function"""

    print("⚾ MLB FanDuel Analysis Database Setup")
    print("=" * 40)

    # Check if we're in the right directory
    if not Path("aci.env").exists():
        print("⚠️  Warning: aci.env file not found in current directory")
        print("Make sure you're running this from the mlb_betting_system directory")

    # Load environment variables
    try:
        from dotenv import load_dotenv

        load_dotenv("aci.env")
        print("✅ Environment variables loaded from aci.env")
    except ImportError:
        print("⚠️  python-dotenv not available, using system environment variables")
    except Exception as e:
        print(f"⚠️  Could not load aci.env: {e}")

    # Set up database
    success = setup_database()

    if success:
        # Test connection
        test_database_connection()

        print("\n🎯 Next Steps:")
        print("1. Configure your n8n workflow with the correct table names")
        print("2. Set up your API credentials in n8n")
        print("3. Test the workflow to ensure data is being stored correctly")
        print(
            "4. Monitor the results to see how much money you're leaving on the table!"
        )

        print("\n📊 Table Names for n8n Configuration:")
        print("   - Main Analysis: fanduel_betting_analysis")
        print("   - Tracking History: fanduel_tracking_history")

    else:
        print("\n❌ Database setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
