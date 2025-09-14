#!/usr/bin/env python3
"""
MLB Opportunity Detector Setup
=============================
Quick setup using your existing Supabase project
"""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_mlb_detector():
    print("🚀 Setting up Enhanced MLB Opportunity Detector...\n")

    # Check existing API keys
    print("✅ Your existing API keys:")
    print(f"   YouTube API: {'Set' if os.getenv('YOUTUBE_API_KEY') else 'Missing'}")
    print(f"   Odds API: {'Set' if os.getenv('THE_ODDS_API_KEY') else 'Missing'}")
    print(f"   Slack Webhook: {'Set' if os.getenv('SLACK_WEBHOOK_URL') else 'Missing'}")

    # Check Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_anon_key:
        print("\n❌ Supabase credentials not found in .env file")
        print("\n📋 To get your Supabase credentials:")
        print("1. Go to your Supabase project dashboard")
        print("2. Settings → API")
        print("3. Copy the Project URL and anon public key")
        print("4. Add them to your .env file:")
        print("\n   SUPABASE_URL=your_project_url_here")
        print("   SUPABASE_ANON_KEY=your_anon_key_here")

        return False

    print(f"   Supabase URL: {'Set' if supabase_url else 'Missing'}")
    print(f"   Supabase Anon Key: {'Set' if supabase_anon_key else 'Missing'}")

    print("\n🎯 All credentials found! Ready to proceed.")

    # Create SQL for MLB tables
    print("\n📝 Creating SQL for MLB tables...")

    sql_script = """
-- MLB Opportunity Detector Tables
-- Run this in your Supabase SQL editor

-- Create sentiment_data table
CREATE TABLE IF NOT EXISTS sentiment_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    source TEXT NOT NULL,
    data JSONB,
    sentiment_summary JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create odds_data table
CREATE TABLE IF NOT EXISTS odds_data (
    id SERIAL PRIMARY KEY,
    game_id TEXT,
    bookmaker TEXT,
    home_team TEXT,
    away_team TEXT,
    commence_time TIMESTAMP,
    moneyline_home INTEGER,
    moneyline_away INTEGER,
    edge_cents INTEGER,
    value_rating TEXT,
    market TEXT,
    source TEXT,
    date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create agent_activity table
CREATE TABLE IF NOT EXISTS agent_activity (
    id SERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    activity_type TEXT NOT NULL,
    data JSONB,
    status TEXT DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for better performance
CREATE INDEX IF NOT EXISTS idx_sentiment_data_date ON sentiment_data(date);
CREATE INDEX IF NOT EXISTS idx_odds_data_date ON odds_data(date);
CREATE INDEX IF NOT EXISTS idx_agent_activity_agent_id ON agent_activity(agent_id);
"""

    # Save SQL to file
    with open("mlb_tables.sql", "w") as f:
        f.write(sql_script)

    print("✅ Created mlb_tables.sql file")

    # Create n8n environment variables
    print("\n🔧 n8n Environment Variables:")
    print("Add these to your n8n instance:")
    print("\n" + "=" * 50)
    print("YOUTUBE_API_KEY=" + (os.getenv("YOUTUBE_API_KEY") or "your_youtube_key"))
    print("ODDS_API_KEY=" + (os.getenv("THE_ODDS_API_KEY") or "your_odds_key"))
    print(
        "SLACK_WEBHOOK_URL=" + (os.getenv("SLACK_WEBHOOK_URL") or "your_slack_webhook")
    )
    print("SUPABASE_URL=" + (supabase_url or "your_supabase_url"))
    print("SUPABASE_ANON_KEY=" + (supabase_anon_key or "your_supabase_anon_key"))
    print("=" * 50)

    # Next steps
    print("\n📋 Next Steps:")
    print("1. Run the SQL in mlb_tables.sql in your Supabase SQL editor")
    print("2. Import enhanced-mlb-opportunity-detector.json into n8n")
    print("3. Configure the environment variables in n8n")
    print("4. Update the Supabase nodes with your credentials")
    print("5. Test and activate the workflow")

    print("\n🎉 Your Enhanced MLB Opportunity Detector will be ready!")

    return True


if __name__ == "__main__":
    setup_mlb_detector()
