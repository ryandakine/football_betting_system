#!/usr/bin/env python3
"""
Test Supabase Setup for MLB Opportunity Detector
"""

import json

import requests

# Your Supabase credentials
SUPABASE_URL = "https://jufurqxkwbkpoclkeyoi.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1ZnVycXhrd2JrcG9jbGtleW9pIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM0MTMyODYsImV4cCI6MjA2ODk4OTI4Nn0.V-VN7tc_QhQv1fajmsRl3LcUctY29LfjuWX_JstRW3M"


def test_supabase_connection():
    """Test Supabase connection and create tables."""
    print("🔍 Testing Supabase connection...")

    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }

    # Test connection
    try:
        response = requests.get(f"{SUPABASE_URL}/rest/v1/", headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ Supabase connection successful!")
        else:
            print(f"❌ Supabase connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Supabase connection error: {e}")
        return False

    # Test inserting data
    print("\n📝 Testing data insertion...")

    test_data = {
        "date": "2025-01-15",
        "source": "test",
        "data": {"test": "data"},
        "sentiment_summary": {"bullish": 1, "bearish": 0, "neutral": 0},
    }

    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/sentiment_data",
            headers=headers,
            json=test_data,
            timeout=10,
        )

        if response.status_code == 201:
            print("✅ Data insertion successful!")
            print("✅ Your Supabase is ready for the MLB Opportunity Detector!")
        else:
            print(f"❌ Data insertion failed: {response.status_code}")
            print("   You may need to create the tables first.")
            return False

    except Exception as e:
        print(f"❌ Data insertion error: {e}")
        return False

    return True


def print_n8n_config():
    """Print n8n configuration instructions."""
    print("\n🔧 n8n Environment Variables:")
    print("=" * 50)
    print("YOUTUBE_API_KEY=AIzaSyAirGlfovjzmg0xUvwA1VGBFDaFgwfQmYY")
    print("ODDS_API_KEY=ba91c642121f5bdd0d0357656c7c11d9")
    print(
        "SLACK_WEBHOOK_URL=https://hooks.slack.com/services/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"
    )
    print("SUPABASE_URL=https://jufurqxkwbkpoclkeyoi.supabase.co")
    print(
        "SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1ZnVycXhrd2JrcG9jbGtleW9pIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM0MTMyODYsImV4cCI6MjA2ODk4OTI4Nn0.V-VN7tc_QhQv1fajmsRl3LcUctY29LfjuWX_JstRW3M"
    )
    print("=" * 50)


def main():
    print("🚀 MLB Opportunity Detector Setup Test\n")

    # Test Supabase
    if test_supabase_connection():
        print_n8n_config()

        print("\n📋 Next Steps:")
        print("1. ✅ Supabase is ready!")
        print("2. Import enhanced-mlb-opportunity-detector.json into n8n")
        print("3. Add the environment variables above to n8n")
        print("4. Configure the Supabase nodes with your credentials")
        print("5. Test and activate the workflow")

        print("\n🎉 Your Enhanced MLB Opportunity Detector is ready to deploy!")
    else:
        print("\n❌ Setup failed. Please check your Supabase configuration.")
        print("📝 Make sure to run the SQL in mlb_tables.sql first:")


if __name__ == "__main__":
    main()
