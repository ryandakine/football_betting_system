#!/usr/bin/env python3
"""
Set environment variables from aci.env
"""

import os

# Set the API keys from your aci.env file
os.environ["SUPABASE_URL"] = "https://jufurqxkwbkpoclkeyoi.supabase.co"
os.environ["SUPABASE_ANON_KEY"] = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1ZnVycXhrd2JrcG9jbGtleW9pIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM0MTMyODYsImV4cCI6MjA2ODk4OTI4Nn0.V-VN7tc_QhQv1fajmsRl3LcUctY29LfjuWX_JstRW3M"
)

# Set placeholder for missing keys (system will use sample data)
os.environ["ODDS_API_KEY"] = "sample_key_for_testing"
os.environ["YOUTUBE_API_KEY"] = "sample_key_for_testing"

print("✅ Environment variables set:")
print(f"  SUPABASE_URL: {os.environ.get('SUPABASE_URL', 'Not set')}")
print(
    f"  SUPABASE_ANON_KEY: {'Set' if os.environ.get('SUPABASE_ANON_KEY') else 'Not set'}"
)
print(f"  ODDS_API_KEY: {'Set' if os.environ.get('ODDS_API_KEY') else 'Not set'}")
print(f"  YOUTUBE_API_KEY: {'Set' if os.environ.get('YOUTUBE_API_KEY') else 'Not set'}")

print("\n🎯 Ready to start the daily prediction system!")
print("The system will use sample data for missing API keys.")
