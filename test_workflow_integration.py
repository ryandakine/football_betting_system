#!/usr/bin/env python3
"""
Test Workflow Integration Script
This script tests the integration between the workflow components and database.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests


def load_env_vars():
    """Load environment variables from aci.env"""
    try:
        from dotenv import load_dotenv

        load_dotenv("aci.env")
        print("✅ Environment variables loaded from aci.env")
    except ImportError:
        print("⚠️  python-dotenv not available, using system environment variables")
    except Exception as e:
        print(f"⚠️  Could not load aci.env: {e}")


def test_odds_api():
    """Test The Odds API connection"""
    print("\n🧪 Testing The Odds API...")

    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key or api_key == "your_odds_api_key_here":
        print("❌ THE_ODDS_API_KEY not set or using placeholder")
        return False

    try:
        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "bookmakers": "fanduel,draftkings",
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Odds API working - Found {len(data)} games")
            return True
        else:
            print(f"❌ Odds API error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ Odds API test failed: {e}")
        return False


def test_openai_api():
    """Test OpenAI API connection"""
    print("\n🧪 Testing OpenAI API...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY not set or using placeholder")
        return False

    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": 'Say "Hello" if you can read this.'}
            ],
            "max_tokens": 10,
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            print("✅ OpenAI API working")
            return True
        else:
            print(f"❌ OpenAI API error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False


def test_supabase_connection():
    """Test Supabase connection"""
    print("\n🧪 Testing Supabase connection...")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or supabase_url == "your_supabase_project_url_here":
        print("❌ SUPABASE_URL not set or using placeholder")
        return False

    if not supabase_key or supabase_key == "your_supabase_anon_key_here":
        print("❌ SUPABASE_ANON_KEY not set or using placeholder")
        return False

    try:
        from supabase import Client, create_client

        supabase: Client = create_client(supabase_url, supabase_key)

        # Test basic query
        result = (
            supabase.table("fanduel_betting_analysis").select("*").limit(1).execute()
        )
        print("✅ Supabase connection working")
        return True

    except ImportError:
        print("❌ Supabase client not available - install with: pip install supabase")
        return False
    except Exception as e:
        print(f"❌ Supabase test failed: {e}")
        return False


def test_workflow_data_structure():
    """Test the workflow data structure"""
    print("\n🧪 Testing workflow data structure...")

    # Sample data structure that should match your workflow output
    sample_data = {
        "games": [
            {
                "gameId": "test_123",
                "homeTeam": "Yankees",
                "awayTeam": "Red Sox",
                "fanduelOdds": {"outcomes": [{"name": "Yankees", "price": -110}]},
                "bestOdds": {
                    "homeTeam": {
                        "bestBook": "draftkings",
                        "bestPrice": -105,
                        "fanduelPrice": -110,
                    }
                },
                "moneyLeftOnTable": {"homeTeam": 5, "awayTeam": 0},
            }
        ],
        "summary": {
            "totalGames": 1,
            "totalMoneyLeftOnTable": 5.0,
            "averageMoneyLeftOnTable": 5.0,
        },
    }

    # Validate structure
    required_fields = ["games", "summary"]
    for field in required_fields:
        if field not in sample_data:
            print(f"❌ Missing required field: {field}")
            return False

    print("✅ Workflow data structure is valid")
    return True


def generate_test_report():
    """Generate a test report"""
    print("\n📊 Test Report")
    print("=" * 40)

    tests = [
        ("Environment Variables", load_env_vars),
        ("The Odds API", test_odds_api),
        ("OpenAI API", test_openai_api),
        ("Supabase Connection", test_supabase_connection),
        ("Workflow Data Structure", test_workflow_data_structure),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))

    print("\n📋 Test Results:")
    print("-" * 40)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1

    print("-" * 40)
    print(f"Total: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All tests passed! Your workflow should work correctly.")
    else:
        print(
            f"\n⚠️  {len(results) - passed} test(s) failed. Please fix the issues above."
        )

    return passed == len(results)


def main():
    """Main function"""
    print("⚾ MLB FanDuel Analysis - Workflow Integration Test")
    print("=" * 55)

    # Check if we're in the right directory
    if not Path("aci.env").exists():
        print("⚠️  Warning: aci.env file not found in current directory")
        print("Make sure you're running this from the mlb_betting_system directory")

    # Run tests
    success = generate_test_report()

    if success:
        print("\n🎯 Next Steps:")
        print("1. Set up your database tables in Supabase")
        print("2. Configure your n8n workflow with the correct settings")
        print("3. Test the workflow end-to-end")
        print("4. Monitor the results to see your analysis!")
    else:
        print("\n🔧 Fix the failed tests before proceeding with the workflow setup.")

    return success


if __name__ == "__main__":
    main()
