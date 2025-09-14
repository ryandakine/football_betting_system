#!/usr/bin/env python3
"""
Test script for the daily prediction system
"""

import os
import sys
from datetime import datetime


def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")

    try:
        import pandas as pd

        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False

    try:
        import numpy as np

        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False

    try:
        from sklearn.ensemble import RandomForestClassifier

        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False

    try:
        import joblib

        print("✅ joblib imported successfully")
    except ImportError as e:
        print(f"❌ joblib import failed: {e}")
        return False

    try:
        import requests

        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
        return False

    return True


def test_learning_system():
    """Test the learning system initialization."""
    print("\nTesting learning system...")

    try:
        from self_learning_system import SelfLearningSystem

        print("✅ SelfLearningSystem imported successfully")

        # Initialize the system
        learning_system = SelfLearningSystem()
        print("✅ Learning system initialized successfully")

        # Test basic functionality
        summary = learning_system.get_learning_summary()
        print(
            f"✅ Learning summary retrieved: {summary['overall_metrics']['total_predictions']} total predictions"
        )

        return True

    except Exception as e:
        print(f"❌ Learning system test failed: {e}")
        return False


def test_daily_prediction_system():
    """Test the daily prediction system."""
    print("\nTesting daily prediction system...")

    try:
        from daily_prediction_system import DailyPredictionSystem
        from self_learning_system import SelfLearningSystem

        print("✅ DailyPredictionSystem imported successfully")

        # Get API key from environment
        odds_api_key = os.getenv("ODDS_API_KEY", "test_key")

        # Initialize systems
        learning_system = SelfLearningSystem()
        daily_system = DailyPredictionSystem(learning_system, odds_api_key)
        print("✅ Daily prediction system initialized successfully")

        return True

    except Exception as e:
        print(f"❌ Daily prediction system test failed: {e}")
        return False


def test_api_server():
    """Test the API server components."""
    print("\nTesting API server components...")

    try:
        from learning_api_server import app

        print("✅ API server app imported successfully")

        # Test basic endpoint
        from fastapi.testclient import TestClient

        client = TestClient(app)

        response = client.get("/")
        print(f"✅ API root endpoint working: {response.status_code}")

        return True

    except Exception as e:
        print(f"❌ API server test failed: {e}")
        return False


def test_environment():
    """Test environment setup."""
    print("\nTesting environment...")

    # Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")

    # Check if required files exist
    required_files = [
        "self_learning_system.py",
        "daily_prediction_system.py",
        "learning_api_server.py",
        "learning_integration.py",
    ]

    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")

    # Check environment variables
    env_vars = ["ODDS_API_KEY", "YOUTUBE_API_KEY", "SUPABASE_URL", "SUPABASE_ANON_KEY"]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is not set")


def main():
    """Run all tests."""
    print("🧪 Testing Daily Prediction System Components")
    print("=" * 50)

    # Test environment
    test_environment()

    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your Python environment.")
        return

    # Test learning system
    if not test_learning_system():
        print("\n❌ Learning system test failed.")
        return

    # Test daily prediction system
    if not test_daily_prediction_system():
        print("\n❌ Daily prediction system test failed.")
        return

    # Test API server
    if not test_api_server():
        print("\n❌ API server test failed.")
        return

    print("\n🎉 All tests passed! Your daily prediction system is ready to use.")
    print("\nNext steps:")
    print("1. Set your API keys in environment variables")
    print("2. Start the learning API server: python learning_api_server.py")
    print("3. Test daily predictions: python daily_prediction_system.py")
    print("4. Import the enhanced n8n workflow")


if __name__ == "__main__":
    main()
