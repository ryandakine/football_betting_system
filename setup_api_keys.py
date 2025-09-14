#!/usr/bin/env python3
"""
Setup script for API keys
"""

import json
import os
from pathlib import Path


def setup_api_keys():
    """Interactive setup for API keys."""
    print("🔑 Setting up API Keys for Daily Prediction System")
    print("=" * 50)

    # Check if aci.env exists
    env_file = Path("aci.env")
    if env_file.exists():
        print("✅ Found aci.env file")

        # Read existing keys
        existing_keys = {}
        with open(env_file) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    existing_keys[key] = value

        print(f"Found {len(existing_keys)} existing environment variables")
    else:
        print("⚠️  No aci.env file found. Creating new one...")
        existing_keys = {}

    # Required API keys
    required_keys = {
        "ODDS_API_KEY": "The Odds API key (get from https://the-odds-api.com/)",
        "YOUTUBE_API_KEY": "YouTube Data API key (get from https://console.developers.google.com/)",
        "SUPABASE_URL": "Supabase project URL",
        "SUPABASE_ANON_KEY": "Supabase anonymous key",
        "OPENAI_API_KEY": "OpenAI API key (for AI analysis)",
        "SLACK_WEBHOOK_URL": "Slack webhook URL (optional, for notifications)",
    }

    # Check which keys are missing
    missing_keys = []
    for key, description in required_keys.items():
        if key not in existing_keys or not existing_keys[key]:
            missing_keys.append((key, description))

    if not missing_keys:
        print("🎉 All required API keys are already set!")
        return

    print(f"\nMissing {len(missing_keys)} API keys:")
    for key, description in missing_keys:
        print(f"  - {key}: {description}")

    print("\n" + "=" * 50)
    print("Please provide the missing API keys:")
    print("(Press Enter to skip if you don't have the key yet)")

    # Collect missing keys
    new_keys = {}
    for key, description in missing_keys:
        print(f"\n{description}")
        value = input(f"Enter {key}: ").strip()
        if value:
            new_keys[key] = value
        else:
            print(f"⚠️  Skipping {key}")

    # Update aci.env file
    if new_keys:
        print(f"\nUpdating aci.env with {len(new_keys)} new keys...")

        # Read existing content
        lines = []
        if env_file.exists():
            with open(env_file) as f:
                lines = f.readlines()

        # Add new keys
        for key, value in new_keys.items():
            # Check if key already exists
            key_exists = False
            for i, line in enumerate(lines):
                if line.startswith(f"{key}="):
                    lines[i] = f"{key}={value}\n"
                    key_exists = True
                    break

            if not key_exists:
                lines.append(f"{key}={value}\n")

        # Write back to file
        with open(env_file, "w") as f:
            f.writelines(lines)

        print("✅ aci.env updated successfully!")
    else:
        print("⚠️  No new keys provided.")

    # Test the keys
    print("\n" + "=" * 50)
    print("Testing API keys...")

    test_results = {}
    for key, value in new_keys.items():
        if key == "ODDS_API_KEY":
            test_results[key] = test_odds_api(value)
        elif key == "YOUTUBE_API_KEY":
            test_results[key] = test_youtube_api(value)
        elif key == "SUPABASE_URL" and "SUPABASE_ANON_KEY" in new_keys:
            test_results[key] = test_supabase(
                new_keys["SUPABASE_URL"], new_keys["SUPABASE_ANON_KEY"]
            )
        else:
            test_results[key] = "✅ Key provided (not tested)"

    # Display test results
    print("\nAPI Key Test Results:")
    for key, result in test_results.items():
        print(f"  {key}: {result}")


def test_odds_api(api_key):
    """Test The Odds API key."""
    try:
        import requests

        url = "https://api.the-odds-api.com/v4/sports"
        response = requests.get(url, params={"apiKey": api_key})

        if response.status_code == 200:
            return "✅ Valid API key"
        elif response.status_code == 401:
            return "❌ Invalid API key"
        else:
            return f"⚠️  API error: {response.status_code}"
    except Exception as e:
        return f"❌ Test failed: {e}"


def test_youtube_api(api_key):
    """Test YouTube API key."""
    try:
        import requests

        url = "https://www.googleapis.com/youtube/v3/search"
        params = {"part": "snippet", "q": "test", "maxResults": 1, "key": api_key}
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return "✅ Valid API key"
        elif response.status_code == 400:
            return "❌ Invalid API key"
        else:
            return f"⚠️  API error: {response.status_code}"
    except Exception as e:
        return f"❌ Test failed: {e}"


def test_supabase(url, key):
    """Test Supabase connection."""
    try:
        import requests

        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        response = requests.get(f"{url}/rest/v1/", headers=headers)

        if response.status_code == 200:
            return "✅ Valid connection"
        else:
            return f"⚠️  Connection error: {response.status_code}"
    except Exception as e:
        return f"❌ Test failed: {e}"


def main():
    """Main setup function."""
    setup_api_keys()

    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Run: python test_daily_system.py")
    print("2. Start the learning API server: python learning_api_server.py")
    print("3. Test daily predictions: python daily_prediction_system.py")


if __name__ == "__main__":
    main()
