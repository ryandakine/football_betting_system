#!/usr/bin/env python3
"""
Enhanced MLB Opportunity Detector Setup Script
==============================================
Quick setup and testing script for the MLB betting opportunity detector.
"""

import json
import os
from datetime import datetime

import requests

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpportunityDetectorSetup:
    """Setup and testing class for the MLB Opportunity Detector."""

    def __init__(self):
        self.api_keys = {}
        self.test_results = {}

    def check_environment_variables(self):
        """Check if all required environment variables are set."""
        print("🔍 Checking environment variables...")

        required_vars = [
            "YOUTUBE_API_KEY",
            "ODDS_API_KEY",
            "SLACK_WEBHOOK_URL",
            "SUPABASE_URL",
            "SUPABASE_ANON_KEY",
        ]

        missing_vars = []
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            else:
                self.api_keys[var] = value
                print(
                    f"  ✅ {var}: {'*' * (len(value) - 8) + value[-8:] if len(value) > 8 else 'Set'}"
                )

        if missing_vars:
            print(f"  ❌ Missing variables: {', '.join(missing_vars)}")
            return False

        print("  ✅ All environment variables are set!")
        return True

    def test_youtube_api(self):
        """Test YouTube API connection."""
        print("\n🎥 Testing YouTube API...")

        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": "MLB betting picks today",
                "type": "video",
                "maxResults": 1,
                "key": self.api_keys["YOUTUBE_API_KEY"],
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if "items" in data and len(data["items"]) > 0:
                    print(
                        f"  ✅ YouTube API working! Found {len(data['items'])} videos"
                    )
                    self.test_results["youtube"] = True
                    return True
                else:
                    print("  ⚠️  YouTube API working but no videos found")
                    self.test_results["youtube"] = True
                    return True
            else:
                print(f"  ❌ YouTube API error: {response.status_code}")
                self.test_results["youtube"] = False
                return False

        except Exception as e:
            print(f"  ❌ YouTube API test failed: {e}")
            self.test_results["youtube"] = False
            return False

    def test_odds_api(self):
        """Test Odds API connection."""
        print("\n🎲 Testing Odds API...")

        try:
            url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
            params = {
                "regions": "us",
                "markets": "h2h",
                "apiKey": self.api_keys["ODDS_API_KEY"],
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if len(data) > 0:
                    print(f"  ✅ Odds API working! Found {len(data)} games")
                    self.test_results["odds"] = True
                    return True
                else:
                    print(
                        "  ⚠️  Odds API working but no games found (might be off-season)"
                    )
                    self.test_results["odds"] = True
                    return True
            else:
                print(f"  ❌ Odds API error: {response.status_code}")
                self.test_results["odds"] = False
                return False

        except Exception as e:
            print(f"  ❌ Odds API test failed: {e}")
            self.test_results["odds"] = False
            return False

    def test_slack_webhook(self):
        """Test Slack webhook."""
        print("\n💬 Testing Slack webhook...")

        try:
            test_message = {
                "text": f"🧪 **MLB Opportunity Detector Test**\n\n✅ Setup test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nThis is a test message to verify your webhook is working correctly."
            }

            response = requests.post(
                self.api_keys["SLACK_WEBHOOK_URL"], json=test_message, timeout=10
            )

            if response.status_code == 200:
                print("  ✅ Slack webhook working! Test message sent.")
                self.test_results["slack"] = True
                return True
            else:
                print(f"  ❌ Slack webhook error: {response.status_code}")
                self.test_results["slack"] = False
                return False

        except Exception as e:
            print(f"  ❌ Slack webhook test failed: {e}")
            self.test_results["slack"] = False
            return False

    def test_supabase_connection(self):
        """Test Supabase connection."""
        print("\n🗄️  Testing Supabase connection...")

        try:
            url = f"{self.api_keys['SUPABASE_URL']}/rest/v1/sentiment_data"
            headers = {
                "apikey": self.api_keys["SUPABASE_ANON_KEY"],
                "Authorization": f"Bearer {self.api_keys['SUPABASE_ANON_KEY']}",
                "Content-Type": "application/json",
            }

            # Test with a simple query
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code in [
                200,
                404,
            ]:  # 404 is OK if table doesn't exist yet
                print("  ✅ Supabase connection working!")
                self.test_results["supabase"] = True
                return True
            else:
                print(f"  ❌ Supabase connection error: {response.status_code}")
                self.test_results["supabase"] = False
                return False

        except Exception as e:
            print(f"  ❌ Supabase connection test failed: {e}")
            self.test_results["supabase"] = False
            return False

    def create_env_template(self):
        """Create a .env template file."""
        print("\n📝 Creating .env template...")

        template = """# Enhanced MLB Opportunity Detector Environment Variables
# Copy this to .env and fill in your actual values

# API Keys
YOUTUBE_API_KEY=your_youtube_api_key_here
ODDS_API_KEY=your_odds_api_key_here
SLACK_WEBHOOK_URL=your_slack_webhook_url_here

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# Optional Settings
SUPABASE_BATCH_SIZE=100
SUPABASE_MAX_RETRIES=3
SUPABASE_RETRY_DELAY=1.0
"""

        with open(".env.template", "w") as f:
            f.write(template)

        print("  ✅ Created .env.template file")
        print("  📋 Copy this to .env and fill in your actual API keys")

    def run_all_tests(self):
        """Run all API tests."""
        print("🚀 Starting Enhanced MLB Opportunity Detector Setup...\n")

        # Check environment variables
        if not self.check_environment_variables():
            print("\n❌ Please set up your environment variables first!")
            self.create_env_template()
            return False

        # Run API tests
        tests = [
            self.test_youtube_api,
            self.test_odds_api,
            self.test_slack_webhook,
            self.test_supabase_connection,
        ]

        all_passed = True
        for test in tests:
            if not test():
                all_passed = False

        # Print summary
        print("\n" + "=" * 50)
        print("📊 SETUP SUMMARY")
        print("=" * 50)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.upper()}: {status}")

        if all_passed:
            print("\n🎉 All tests passed! Your setup is ready.")
            print("\n📋 Next steps:")
            print("1. Import the workflow into n8n")
            print("2. Configure the Supabase nodes with your credentials")
            print("3. Activate the workflow")
            print("4. Check your Slack channel for alerts!")
        else:
            print("\n⚠️  Some tests failed. Please check the errors above.")

        return all_passed

    def generate_n8n_config(self):
        """Generate n8n configuration instructions."""
        print("\n🔧 n8n Configuration Instructions:")
        print("=" * 50)

        config = f"""
1. Import the workflow file: n8n-workflows/enhanced-mlb-opportunity-detector.json

2. Configure Supabase nodes:
   - URL: {self.api_keys.get('SUPABASE_URL', 'YOUR_SUPABASE_URL')}
   - API Key: {self.api_keys.get('SUPABASE_ANON_KEY', 'YOUR_SUPABASE_ANON_KEY')}

3. Configure Slack node:
   - Webhook URL: {self.api_keys.get('SLACK_WEBHOOK_URL', 'YOUR_SLACK_WEBHOOK')}
   - Channel: mlb-opportunities

4. Test the workflow by clicking "Execute Workflow"

5. Activate the workflow when ready
"""

        print(config)

        # Save config to file
        with open("n8n_config_instructions.txt", "w") as f:
            f.write(config)

        print("✅ Saved n8n configuration instructions to n8n_config_instructions.txt")


def main():
    """Main setup function."""
    setup = OpportunityDetectorSetup()

    # Run all tests
    success = setup.run_all_tests()

    if success:
        # Generate n8n config
        setup.generate_n8n_config()

        print("\n🎯 Your Enhanced MLB Opportunity Detector is ready to deploy!")
        print("📖 Check QUICK_SETUP_GUIDE.md for detailed instructions")
    else:
        print("\n🔧 Please fix the failed tests before proceeding")


if __name__ == "__main__":
    main()
