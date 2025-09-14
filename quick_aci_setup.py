#!/usr/bin/env python3
"""
Quick ACI.dev Setup Script
"""

import json
import os
from pathlib import Path


def create_aci_config():
    """Create ACI.dev configuration files"""

    print("🚀 ACI.dev Configuration Setup")
    print("=" * 50)
    print("")

    # Create sample configuration
    config = {
        "THE_ODDS_API_KEY": "your_odds_api_key_here",
        "NEWSAPI_KEY": "your_newsapi_key_here",
        "TWITTER_API_KEY": "your_twitter_api_key_here",
        "SLACK_WEBHOOK_URL": "your_slack_webhook_url_here",
        "DISCORD_WEBHOOK_URL": "your_discord_webhook_url_here",
    }

    # Create .env file
    env_content = "# ACI.dev Configuration\n"
    env_content += "# Update these values with your actual API keys\n\n"

    for key, value in config.items():
        env_content += f"{key}={value}\n"

    with open(".env", "w") as f:
        f.write(env_content)

    # Create JSON config
    with open("aci_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Configuration files created!")
    print("")
    print("📁 Files created:")
    print("  - .env (environment variables)")
    print("  - aci_config.json (JSON configuration)")
    print("")
    print("🔧 Next steps:")
    print("1. Edit .env file with your actual API keys")
    print("2. Get API keys from:")
    print("   - The Odds API: https://the-odds-api.com/")
    print("   - NewsAPI: https://newsapi.org/")
    print("   - Twitter API: https://developer.twitter.com/")
    print("   - Slack: https://api.slack.com/")
    print("   - Discord: https://discord.com/developers/docs/")
    print("")
    print("💡 You can now use aci_integration.py in your betting system!")


if __name__ == "__main__":
    create_aci_config()
