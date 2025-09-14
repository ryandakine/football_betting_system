#!/usr/bin/env python3
"""
Interactive ACI.dev Setup Script
"""

import json
import os
from pathlib import Path


def setup_aci_dev():
    """Interactive setup for ACI.dev configuration"""

    print("🚀 ACI.dev Configuration Setup")
    print("=" * 50)
    print("")

    config = {}

    # Sports Data APIs
    print("📊 Sports Data APIs:")
    print("1. The Odds API - https://the-odds-api.com/")
    print("2. ESPN API - https://developer.espn.com/")
    print("3. Sportradar API - https://www.sportradar.com/")
    print("")

    odds_api_key = input("Enter The Odds API key (or press Enter to skip): ").strip()
    if odds_api_key:
        config["THE_ODDS_API_KEY"] = odds_api_key

    # News APIs
    print("")
    print("📰 News APIs:")
    print("1. NewsAPI - https://newsapi.org/")
    print("2. GNews - https://gnews.io/")
    print("")

    news_api_key = input("Enter NewsAPI key (or press Enter to skip): ").strip()
    if news_api_key:
        config["NEWSAPI_KEY"] = news_api_key

    # Social Media APIs
    print("")
    print("📱 Social Media APIs:")
    print("1. Twitter API - https://developer.twitter.com/")
    print("2. Reddit API - https://www.reddit.com/dev/api/")
    print("")

    twitter_api_key = input("Enter Twitter API key (or press Enter to skip): ").strip()
    if twitter_api_key:
        config["TWITTER_API_KEY"] = twitter_api_key

    # Notification Services
    print("")
    print("🔔 Notification Services:")
    print("1. Slack - https://api.slack.com/")
    print("2. Discord - https://discord.com/developers/docs/")
    print("")

    slack_webhook = input("Enter Slack webhook URL (or press Enter to skip): ").strip()
    if slack_webhook:
        config["SLACK_WEBHOOK_URL"] = slack_webhook

    discord_webhook = input(
        "Enter Discord webhook URL (or press Enter to skip): "
    ).strip()
    if discord_webhook:
        config["DISCORD_WEBHOOK_URL"] = discord_webhook

    # Save configuration
    if config:
        # Save to .env file
        env_content = "# ACI.dev Configuration\n"
        for key, value in config.items():
            env_content += f"{key}={value}\n"

        with open(".env", "w") as f:
            f.write(env_content)

        # Save to JSON for programmatic access
        with open("aci_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print("")
        print("✅ Configuration saved!")
        print(f"📁 Files created: .env, aci_config.json")
        print(f"🔧 Configured {len(config)} API keys/services")

        # Show summary
        print("")
        print("📋 Configuration Summary:")
        for key, value in config.items():
            masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"  {key}: {masked_value}")
    else:
        print("")
        print("ℹ️ No API keys configured. You can edit aci.env manually later.")

    print("")
    print("🎉 ACI.dev setup complete!")
    print("💡 You can now use the aci_integration.py module in your betting system.")


if __name__ == "__main__":
    setup_aci_dev()
