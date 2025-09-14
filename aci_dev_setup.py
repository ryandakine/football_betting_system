"""
ACI.dev Configuration and Setup for MLB Betting System
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


class ACIDevConfig:
    """ACI.dev configuration manager"""

    def __init__(self, config_file: str = "aci_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        return {}

    def _save_config(self):
        """Save configuration to file"""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def set_api_key(self, service: str, api_key: str):
        """Set API key for a specific service"""
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}

        self.config["api_keys"][service] = api_key
        self._save_config()
        print(f"✅ API key set for {service}")

    def get_api_key(self, service: str) -> str | None:
        """Get API key for a specific service"""
        return self.config.get("api_keys", {}).get(service)

    def set_webhook_url(self, service: str, webhook_url: str):
        """Set webhook URL for notifications"""
        if "webhooks" not in self.config:
            self.config["webhooks"] = {}

        self.config["webhooks"][service] = webhook_url
        self._save_config()
        print(f"✅ Webhook URL set for {service}")

    def get_webhook_url(self, service: str) -> str | None:
        """Get webhook URL for a specific service"""
        return self.config.get("webhooks", {}).get(service)

    def list_configured_services(self):
        """List all configured services"""
        print("🔧 Configured Services:")
        print("")

        if "api_keys" in self.config:
            print("📡 API Keys:")
            for service, key in self.config["api_keys"].items():
                masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
                print(f"  {service}: {masked_key}")

        if "webhooks" in self.config:
            print("")
            print("🔔 Webhooks:")
            for service, url in self.config["webhooks"].items():
                print(f"  {service}: {url}")

        if not self.config:
            print("  No services configured yet.")


def setup_aci_dev():
    """Interactive setup for ACI.dev configuration"""

    print("🚀 ACI.dev Configuration Setup")
    print("=" * 50)
    print("")

    config = ACIDevConfig()

    # Sports Data APIs
    print("📊 Sports Data APIs:")
    print("1. The Odds API - https://the-odds-api.com/")
    print("2. ESPN API - https://developer.espn.com/")
    print("3. Sportradar API - https://www.sportradar.com/")
    print("")

    odds_api_key = input("Enter The Odds API key (or press Enter to skip): ").strip()
    if odds_api_key:
        config.set_api_key("the_odds_api", odds_api_key)

    # News APIs
    print("")
    print("📰 News APIs:")
    print("1. NewsAPI - https://newsapi.org/")
    print("2. GNews - https://gnews.io/")
    print("")

    news_api_key = input("Enter NewsAPI key (or press Enter to skip): ").strip()
    if news_api_key:
        config.set_api_key("newsapi", news_api_key)

    # Social Media APIs
    print("")
    print("📱 Social Media APIs:")
    print("1. Twitter API - https://developer.twitter.com/")
    print("2. Reddit API - https://www.reddit.com/dev/api/")
    print("")

    twitter_api_key = input("Enter Twitter API key (or press Enter to skip): ").strip()
    if twitter_api_key:
        config.set_api_key("twitter", twitter_api_key)

    # Notification Services
    print("")
    print("🔔 Notification Services:")
    print("1. Slack - https://api.slack.com/")
    print("2. Discord - https://discord.com/developers/docs/")
    print("3. Email (SMTP)")
    print("")

    slack_webhook = input("Enter Slack webhook URL (or press Enter to skip): ").strip()
    if slack_webhook:
        config.set_webhook_url("slack", slack_webhook)

    discord_webhook = input(
        "Enter Discord webhook URL (or press Enter to skip): "
    ).strip()
    if discord_webhook:
        config.set_webhook_url("discord", discord_webhook)

    # Email Configuration
    print("")
    print("📧 Email Configuration:")
    smtp_server = input("Enter SMTP server (e.g., smtp.gmail.com): ").strip()
    if smtp_server:
        config.config.setdefault("email", {})["smtp_server"] = smtp_server

        smtp_port = input("Enter SMTP port (e.g., 587): ").strip()
        if smtp_port:
            config.config["email"]["smtp_port"] = int(smtp_port)

        email_username = input("Enter email username: ").strip()
        if email_username:
            config.config["email"]["username"] = email_username

        email_password = input("Enter email password/app password: ").strip()
        if email_password:
            config.config["email"]["password"] = email_password

        config._save_config()

    print("")
    print("✅ ACI.dev configuration complete!")
    print("")
    config.list_configured_services()

    return config


def create_aci_env_file():
    """Create .env file for ACI.dev configuration"""

    env_content = """# ACI.dev Configuration
# Copy this to your .env file and update with your actual values

# Sports Data APIs
THE_ODDS_API_KEY=your_odds_api_key_here
ESPN_API_KEY=your_espn_api_key_here
SPORTRADAR_API_KEY=your_sportradar_api_key_here

# News APIs
NEWSAPI_KEY=your_newsapi_key_here
GNEWS_API_KEY=your_gnews_api_key_here

# Social Media APIs
TWITTER_API_KEY=your_twitter_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here

# Notification Services
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here

# ACI.dev Core
ACI_API_KEY=your_aci_api_key_here
ACI_BASE_URL=https://api.aci.dev
"""

    with open("aci.env", "w") as f:
        f.write(env_content)

    print("✅ Created aci.env file")
    print("📝 Please update the values in aci.env with your actual API keys")


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Interactive setup")
    print("2. Create .env template")
    print("3. List current configuration")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        setup_aci_dev()
    elif choice == "2":
        create_aci_env_file()
    elif choice == "3":
        config = ACIDevConfig()
        config.list_configured_services()
    else:
        print("Invalid choice. Please run the script again.")
