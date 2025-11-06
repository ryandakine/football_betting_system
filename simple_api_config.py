#!/usr/bin/env python3
"""
Simple API Configuration for Fixed MLB System
"""

import logging
import os

logger = logging.getLogger(__name__)


def get_trimodel_api_keys():
    """Get API keys from environment variables"""
    api_keys = {
        "odds_api": os.getenv("THE_ODDS_API_KEY", os.getenv("ODDS_API_KEY", "missing")),
        "claude": os.getenv(
            "ANTHROPIC_API_KEY", os.getenv("CLAUDE_API_KEY", "missing")
        ),
        "openai": os.getenv("OPENAI_API_KEY", "missing"),
        "grok": os.getenv("GROK_API_KEY", "missing"),
    }

    # Log which keys are found
    for service, key in api_keys.items():
        if key != "missing":
            logger.info(f"✅ {service.upper()} API key loaded")
        else:
            logger.warning(f"⚠️ {service.upper()} API key missing")

    return api_keys


def get_alert_config():
    """Get alert configuration"""
    return {
        "enabled": True,
        "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
        "email_config": {
            "smtp_host": os.getenv("SMTP_HOST", ""),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "username": os.getenv("SMTP_USERNAME", ""),
            "password": os.getenv("SMTP_PASSWORD", ""),
            "from_email": os.getenv("SMTP_FROM_EMAIL", ""),
        },
    }


if __name__ == "__main__":
    # Test the configuration
    logging.basicConfig(level=logging.INFO)

    print("🔧 Testing API Configuration...")

    api_keys = get_trimodel_api_keys()
    alert_config = get_alert_config()

    valid_keys = sum(1 for key in api_keys.values() if key != "missing")

    print(f"\n📋 Configuration Summary:")
    print(f"   • Valid API Keys: {valid_keys}/4")
    print(f"   • Alert Config: {'Enabled' if alert_config['enabled'] else 'Disabled'}")

    if valid_keys < 2:
        print("\n⚠️  Warning: Not enough API keys configured")
        print("   • Check your .env file")
        print("   • Ensure environment variables are set correctly")
    else:
        print(f"\n✅ Configuration looks good! ({valid_keys}/4 APIs available)")
