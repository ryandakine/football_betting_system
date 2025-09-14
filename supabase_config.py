#!/usr/bin/env python3
"""
Supabase Configuration for MLB Betting System
=============================================
Configuration settings and environment variables for Supabase integration.
"""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SupabaseConfig:
    """Configuration class for Supabase integration."""

    # Supabase connection settings
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    # Table names
    TABLES = {
        "AI_PREDICTIONS": "ai_predictions",
        "ENSEMBLE_CONSENSUS": "ensemble_consensus",
        "DAILY_PORTFOLIOS": "daily_portfolios",
        "AI_PERFORMANCE": "ai_performance",
        "LEARNING_METRICS": "learning_metrics",
        "RECOMMENDATIONS": "recommendations",
        "ANALYSIS_HISTORY": "analysis_history",
        "PROFESSIONAL_BETS": "professional_bets",
        "DAILY_PERFORMANCE": "daily_performance",
        "UNIT_BETS": "unit_bets",
        "BETS": "bets",
        "RESULTS": "results",
        "METRICS": "metrics",
        "ODDS_DATA": "odds_data",
        "SENTIMENT_DATA": "sentiment_data",
    }

    # Batch processing settings
    BATCH_SIZE = int(os.getenv("SUPABASE_BATCH_SIZE", "100"))
    MAX_RETRIES = int(os.getenv("SUPABASE_MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("SUPABASE_RETRY_DELAY", "1.0"))

    # Connection settings
    CONNECTION_TIMEOUT = int(os.getenv("SUPABASE_TIMEOUT", "30"))

    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        missing_vars = []

        if not cls.SUPABASE_URL:
            missing_vars.append("SUPABASE_URL")
        if not cls.SUPABASE_ANON_KEY:
            missing_vars.append("SUPABASE_ANON_KEY")

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        return True

    @classmethod
    def get_connection_string(cls):
        """Get formatted connection string for logging."""
        if cls.SUPABASE_URL:
            return f"Supabase: {cls.SUPABASE_URL}"
        return "Supabase: Not configured"

    @classmethod
    def print_config(cls):
        """Print current configuration (without sensitive data)."""
        print("🔧 Supabase Configuration:")
        print(f"  URL: {cls.SUPABASE_URL or 'Not set'}")
        print(f"  Anon Key: {'Set' if cls.SUPABASE_ANON_KEY else 'Not set'}")
        print(
            f"  Service Role Key: {'Set' if cls.SUPABASE_SERVICE_ROLE_KEY else 'Not set'}"
        )
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Max Retries: {cls.MAX_RETRIES}")
        print(f"  Connection Timeout: {cls.CONNECTION_TIMEOUT}s")


# Environment variable template
ENV_TEMPLATE = """
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here

# Optional Settings
SUPABASE_BATCH_SIZE=100
SUPABASE_MAX_RETRIES=3
SUPABASE_RETRY_DELAY=1.0
SUPABASE_TIMEOUT=30
"""


def create_env_template():
    """Create a .env template file."""
    with open(".env.template", "w") as f:
        f.write(ENV_TEMPLATE)
    print("📝 Created .env.template file")


if __name__ == "__main__":
    # Print configuration
    SupabaseConfig.print_config()

    # Validate configuration
    try:
        SupabaseConfig.validate_config()
        print("✅ Configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nTo fix this, create a .env file with the required variables:")
        create_env_template()
