# api_config.py

import logging
import os

from dotenv import load_dotenv

# Configure logging for this module
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_api_keys() -> dict:
    """
    Loads all required API keys from the .env file in the project root.

    Your .env file should look like this:
    ODDS_API_KEY = "aa49772bf36d88bf4962faa14015d882""
    CLAUDE_API_KEY="your_claude_api_key_here"
    OPENAI_API_KEY="your_openai_api_key_here"
    GROK_API_KEY="your_grok_api_key_here"

    Returns:
        A dictionary containing the API keys.
    """
    # Find the .env file in the same directory as this script or parent directories
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(dotenv_path):
        # Fallback for running from different directories
        dotenv_path = os.path.join(os.getcwd(), ".env")

    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        logger.error(".env file not found. API keys will not be loaded.")
        return {}

    keys = {
        "odds_api": os.getenv("THE_ODDS_API_KEY"),
        "claude": os.getenv("CLAUDE_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "grok": os.getenv("GROK_API_KEY"),
    }

    # Verify that keys were loaded
    for service, key in keys.items():
        if not key:
            logger.warning(f"⚠️  API Key for '{service}' not found in .env file.")
        else:
            # Only log the service name, not the key itself for security
            logger.info(f"✅ API Key for '{service}' loaded.")

    return keys


if __name__ == "__main__":
    # A simple test to verify the function works when run directly
    print("Testing API key loader...")
    loaded_keys = get_api_keys()
    if loaded_keys.get("odds_api"):
        print("Successfully loaded the Odds API key.")
    if loaded_keys.get("claude"):
        print("Successfully loaded the Claude API key.")
