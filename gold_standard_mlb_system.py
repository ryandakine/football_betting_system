#!/usr/bin/env python3
# gold_standard_mlb_launcher.py
"""
🏆  Gold-Standard MLB Betting System – **Launcher v2.1.6**

What changed?
──────────────
• Fixed API key passing to system
• Added proper imports and error handling
• Fixed syntax errors and structure
• Added debug output for API keys
• Robust error handling throughout
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from pydantic_settings import BaseSettings
from tqdm import tqdm


# Configuration class to load from .env
class SystemConfig(BaseSettings):
    FULL_ANALYSIS: bool = True
    MAX_OPPORTUNITIES: int = 690
    BATCH_SIZE: int = 20
    MAX_CONCURRENT_REQUESTS: int = 5
    BANKROLL: float = 500.0
    BASE_UNIT_SIZE: float = 5.0
    MAX_UNITS: int = 5
    CONFIDENCE_THRESHOLD: float = 0.55

    # Allow extra fields from .env
    ODDS_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GROK_API_KEY: str = ""
    SLACK_WEBHOOK_URL: str = ""
    SMTP_HOST: str = ""
    SMTP_PORT: str = ""
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = ""
    ALERT_TO_EMAILS: str = ""
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_PHONE: str = ""
    ALERT_TO_PHONES: str = ""
    LOG_LEVEL: str = "INFO"
    MIN_CONFIDENCE_THRESHOLD: float = 0.65

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "GSMLB_"
        extra = "allow"

# Load config
config = SystemConfig()

def setup_logging() -> logging.Logger:
    """File + console logger – always at INFO level."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = log_dir / f"launcher_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(logfile, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("launcher")
    logger.info("Log file → %s", logfile)
    return logger

def print_banner() -> None:
    banner = f"""
╔═══════════════════════════════════════════════════════════════════╗
║  🏆  GOLD-STANDARD MLB BETTING SYSTEM  –  FULL MODE  (2025-06-18)  ║
║  Opportunities {config.MAX_OPPORTUNITIES}  │ Batch {config.BATCH_SIZE} │ Parallel {config.MAX_CONCURRENT_REQUESTS}                       ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(banner, flush=True)

async def main() -> int:
    print_banner()
    logger = setup_logging()

    # Ensure folders exist
    for d in ("results", "data", "analysis", "reports"):
        Path(d).mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    logger.info("System ready • date=%s  max=%d  batch=%d  parallel=%d",
                start.strftime("%Y-%m-%d"), config.MAX_OPPORTUNITIES, config.BATCH_SIZE, config.MAX_CONCURRENT_REQUESTS)

    rc = 1  # Default exit code
    try:
        # Import system components
        from gold_standard_main import GoldStandardMLBSystem
        from tri_model_api_config import get_alert_config, get_trimodel_api_keys

        # Set environment variables for the API config
        os.environ["ODDS_API_KEY"] = config.ODDS_API_KEY
        os.environ["ANTHROPIC_API_KEY"] = config.ANTHROPIC_API_KEY
        os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
        os.environ["GROK_API_KEY"] = config.GROK_API_KEY

        # Get API keys and config
        api_keys = get_trimodel_api_keys()
        alert_config = get_alert_config()

        # Debug output to see what keys we have
        logger.info("DEBUG: Available API keys = %s", list(api_keys.keys()))
        logger.info("DEBUG: API key values = %s", {k: "***" if v else "MISSING" for k, v in api_keys.items()})

        # Initialize the system
        system = GoldStandardMLBSystem(
            bankroll=config.BANKROLL,
            base_unit_size=config.BASE_UNIT_SIZE,
            max_units=config.MAX_UNITS,
            confidence_threshold=config.CONFIDENCE_THRESHOLD
        )

        # Pass configurations to the system
        system.alert_config = alert_config
        # Add after line 136 where you set system.api_keys = api_keys:
        system.api_keys = api_keys
        logger.info("DEBUG: API keys passed to system = %s", {k: "***" if v else "MISSING" for k, v in api_keys.items()})

        logger.info("Running pipeline …")
        logger.info("Fetching odds for %s …", start.strftime("%Y-%m-%d"))

        # Run the main pipeline
        opportunities = await system.run_gold_standard_pipeline()

        # Handle opportunities if they exist
        if not opportunities:
            logger.warning("No opportunities fetched")
            print("[WARN] No opportunities fetched", flush=True)
            return 1

        total_opportunities = len(opportunities)
        logger.info("Pulled %d raw opportunities", total_opportunities)

        # Check if system has analyze_opportunities_concurrently method
        if hasattr(system, 'analyze_opportunities_concurrently'):
            logger.info("🧠 Analyzing %d opportunities (batch %d, parallel %d)",
                        total_opportunities, config.BATCH_SIZE, config.MAX_CONCURRENT_REQUESTS)

            # Analyze opportunities with progress bar
            recommendations = []
            with tqdm(total=total_opportunities, desc="Analyzing Opportunities", unit="opps") as pbar:
                opps_list = opportunities if isinstance(opportunities, list) else list(opportunities.get('opportunities', []))
total_opportunities = len(opps_list)
for i in range(0, total_opportunities, config.BATCH_SIZE):
    batch = opps_list[i:i + config.BATCH_SIZE]
                    try:
                        batch_recs = await system.analyze_opportunities_concurrently(batch)
                        recommendations.extend(batch_recs)
                    except Exception as e:
                        logger.error("Batch analysis error: %s", e)
                        # Continue with next batch
                    pbar.update(len(batch))

            logger.info("🏁 Produced %d recommendations", len(recommendations))
        else:
            logger.info("System pipeline completed successfully")
            recommendations = []

        # Save results
        results_data = {
            "opportunities": opportunities,
            "recommendations": recommendations,
            "total_opportunities": total_opportunities,
            "analysis_timestamp": datetime.now().isoformat(),
            "config": {
                "bankroll": config.BANKROLL,
                "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                "batch_size": config.BATCH_SIZE
            }
        }

        out = Path("results") / f"results_{datetime.now():%Y-%m-%d_%H%M%S}.json"
        try:
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, default=str)
            logger.info("Saved results → %s", out)
        except Exception as e:
            logger.error("Could not write results file: %s", e)
            print("[ERROR] Failed to save results", flush=True)
            return 1

        elapsed = (datetime.now() - start).total_seconds()
        logger.info("Total execution time: %.1f s (%.1f min)", elapsed, elapsed / 60)
        rc = 0  # Success

    except ModuleNotFoundError as exc:
        logger.error("Missing dependency: %s", exc)
        print(f"[ERROR] Module not found: {exc}", flush=True)
    except Exception as e:
        logger.error("Unexpected error during execution", exc_info=True)
        print(f"\n[FATAL] Unhandled exception: {e}", flush=True)
        traceback.print_exc()

    finally:
        elapsed = (datetime.now() - start).total_seconds()
        logger.info("Total execution time: %.1f s (%.1f min)", elapsed, elapsed / 60)
        print(f"[EXIT] code={rc}", flush=True)

    return rc

if __name__ == "__main__":
    try:
        print(f"[START] {datetime.now():%H:%M:%S} – launching", flush=True)
        rc = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User aborted.", flush=True)
        rc = 1
    except Exception:
        print("\n[FATAL] Unhandled exception – see below", flush=True)
        traceback.print_exc()
        rc = 1
    finally:
        print(f"[EXIT] code={rc}", flush=True)
        sys.exit(rc)
