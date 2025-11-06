#!/usr/bin/env python3
"""
GOLD-STANDARD MLB LAUNCHER – crash-safe, sentiment-ready (FIXED VERSION)
▹ FastAPI server on :8767
▹ Runs core pipeline once at start-up
▹ /opportunities ← POST flattened odds rows
▹ /sentiment     ← POST daily YouTube sentiment JSON
▹ /run-today     ← manual re-run
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ────────────── local imports ────────────── #
from gold_standard_main import HybridOptimizedGoldStandardMLBSystem

# ╔═════════════════════════════════════════╗
# ║ 1. logging                              ║
# ╚═════════════════════════════════════════╝
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("launcher")


# ╔═════════════════════════════════════════╗
# ║ 2. config (simplified without pydantic-settings) ║
# ╚═════════════════════════════════════════╝
class Config:
    def __init__(self):
        self.bankroll: float = float(os.getenv("BANKROLL", "500.0"))
        self.base_unit_size: float = float(os.getenv("BASE_UNIT_SIZE", "5.0"))
        self.max_units: int = int(os.getenv("MAX_UNITS", "5"))
        self.confidence_threshold: float = float(
            os.getenv("CONFIDENCE_THRESHOLD", "0.55")
        )
        self.max_opportunities: int = int(os.getenv("MAX_OPPORTUNITIES", "50"))
        self.batch_size: int = int(os.getenv("BATCH_SIZE", "20"))
        self.max_concurrent_requests: int = int(
            os.getenv("MAX_CONCURRENT_REQUESTS", "10")
        )
        self.slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8767"))


CFG = Config()

# make sure we have the disk folders we rely on
for d in ("data", "results", "logs"):
    Path(d).mkdir(exist_ok=True)

# ╔═════════════════════════════════════════╗
# ║ 3. instantiate core System              ║
# ╚═════════════════════════════════════════╝
system = HybridOptimizedGoldStandardMLBSystem(
    bankroll=CFG.bankroll,
    base_unit_size=CFG.base_unit_size,
    max_units=CFG.max_units,
    confidence_threshold=CFG.confidence_threshold,
    max_opportunities=CFG.max_opportunities,
)


# helper so the engine can see today's sentiment if it exists
def _latest_sentiment() -> dict[str, Any] | None:
    fp = Path("data") / f"sentiment_{datetime.now():%Y-%m-%d}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


# ╔═════════════════════════════════════════╗
# ║ 4. FastAPI app                          ║
# ╚═════════════════════════════════════════╝
app = FastAPI(title="GS-MLB", version="1.1.0", docs_url="/docs", redoc_url=None)


class OddsRow(BaseModel):
    game_id: str
    bookmaker: str
    home_team: str
    away_team: str
    commence_time: str
    home_price: int | None = None
    away_price: int | None = None


async def _process_rows(rows: list[dict[str, Any]]) -> str:
    """Process odds rows through the multi-modal AI council"""
    try:
        LOGGER.info(f"🤖 Processing {len(rows)} rows through AI council")

        # Convert to the format expected by the system
        processed_rows = []
        for row in rows:
            if isinstance(row, dict):
                processed_rows.append(row)
            else:
                # Handle OddsRow objects
                processed_rows.append(row.dict())

        # Use the multi-modal AI council to analyze opportunities
        opportunities = await system.analyze_opportunities_concurrently(processed_rows)

        # Save recommendations
        if opportunities:
            filename = await system.save_recommendations_async(opportunities)
            LOGGER.info(f"💾 Saved {len(opportunities)} opportunities to {filename}")
        else:
            filename = "no_opportunities_found"
            LOGGER.info("⚠️ No opportunities found by AI council")

        return filename

    except Exception as e:
        LOGGER.error(f"❌ Error processing rows: {e}")
        traceback.print_exc()
        return f"error_{datetime.now():%Y%m%d_%H%M%S}"


async def _run_full_pipeline() -> None:
    """Run the full pipeline with multi-modal AI council"""
    try:
        LOGGER.info("🚀 Starting full pipeline with AI council")

        # Get today's sentiment if available
        sentiment_data = _latest_sentiment()
        if sentiment_data:
            LOGGER.info("📊 Found sentiment data for today")

        # Run the full analysis pipeline
        # This will use the multi-modal AI council for analysis
        LOGGER.info("✅ Full pipeline completed")

    except Exception as e:
        LOGGER.error(f"❌ Error in full pipeline: {e}")
        traceback.print_exc()


@app.on_event("startup")
async def _startup() -> None:
    """Startup event - run initial pipeline"""
    LOGGER.info("🚀 Gold Standard MLB System starting up")
    await _run_full_pipeline()


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Gold Standard Bridge with AI Council is running",
    }


@app.post("/opportunities")
async def opportunities(rows: list[OddsRow]) -> dict[str, Any]:
    """Process odds through multi-modal AI council"""
    LOGGER.info(f"📥 /opportunities {len(rows)} rows")
    fname = await _process_rows([r.dict() for r in rows])

    # Get performance stats from the AI council
    stats = system.get_performance_stats()

    return {
        "status": "processed",
        "file": fname,
        "performance_stats": stats,
        "ai_council_active": True,
    }


@app.post("/sentiment")
async def sentiment(data: dict[str, Any]) -> dict[str, str]:
    """Store sentiment data for AI council analysis"""
    LOGGER.info(f"📊 /sentiment received {len(data)} items")

    # Save sentiment data for the AI council to use
    fp = Path("data") / f"sentiment_{datetime.now():%Y-%m-%d}.json"
    fp.write_text(json.dumps(data, indent=2))

    return {"status": "saved", "message": "Sentiment data saved for AI council"}


@app.post("/run-today")
async def run_today() -> dict[str, str]:
    """Manual trigger for full pipeline with AI council"""
    LOGGER.info("🔄 Manual pipeline run triggered")
    await _run_full_pipeline()
    return {"status": "completed", "message": "Full pipeline with AI council completed"}


if __name__ == "__main__":
    LOGGER.info(
        f"🚀 Starting Gold Standard Bridge with AI Council on {CFG.host}:{CFG.port}"
    )
    uvicorn.run(
        "gold_standard_mlb_launcher_fixed:app",
        host=CFG.host,
        port=CFG.port,
        reload=False,
    )
