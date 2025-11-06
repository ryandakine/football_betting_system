#!/usr/bin/env python3
"""
Simple Mock Gold Standard Bridge for testing n8n integration
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("simple_bridge")

# FastAPI app
app = FastAPI(title="Simple-GS-MLB", version="1.0.0", docs_url="/docs")


class OddsRow(BaseModel):
    game_id: str
    bookmaker: str
    home_team: str
    away_team: str
    commence_time: str
    home_price: int | None = None
    away_price: int | None = None


@app.get("/ping")
async def ping() -> dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok", "message": "Simple Gold Standard Bridge is running"}


@app.post("/opportunities")
async def opportunities(rows: list[OddsRow]) -> dict[str, Any]:
    """Mock AI analysis endpoint"""
    LOGGER.info(f"📥 Received {len(rows)} odds rows for analysis")

    # Mock AI analysis
    predictions = []
    for i, row in enumerate(rows[:3]):  # Limit to 3 predictions
        predictions.append(
            {
                "team": row.home_team if i % 2 == 0 else row.away_team,
                "bet_type": "moneyline",
                "confidence": 75 + (i * 5),  # Mock confidence scores
            }
        )

    # Mock recommended bets
    recommended_bets = (
        [{"description": f"{rows[0].home_team} ML", "edge": 8.5, "confidence": 78}]
        if rows
        else []
    )

    result = {
        "status": "analysis_complete",
        "confidence_score": 0.78,
        "predictions": predictions,
        "recommended_bets": recommended_bets,
        "analysis_time": datetime.now().isoformat(),
        "games_analyzed": len(rows),
    }

    LOGGER.info(f"🤖 Mock AI analysis complete: {len(predictions)} predictions")
    return result


@app.post("/sentiment")
async def sentiment(data: dict[str, Any]) -> dict[str, str]:
    """Mock sentiment analysis endpoint"""
    LOGGER.info(f"📊 Received sentiment data: {len(data)} items")
    return {
        "status": "sentiment_processed",
        "message": "Mock sentiment analysis complete",
    }


@app.post("/run-today")
async def run_today() -> dict[str, str]:
    """Mock full pipeline run"""
    LOGGER.info("🚀 Mock full pipeline run triggered")
    return {"status": "pipeline_complete", "message": "Mock pipeline run finished"}


if __name__ == "__main__":
    LOGGER.info("🚀 Starting Simple Gold Standard Bridge on port 8767")
    uvicorn.run(app, host="0.0.0.0", port=8767, log_level="info")
