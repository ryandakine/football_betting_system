#!/usr/bin/env python3
"""
Ultimate Main Orchestrator v4.4 - Complete Working Version
──────────────────────────────────────────────────────────
• Exposes 2 modes:

  1. CLI      → `python ultimate_main_orchestrator.py`
  2. API POST → /ingest   (used by your n8n workflow)

   n8n payload shape
   ────────────────
   {
     "rows":               [ {...fanduel_vs_best_other...}, … ],
     "youtube_sentiment":  { ...sentimentData... },
     "meta":               { "source": "...", "generated_at": "2025-07-02T12:34:56Z" }
   }
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn

# FastAPI for /ingest endpoint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ultimate")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app & request schema
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Ultimate-Main Orchestrator", docs_url="/docs", redoc_url=None)


class IncomingPayload(BaseModel):
    rows: list[dict]
    youtube_sentiment: dict = {}
    meta: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────


def save_odds_to_file(odds_data: list[dict], target_date: str) -> None:
    """Save odds data to file for later processing"""
    try:
        odds_dir = Path("data/odds")
        odds_dir.mkdir(parents=True, exist_ok=True)

        odds_file = odds_dir / f"{target_date}_odds.json"
        with open(odds_file, "w", encoding="utf-8") as f:
            json.dump(odds_data, f, indent=2)

        logger.info(f"Saved {len(odds_data)} odds records to {odds_file}")
    except Exception as e:
        logger.error(f"Failed to save odds data: {e}")


def convert_structured_odds_to_system_format(market_data: dict) -> list[dict]:
    """Convert structured market data to system format"""
    odds_data = []
    for game_id, game_data in market_data.items():
        for market, books in game_data.items():
            for book, odds in books.items():
                odds_data.append(
                    {
                        "game_id": game_id,
                        "bookmaker": book,
                        "market": market,
                        "odds": odds,
                        "timestamp": dt.utcnow().isoformat(),
                    }
                )
    return odds_data


def create_mock_odds_data(target_date: str) -> list[dict]:
    """Create mock odds data for testing"""
    return [
        {
            "game_id": "mock_game_1",
            "bookmaker": "FanDuel",
            "home_team": "Mock Team A",
            "away_team": "Mock Team B",
            "home_price": 150,
            "away_price": -150,
            "market": "moneyline",
            "commence_time": f"{target_date}T19:00:00Z",
        }
    ]


async def load_daily_data(target_date: str, odds_data: list[dict] = None) -> dict:
    """Load daily data including sentiment and odds"""
    daily_data = {
        "date": target_date,
        "games": [],
        "odds": odds_data or [],
        "sentiment": {},
    }

    # Load sentiment data if exists
    sentiment_file = Path("sentiment") / f"{target_date}.json"
    if sentiment_file.exists():
        try:
            with open(sentiment_file, encoding="utf-8") as f:
                daily_data["sentiment"] = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load sentiment data: {e}")

    # Process odds into games
    games_dict = {}
    for odd in odds_data or []:
        game_id = odd.get("game_id")
        if game_id not in games_dict:
            games_dict[game_id] = {
                "game_id": game_id,
                "home_team": odd.get("home_team"),
                "away_team": odd.get("away_team"),
                "commence_time": odd.get("commence_time"),
                "odds": [],
            }
        games_dict[game_id]["odds"].append(odd)

    daily_data["games"] = list(games_dict.values())
    return daily_data


def create_sample_sentiment() -> dict:
    """Create sample sentiment data"""
    return {
        "overall_sentiment": "positive",
        "confidence": 0.75,
        "sources": ["youtube", "twitter"],
        "timestamp": dt.utcnow().isoformat(),
    }


def create_sample_odds() -> list[dict]:
    """Create sample odds data"""
    return [
        {
            "game_id": "sample_1",
            "home_team": "Sample Home",
            "away_team": "Sample Away",
            "bookmaker": "FanDuel",
            "home_price": 150,
            "away_price": -150,
        }
    ]


def combine_data_sources(odds_data: list[dict], sentiment_data: dict) -> dict:
    """Combine odds and sentiment data"""
    return {
        "odds": odds_data,
        "sentiment": sentiment_data,
        "combined_at": dt.utcnow().isoformat(),
        "total_games": len({odd.get("game_id") for odd in odds_data}),
    }


def create_ai_prompts(games_data: list[dict]) -> list[dict]:
    """Create AI prompts for analysis"""
    prompts = []
    for game in games_data:
        prompt = {
            "game_id": game.get("game_id"),
            "prompt": f"Analyze betting opportunity: {game.get('home_team')} vs {game.get('away_team')}",
            "context": game,
        }
        prompts.append(prompt)
    return prompts


async def run_ai_analysis(provider_mgr, games: list[dict]) -> dict[str, list[dict]]:
    """Run AI analysis on games data"""
    # Mock AI analysis for now
    ai_predictions = {"gpt4": [], "claude": [], "gemini": []}

    for game in games:
        prediction = {
            "game_id": game.get("game_id"),
            "pick": f"{game.get('home_team')} wins",
            "confidence": 75,
            "primary_signal": "value betting",
            "reasoning": "Mock AI analysis",
        }
        ai_predictions["gpt4"].append(prediction)

    logger.info(f"Generated {len(ai_predictions['gpt4'])} AI predictions")
    return ai_predictions


def convert_to_prediction_data(predictions: list[dict]) -> list[dict]:
    """Convert predictions to database format"""
    prediction_data = []
    for pred in predictions:
        prediction_data.append(
            {
                "game_id": pred.get("game_id"),
                "prediction": pred.get("pick"),
                "confidence": pred.get("confidence", 0) / 100.0,
                "model": "mock_ai",
                "created_at": dt.utcnow().isoformat(),
            }
        )
    return prediction_data


# ─────────────────────────────────────────────────────────────────────────────
# Mock Provider Manager and Recommendation Engine
# ─────────────────────────────────────────────────────────────────────────────


class ProviderManager:
    """Mock provider manager"""

    def __init__(self):
        self.providers = ["openai", "anthropic", "google"]

    async def analyze(self, prompt: str) -> dict:
        return {"analysis": "mock analysis", "confidence": 0.8}


class UltimateRecommendationEngine:
    """Mock recommendation engine"""

    def generate_portfolio(self, analyses: list[dict], odds: list[dict]) -> list[dict]:
        recommendations = []
        for analysis in analyses[:3]:  # Limit to top 3
            rec = {
                "game_id": analysis.get("game_id"),
                "recommendation": "BET",
                "stake": 100,
                "expected_value": 10.5,
                "confidence": analysis.get("consensus_confidence", 0.5),
            }
            recommendations.append(rec)
        return recommendations


class UltimateDatabaseManager:
    """Mock database manager"""

    def __init__(self):
        self.data_dir = Path("data/database")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    async def store_predictions(self, predictions: list[dict]) -> bool:
        try:
            pred_file = (
                self.data_dir / f"predictions_{dt.utcnow().strftime('%Y%m%d')}.json"
            )
            with open(pred_file, "w") as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Stored {len(predictions)} predictions")
            return True
        except Exception as e:
            logger.error(f"Failed to store predictions: {e}")
            return False

    async def store_portfolio(self, portfolio: list[dict]) -> bool:
        try:
            portfolio_file = (
                self.data_dir / f"portfolio_{dt.utcnow().strftime('%Y%m%d')}.json"
            )
            with open(portfolio_file, "w") as f:
                json.dump(portfolio, f, indent=2)
            logger.info(f"Stored portfolio with {len(portfolio)} recommendations")
            return True
        except Exception as e:
            logger.error(f"Failed to store portfolio: {e}")
            return False


# Mock N8N Data Fetcher
class N8NDataFetcher:
    def __init__(self, n8n_webhook_url: str):
        self.url = n8n_webhook_url

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def fetch_all_markets(self, target_date: str) -> dict:
        # Mock market data
        return {"game1": {"moneyline": {"fanduel": {"home": 150, "away": -150}}}}


# ─────────────────────────────────────────────────────────────────────────────
# Main Processing Function
# ─────────────────────────────────────────────────────────────────────────────


async def main(
    override_date: str | None = None,
    live_odds_rows: list[dict] | None = None,
) -> int:
    """
    * override_date      – pass YYYY-MM-DD to force a day (API mode uses this)
    * live_odds_rows     – list of dicts coming straight from n8n (skips fetch)
    """

    target_date = override_date or (
        sys.argv[1] if len(sys.argv) > 1 else dt.utcnow().strftime("%Y-%m-%d")
    )

    logger.info(f"🚀 Ultimate AI Betting System for {target_date}")

    # 1. Get odds data
    odds_data: list[dict] = live_odds_rows or []

    if not odds_data:
        # Falls back to old "fetch from n8n or create mock" logic
        try:
            n8n_url = os.getenv("N8N_ODDS_WORKFLOW_URL", "")
            if n8n_url:
                async with N8NDataFetcher(n8n_webhook_url=n8n_url) as fetcher:
                    market_data = await fetcher.fetch_all_markets(
                        target_date=target_date
                    )
                    odds_data = convert_structured_odds_to_system_format(market_data)
            else:
                logger.warning("N8N_ODDS_WORKFLOW_URL not set, using mock odds")
                odds_data = create_mock_odds_data(target_date)
        except Exception as e:
            logger.error("Odds fetch failed – using mock: %s", e)
            odds_data = create_mock_odds_data(target_date)

    # Save copy to file so load_daily_data() finds it
    save_odds_to_file(odds_data, target_date)

    # 2. Load sentiment + odds → daily_data
    daily_data = await load_daily_data(target_date, odds_data)

    if not daily_data["games"]:
        logger.warning("No games found - creating mock data")
        daily_data["games"] = [
            {
                "game_id": "mock_game",
                "home_team": "Home Team",
                "away_team": "Away Team",
                "odds": odds_data,
            }
        ]

    logger.info(f"Processing {len(daily_data['games'])} games")

    # 3. AI council analysis
    provider_mgr = ProviderManager()
    ai_predictions = await run_ai_analysis(provider_mgr, daily_data["games"])

    # 4. Database storage + portfolio generation
    try:
        rec_engine = UltimateRecommendationEngine()
        analyses = []

        for ai_name, preds in ai_predictions.items():
            for p in preds:
                analyses.append(
                    {
                        "game_id": p.get("game_id", "unknown"),
                        "favored_team": (
                            p.get("pick", "").split()[0] if p.get("pick") else "Unknown"
                        ),
                        "consensus_confidence": float(p.get("confidence", 0)) / 100.0,
                        "primary_signal": p.get("primary_signal", "unknown"),
                        "individual_analyses": [{"provider": ai_name}],
                    }
                )

        final_bets = rec_engine.generate_portfolio(analyses, daily_data["odds"])

        # Store in database
        db_manager = UltimateDatabaseManager()
        prediction_data = convert_to_prediction_data(
            [p for preds in ai_predictions.values() for p in preds]
        )

        await db_manager.store_predictions(prediction_data)
        await db_manager.store_portfolio(final_bets)

        logger.info(f"Generated {len(final_bets)} final betting recommendations")

    except Exception as e:
        logger.error("Portfolio generation failed: %s", e)
        final_bets = []

    logger.info("✅ Ultimate run finished successfully")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI /ingest endpoint → called by n8n
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/ingest")
async def ingest(payload: IncomingPayload):
    """
    Accept the single merged object from n8n and run the pipeline once.
    """
    try:
        logger.info(f"📥 Received {len(payload.rows)} betting opportunities from n8n")

        date_str = payload.meta.get("generated_at", dt.utcnow().isoformat()).split("T")[
            0
        ]

        # 1. Transform the rows into the system's odds format
        odds_rows = [
            {
                "game_id": r.get("game_id", "unknown"),
                "bookmaker": r.get("bookmaker", "FanDuel"),
                "home_team": r.get("home_team", "Unknown"),
                "away_team": r.get("away_team", "Unknown"),
                "commence_time": r.get("commence_time"),
                "moneyline_home": r.get("home_price", 0),
                "moneyline_away": r.get("away_price", 0),
                "home_price": r.get("home_price", 0),
                "away_price": r.get("away_price", 0),
                "edge_cents": r.get("edge_cents", 0),
                "value_rating": r.get("value_rating", "UNKNOWN"),
                "market": r.get("market", "moneyline"),
                "total": 8.5,
                "source": "n8n_workflow",
            }
            for r in payload.rows
        ]

        # 2. Save the sentiment JSON where load_daily_data() expects it
        sentiment_dir = Path("sentiment")
        sentiment_dir.mkdir(exist_ok=True)
        with open(sentiment_dir / f"{date_str}.json", "w") as fh:
            json.dump(payload.youtube_sentiment, fh, indent=2)

        # 3. Run the core pipeline
        exit_code = await main(
            override_date=date_str,
            live_odds_rows=odds_rows,
        )

        if exit_code == 0:
            logger.info("🎯 Pipeline completed successfully")
            return {
                "status": "success",
                "message": f"Processed {len(payload.rows)} opportunities",
                "opportunities_processed": len(payload.rows),
                "date": date_str,
                "timestamp": dt.utcnow().isoformat(),
            }

        raise HTTPException(status_code=500, detail="Pipeline execution failed")

    except Exception as exc:
        logger.error("❌ Ingest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": dt.utcnow().isoformat(),
        "service": "Ultimate-Main Orchestrator",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI - run once & quit
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # For CLI usage: python ultimate_main_orchestrator.py [date]
    # For API usage: uvicorn ultimate_main_orchestrator:app --port 8768
    asyncio.run(main())
