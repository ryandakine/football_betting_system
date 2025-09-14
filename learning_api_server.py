#!/usr/bin/env python3
"""
Learning System API Server
==========================
FastAPI server that provides endpoints for n8n workflows with the self-learning betting system.
"""

import json
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from learning_integration import LearningIntegration
from providers import ask_gemini
from self_learning_system import SelfLearningSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MLB Betting Learning System API",
    description="API for integrating n8n workflows with the self-learning betting system",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize learning system
learning_system = SelfLearningSystem()
integration = LearningIntegration(learning_system)


# Pydantic models for API requests/responses
class N8NDataRequest(BaseModel):
    n8n_data: dict[str, Any]


class GameOutcomeRequest(BaseModel):
    game_id: str
    winner: str
    profit: float
    date: str


class BacktestRequest(BaseModel):
    start_date: str
    end_date: str


class PredictionRequest(BaseModel):
    game_features: dict[str, Any]


class GeminiInsightsRequest(BaseModel):
    prompt: str
    simple: bool = False


@app.on_event("startup")
async def startup_event():
    """Initialize the learning system on startup."""
    logger.info("Starting MLB Betting Learning System API")

    # Load historical data and train models
    try:
        historical_data = integration.load_historical_data_for_training()
        if not historical_data.empty:
            learning_system.train_models(historical_data)
            logger.info("Models trained successfully")
        else:
            logger.warning("No historical data available for training")
    except Exception as e:
        logger.error(f"Error during startup: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "MLB Betting Learning System API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if learning system is working
        summary = learning_system.get_learning_summary()
        return {
            "status": "healthy",
            "learning_system": "operational",
            "total_predictions": summary["overall_metrics"]["total_predictions"],
            "current_accuracy": summary["overall_metrics"]["accuracy"],
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.post("/learning/process")
async def process_n8n_data(request: N8NDataRequest):
    """Process data from n8n workflow and feed it to the learning system."""
    try:
        logger.info("Processing n8n data for learning system")

        result = await integration.process_n8n_data(request.n8n_data)

        return {
            "status": "success",
            "message": "Data processed successfully",
            "predictions_made": result.get("predictions_made", 0),
            "learning_metrics": result.get("learning_metrics", {}),
        }
    except Exception as e:
        logger.error(f"Error processing n8n data: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {e}")


@app.post("/learning/predict")
async def make_prediction(request: PredictionRequest):
    """Make a prediction using the learning system."""
    try:
        logger.info("Making prediction for game features")

        prediction = learning_system.make_prediction(request.game_features)

        return {"status": "success", "prediction": prediction}
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")


@app.post("/learning/outcome")
async def record_outcome(request: GameOutcomeRequest):
    """Record the outcome of a game for learning."""
    try:
        logger.info(f"Recording outcome for game {request.game_id}")

        learning_system.record_outcome(request.game_id, request.winner, request.profit)

        return {
            "status": "success",
            "message": f"Outcome recorded for game {request.game_id}",
        }
    except Exception as e:
        logger.error(f"Error recording outcome: {e}")
        raise HTTPException(status_code=500, detail=f"Error recording outcome: {e}")


@app.post("/learning/outcomes/batch")
async def record_outcomes_batch(outcomes: list[GameOutcomeRequest]):
    """Record multiple game outcomes at once."""
    try:
        logger.info(f"Recording {len(outcomes)} outcomes")

        for outcome in outcomes:
            learning_system.record_outcome(outcome.game_id, outcome.winner, outcome.profit)

        # Retrain models if needed
        learning_system.retrain_with_new_data()

        return {
            "status": "success",
            "message": f"Recorded {len(outcomes)} outcomes",
            "outcomes_processed": len(outcomes),
        }
    except Exception as e:
        logger.error(f"Error recording outcomes batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error recording outcomes: {e}")


@app.get("/learning/summary")
async def get_learning_summary():
    """Get a summary of the learning system's performance."""
    try:
        summary = learning_system.get_learning_summary()
        return {"status": "success", "summary": summary}
    except Exception as e:
        logger.error(f"Error getting learning summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting summary: {e}")


@app.post("/learning/backtest")
async def run_backtest(request: BacktestRequest):
    """Run a backtest on historical data."""
    try:
        logger.info(f"Running backtest from {request.start_date} to {request.end_date}")

        results = integration.run_comprehensive_backtest(request.start_date, request.end_date)

        return {"status": "success", "backtest_results": results}
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=f"Error running backtest: {e}")


@app.get("/learning/recommendations")
async def get_recommendations():
    """Get enhanced betting recommendations."""
    try:
        # For now, return a sample recommendation structure
        # In practice, this would integrate with your n8n data
        sample_data = {
            "games": [
                {
                    "game_id": "sample_1",
                    "home_team": "NYY",
                    "away_team": "BOS",
                    "home_odds": 1.85,
                    "away_odds": 2.15,
                }
            ],
            "odds": [],
            "sentiment": {},
            "ai_analysis": {},
        }

        recommendations = await integration.get_enhanced_recommendations(sample_data)

        return {"status": "success", "recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {e}")


@app.post("/learning/retrain")
async def retrain_models():
    """Manually trigger model retraining."""
    try:
        logger.info("Manually triggering model retraining")

        # Load fresh historical data
        historical_data = integration.load_historical_data_for_training()

        if not historical_data.empty:
            learning_system.train_models(historical_data, force_retrain=True)

            return {
                "status": "success",
                "message": "Models retrained successfully",
                "training_samples": len(historical_data),
            }
        else:
            return {
                "status": "warning",
                "message": "No historical data available for retraining",
            }
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        raise HTTPException(status_code=500, detail=f"Error retraining models: {e}")


@app.get("/learning/models/status")
async def get_model_status():
    """Get status of all models."""
    try:
        model_status = {}

        for model_name, model in learning_system.models.items():
            model_status[model_name] = {
                "type": type(model).__name__,
                "trained": (hasattr(model, "feature_importances_") if hasattr(model, "feature_importances_") else True),
                "n_features": (getattr(model, "n_features_in_", "unknown") if hasattr(model, "n_features_in_") else "unknown"),
            }

        return {"status": "success", "models": model_status}
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model status: {e}")


@app.get("/learning/performance")
async def get_performance_metrics():
    """Get detailed performance metrics."""
    try:
        metrics = learning_system.analyze_performance()

        return {
            "status": "success",
            "metrics": {
                "total_predictions": metrics.total_predictions,
                "correct_predictions": metrics.correct_predictions,
                "accuracy": metrics.accuracy,
                "total_profit": metrics.total_profit,
                "total_invested": metrics.total_invested,
                "roi": metrics.roi,
                "model_performance": metrics.model_performance,
                "pattern_insights": metrics.pattern_insights,
            },
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {e}")


@app.post("/insights/gemini")
async def gemini_insights(request: GeminiInsightsRequest):
    """Generate insights using Google's Gemini (smartest model)."""
    try:
        results = await ask_gemini(request.prompt)
        if request.simple:
            first = results[0] if isinstance(results, list) and results else {}
            text = first.get("rationale") or first.get("text") or json.dumps(first) if first else ""
            return {"status": "success", "insight": first, "insight_text": text}
        return {"status": "success", "insights": results}
    except Exception as e:
        logger.error(f"Gemini insights error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini insights error: {e}")


@app.get("/insights/gemini")
async def gemini_insights_get(prompt: str, simple: bool = False):
    """GET variant for simple integrations (e.g., Home Assistant)."""
    try:
        results = await ask_gemini(prompt)
        if simple:
            first = results[0] if isinstance(results, list) and results else {}
            text = first.get("rationale") or first.get("text") or json.dumps(first) if first else ""
            return {"status": "success", "insight": first, "insight_text": text}
        return {"status": "success", "insights": results}
    except Exception as e:
        logger.error(f"Gemini insights error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini insights error: {e}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "learning_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
