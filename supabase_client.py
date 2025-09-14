#!/usr/bin/env python3
"""
MLB Betting System Supabase Client
==================================
Replaces SQLite database managers with Supabase integration for cloud storage.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise ValueError(
        "SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables"
    )


class MLBSupabaseClient:
    """Supabase client for MLB betting system data storage."""

    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        logger.info("🔗 Connected to Supabase")

    # AI Predictions Methods
    async def save_ai_predictions(self, predictions: list[dict[str, Any]]) -> int:
        """Save AI predictions to Supabase."""
        if not predictions:
            return 0

        try:
            # Add created_at timestamp if not present
            for pred in predictions:
                if "created_at" not in pred:
                    pred["created_at"] = datetime.now().isoformat()

            result = self.supabase.table("ai_predictions").upsert(predictions).execute()
            logger.info(f"💾 Saved {len(predictions)} AI predictions to Supabase")
            return len(predictions)
        except Exception as e:
            logger.error(f"❌ Error saving AI predictions: {e}")
            raise

    async def get_ai_predictions(
        self, run_id: str | None = None, date: str | None = None
    ) -> list[dict[str, Any]]:
        """Get AI predictions from Supabase."""
        try:
            query = self.supabase.table("ai_predictions").select("*")

            if run_id:
                query = query.eq("run_id", run_id)
            if date:
                query = query.eq("date", date)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching AI predictions: {e}")
            return []

    # Ensemble Consensus Methods
    async def save_ensemble_consensus(self, consensus_data: dict[str, Any]) -> str:
        """Save ensemble consensus to Supabase."""
        try:
            if "created_at" not in consensus_data:
                consensus_data["created_at"] = datetime.now().isoformat()

            result = (
                self.supabase.table("ensemble_consensus")
                .upsert(consensus_data)
                .execute()
            )
            logger.info(
                f"💾 Saved ensemble consensus: {consensus_data.get('consensus_id', 'unknown')}"
            )
            return consensus_data.get("consensus_id", "")
        except Exception as e:
            logger.error(f"❌ Error saving ensemble consensus: {e}")
            raise

    async def get_ensemble_consensus(
        self, run_id: str | None = None, date: str | None = None
    ) -> list[dict[str, Any]]:
        """Get ensemble consensus from Supabase."""
        try:
            query = self.supabase.table("ensemble_consensus").select("*")

            if run_id:
                query = query.eq("run_id", run_id)
            if date:
                query = query.eq("date", date)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching ensemble consensus: {e}")
            return []

    # Daily Portfolios Methods
    async def save_daily_portfolio(self, portfolio_data: dict[str, Any]) -> str:
        """Save daily portfolio to Supabase."""
        try:
            if "created_at" not in portfolio_data:
                portfolio_data["created_at"] = datetime.now().isoformat()

            result = (
                self.supabase.table("daily_portfolios").upsert(portfolio_data).execute()
            )
            logger.info(
                f"💾 Saved daily portfolio: {portfolio_data.get('portfolio_id', 'unknown')}"
            )
            return portfolio_data.get("portfolio_id", "")
        except Exception as e:
            logger.error(f"❌ Error saving daily portfolio: {e}")
            raise

    async def get_daily_portfolios(
        self, date: str | None = None
    ) -> list[dict[str, Any]]:
        """Get daily portfolios from Supabase."""
        try:
            query = self.supabase.table("daily_portfolios").select("*")

            if date:
                query = query.eq("date", date)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching daily portfolios: {e}")
            return []

    # Recommendations Methods
    async def save_recommendations(self, recommendations: list[dict[str, Any]]) -> int:
        """Save betting recommendations to Supabase."""
        if not recommendations:
            return 0

        try:
            # Add created_at timestamp if not present
            for rec in recommendations:
                if "created_at" not in rec:
                    rec["created_at"] = datetime.now().isoformat()

            result = (
                self.supabase.table("recommendations").insert(recommendations).execute()
            )
            logger.info(f"💾 Saved {len(recommendations)} recommendations to Supabase")
            return len(recommendations)
        except Exception as e:
            logger.error(f"❌ Error saving recommendations: {e}")
            raise

    async def get_recommendations(
        self, game_id: str | None = None, date: str | None = None
    ) -> list[dict[str, Any]]:
        """Get recommendations from Supabase."""
        try:
            query = self.supabase.table("recommendations").select("*")

            if game_id:
                query = query.eq("game_id", game_id)
            if date:
                query = query.eq("date", date)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching recommendations: {e}")
            return []

    # Professional Bets Methods
    async def save_professional_bet(self, bet_data: dict[str, Any]) -> str:
        """Save professional bet to Supabase."""
        try:
            if "created_at_db" not in bet_data:
                bet_data["created_at_db"] = datetime.now().isoformat()

            result = self.supabase.table("professional_bets").upsert(bet_data).execute()
            logger.info(f"💾 Saved professional bet: {bet_data.get('id', 'unknown')}")
            return bet_data.get("id", "")
        except Exception as e:
            logger.error(f"❌ Error saving professional bet: {e}")
            raise

    async def get_professional_bets(
        self, game_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get professional bets from Supabase."""
        try:
            query = self.supabase.table("professional_bets").select("*")

            if game_id:
                query = query.eq("game_id", game_id)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching professional bets: {e}")
            return []

    # Unit Bets Methods
    async def save_unit_bet(self, bet_data: dict[str, Any]) -> str:
        """Save unit bet to Supabase."""
        try:
            if "created_at_db" not in bet_data:
                bet_data["created_at_db"] = datetime.now().isoformat()

            result = self.supabase.table("unit_bets").upsert(bet_data).execute()
            logger.info(f"💾 Saved unit bet: {bet_data.get('id', 'unknown')}")
            return bet_data.get("id", "")
        except Exception as e:
            logger.error(f"❌ Error saving unit bet: {e}")
            raise

    async def get_unit_bets(self, team: str | None = None) -> list[dict[str, Any]]:
        """Get unit bets from Supabase."""
        try:
            query = self.supabase.table("unit_bets").select("*")

            if team:
                query = query.eq("team", team)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching unit bets: {e}")
            return []

    # Odds Data Methods
    async def save_odds_data(self, odds_data: list[dict[str, Any]]) -> int:
        """Save odds data to Supabase."""
        if not odds_data:
            return 0

        try:
            # Add created_at timestamp if not present
            for odds in odds_data:
                if "created_at" not in odds:
                    odds["created_at"] = datetime.now().isoformat()

            result = self.supabase.table("odds_data").insert(odds_data).execute()
            logger.info(f"💾 Saved {len(odds_data)} odds records to Supabase")
            return len(odds_data)
        except Exception as e:
            logger.error(f"❌ Error saving odds data: {e}")
            raise

    async def get_odds_data(
        self,
        game_id: str | None = None,
        date: str | None = None,
        bookmaker: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get odds data from Supabase."""
        try:
            query = self.supabase.table("odds_data").select("*")

            if game_id:
                query = query.eq("game_id", game_id)
            if date:
                query = query.eq("date", date)
            if bookmaker:
                query = query.eq("bookmaker", bookmaker)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching odds data: {e}")
            return []

    # Sentiment Data Methods
    async def save_sentiment_data(self, sentiment_data: dict[str, Any]) -> str:
        """Save sentiment data to Supabase."""
        try:
            if "created_at" not in sentiment_data:
                sentiment_data["created_at"] = datetime.now().isoformat()

            result = (
                self.supabase.table("sentiment_data").insert(sentiment_data).execute()
            )
            logger.info(
                f"💾 Saved sentiment data for {sentiment_data.get('date', 'unknown')}"
            )
            return result.data[0]["id"] if result.data else ""
        except Exception as e:
            logger.error(f"❌ Error saving sentiment data: {e}")
            raise

    async def get_sentiment_data(
        self, date: str | None = None, source: str | None = None
    ) -> list[dict[str, Any]]:
        """Get sentiment data from Supabase."""
        try:
            query = self.supabase.table("sentiment_data").select("*")

            if date:
                query = query.eq("date", date)
            if source:
                query = query.eq("source", source)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching sentiment data: {e}")
            return []

    # Metrics Methods
    async def save_metrics(self, metrics_data: dict[str, Any]) -> str:
        """Save metrics to Supabase."""
        try:
            if "created_at" not in metrics_data:
                metrics_data["created_at"] = datetime.now().isoformat()

            result = self.supabase.table("metrics").upsert(metrics_data).execute()
            logger.info(
                f"💾 Saved metrics for {metrics_data.get('run_date', 'unknown')}"
            )
            return metrics_data.get("run_date", "")
        except Exception as e:
            logger.error(f"❌ Error saving metrics: {e}")
            raise

    async def get_metrics(self, run_date: str | None = None) -> list[dict[str, Any]]:
        """Get metrics from Supabase."""
        try:
            query = self.supabase.table("metrics").select("*")

            if run_date:
                query = query.eq("run_date", run_date)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching metrics: {e}")
            return []

    # Results Methods
    async def save_result(self, result_data: dict[str, Any]) -> str:
        """Save game result to Supabase."""
        try:
            if "created_at" not in result_data:
                result_data["created_at"] = datetime.now().isoformat()

            result = self.supabase.table("results").upsert(result_data).execute()
            logger.info(
                f"💾 Saved result for game {result_data.get('game_id', 'unknown')}"
            )
            return result_data.get("game_id", "")
        except Exception as e:
            logger.error(f"❌ Error saving result: {e}")
            raise

    async def get_results(
        self, game_id: str | None = None, game_date: str | None = None
    ) -> list[dict[str, Any]]:
        """Get results from Supabase."""
        try:
            query = self.supabase.table("results").select("*")

            if game_id:
                query = query.eq("game_id", game_id)
            if game_date:
                query = query.eq("game_date", game_date)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching results: {e}")
            return []

    # Analysis History Methods
    async def save_analysis_session(self, session_data: dict[str, Any]) -> str:
        """Save analysis session to Supabase."""
        try:
            if "created_at" not in session_data:
                session_data["created_at"] = datetime.now().isoformat()

            result = (
                self.supabase.table("analysis_history").insert(session_data).execute()
            )
            logger.info("💾 Saved analysis session to Supabase")
            return result.data[0]["id"] if result.data else ""
        except Exception as e:
            logger.error(f"❌ Error saving analysis session: {e}")
            raise

    # AI Performance Methods
    async def save_ai_performance(self, performance_data: dict[str, Any]) -> str:
        """Save AI performance data to Supabase."""
        try:
            if "created_at" not in performance_data:
                performance_data["created_at"] = datetime.now().isoformat()

            result = (
                self.supabase.table("ai_performance").upsert(performance_data).execute()
            )
            logger.info(
                f"💾 Saved AI performance: {performance_data.get('performance_id', 'unknown')}"
            )
            return performance_data.get("performance_id", "")
        except Exception as e:
            logger.error(f"❌ Error saving AI performance: {e}")
            raise

    async def get_ai_performance(
        self, ai_model: str | None = None
    ) -> list[dict[str, Any]]:
        """Get AI performance data from Supabase."""
        try:
            query = self.supabase.table("ai_performance").select("*")

            if ai_model:
                query = query.eq("ai_model", ai_model)

            result = query.execute()
            return result.data
        except Exception as e:
            logger.error(f"❌ Error fetching AI performance: {e}")
            return []

    # Learning Metrics Methods
    async def save_learning_metric(self, metric_data: dict[str, Any]) -> str:
        """Save learning metric to Supabase."""
        try:
            if "created_at" not in metric_data:
                metric_data["created_at"] = datetime.now().isoformat()

            result = (
                self.supabase.table("learning_metrics").insert(metric_data).execute()
            )
            logger.info(
                f"💾 Saved learning metric: {metric_data.get('metric_id', 'unknown')}"
            )
            return result.data[0]["id"] if result.data else ""
        except Exception as e:
            logger.error(f"❌ Error saving learning metric: {e}")
            raise

    # Utility Methods
    async def test_connection(self) -> bool:
        """Test Supabase connection."""
        try:
            result = self.supabase.table("metrics").select("count").limit(1).execute()
            logger.info("✅ Supabase connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")
            return False

    async def get_table_count(self, table_name: str) -> int:
        """Get count of records in a table."""
        try:
            result = (
                self.supabase.table(table_name).select("*", count="exact").execute()
            )
            return result.count or 0
        except Exception as e:
            logger.error(f"❌ Error getting count for {table_name}: {e}")
            return 0


# Create a global instance
supabase_client = MLBSupabaseClient()


# Convenience functions for backward compatibility
async def save_ai_predictions(predictions: list[dict[str, Any]]) -> int:
    """Save AI predictions using the global client."""
    return await supabase_client.save_ai_predictions(predictions)


async def save_recommendations(recommendations: list[dict[str, Any]]) -> int:
    """Save recommendations using the global client."""
    return await supabase_client.save_recommendations(recommendations)


async def save_odds_data(odds_data: list[dict[str, Any]]) -> int:
    """Save odds data using the global client."""
    return await supabase_client.save_odds_data(odds_data)


async def save_sentiment_data(sentiment_data: dict[str, Any]) -> str:
    """Save sentiment data using the global client."""
    return await supabase_client.save_sentiment_data(sentiment_data)
