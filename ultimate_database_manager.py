#!/usr/bin/env python3
"""
Ultimate Database Manager v4.0 - ASYNC EDITION
==============================================
Combines v3.0 personality tracking with advanced asynchronous learning database.
Stores rich AI personality data for the adaptive learning system with type safety.
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)


# --- Data Contracts for Database Operations ---
@dataclass
class AIPredictionData:
    """A structured dataclass for a single AI's prediction."""

    prediction_id: str
    run_id: str
    date: str
    game_id: str
    timestamp: str
    ai_model: str
    analyst_role: str
    specialty: str
    analyst_type: str
    pick: str
    confidence: int
    stake_units: int
    rationale: str
    primary_signal: str
    sentiment_weight: float
    odds_weight: float
    expected_value: str | None = None
    fade_strength: str | None = None
    situational_edge: str | None = None
    signal_convergence: str | None = None
    token_estimate: int | None = 0
    actual_outcome: str | None = None
    was_correct: int | None = None
    roi: float | None = None
    pnl: float | None = None
    prompt_type: str = "specialized_personality"
    learning_phase: str = "collection"


@dataclass
class EnsembleConsensusData:
    """A structured dataclass for the final ensemble consensus."""

    consensus_id: str
    run_id: str
    date: str
    game_id: str
    timestamp: str
    total_predictions: int
    consensus_pick: str
    consensus_confidence: float
    consensus_stake_units: int
    unanimous_agreement: bool
    majority_agreement: bool
    signal_agreement_score: float
    supporting_analysts: list[str]
    conflicting_analysts: list[str]
    recommended_bet: str
    recommended_stake: float
    edge_estimate: float
    final_outcome: str | None = None
    consensus_roi: float | None = None
    consensus_accuracy: int | None = None


@dataclass
class DailyPortfolioData:
    """A structured dataclass for daily portfolio tracking."""

    portfolio_id: str
    run_id: str
    date: str
    timestamp: str
    total_bets: int
    total_stake: float
    total_units: int
    estimated_edge: float
    quant_bets: int = 0
    sentiment_bets: int = 0
    news_bets: int = 0
    synthesizer_bets: int = 0
    actual_roi: float | None = None
    actual_pnl: float | None = None
    winning_bets: int | None = None
    portfolio_accuracy: float | None = None
    bets_json: str = "{}"


class UltimateDatabaseManager:
    """
    Advanced async database manager for the Ultimate AI Betting System.
    Tracks AI personalities, signal weights, predictions, and learning metrics with type safety.
    """

    def __init__(self, db_path: str = "data/ultimate_betting_system.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = None
        self.cursor = None

    async def __aenter__(self):
        self.conn = await aiosqlite.connect(self.db_path, timeout=30.0)
        self.cursor = await self.conn.cursor()
        await self._create_all_tables()
        logger.info(f"🗄️ Ultimate Database initialized at {self.db_path}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            await self.conn.close()
        logger.info("🗄️ Database connection closed.")

    async def _create_all_tables(self):
        """Creates comprehensive schema for the ultimate learning system with indexes."""

        # Main AI predictions table with full schema
        await self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_predictions (
                prediction_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                date TEXT NOT NULL,
                game_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- AI Personality Info
                ai_model TEXT NOT NULL,
                analyst_role TEXT NOT NULL,
                specialty TEXT NOT NULL,
                analyst_type TEXT NOT NULL,

                -- Prediction Details
                pick TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                stake_units INTEGER NOT NULL,
                rationale TEXT NOT NULL,
                primary_signal TEXT NOT NULL,

                -- Signal Weights Used
                sentiment_weight REAL NOT NULL,
                odds_weight REAL NOT NULL,

                -- Advanced Metrics
                expected_value TEXT,
                fade_strength TEXT,
                situational_edge TEXT,
                signal_convergence TEXT,
                token_estimate INTEGER,

                -- Results (filled later by results collector)
                actual_outcome TEXT,
                was_correct INTEGER,
                roi REAL,
                pnl REAL,

                -- Learning Metadata
                prompt_type TEXT DEFAULT 'specialized_personality',
                learning_phase TEXT DEFAULT 'collection'
            )
        """
        )

        # Ensemble consensus table
        await self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ensemble_consensus (
                consensus_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                date TEXT NOT NULL,
                game_id TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,

                -- Consensus Results
                total_predictions INTEGER NOT NULL,
                consensus_pick TEXT,
                consensus_confidence REAL,
                consensus_stake_units INTEGER,

                -- AI Agreement Metrics
                unanimous_agreement INTEGER DEFAULT 0,
                majority_agreement INTEGER DEFAULT 0,
                signal_agreement_score REAL,

                -- Supporting Evidence
                supporting_analysts TEXT, -- JSON array of supporting AI names
                conflicting_analysts TEXT, -- JSON array of dissenting AI names

                -- Final Recommendation
                recommended_bet TEXT,
                recommended_stake REAL,
                edge_estimate REAL,

                -- Results
                final_outcome TEXT,
                consensus_roi REAL,
                consensus_accuracy INTEGER
            )
        """
        )

        # Daily portfolios table
        await self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_portfolios (
                portfolio_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                date TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- Portfolio Metrics
                total_bets INTEGER NOT NULL,
                total_stake REAL NOT NULL,
                total_units INTEGER NOT NULL,
                estimated_edge REAL,

                -- AI Contribution
                quant_bets INTEGER DEFAULT 0,
                sentiment_bets INTEGER DEFAULT 0,
                news_bets INTEGER DEFAULT 0,
                synthesizer_bets INTEGER DEFAULT 0,

                -- Results
                actual_roi REAL,
                actual_pnl REAL,
                winning_bets INTEGER,
                portfolio_accuracy REAL,

                -- Portfolio JSON
                bets_json TEXT -- Full portfolio details as JSON
            )
        """
        )

        # AI Performance tracking table
        await self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_performance (
                performance_id TEXT PRIMARY KEY,
                ai_model TEXT NOT NULL,
                analyst_role TEXT NOT NULL,
                date_range_start TEXT NOT NULL,
                date_range_end TEXT NOT NULL,

                -- Performance Metrics
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy_rate REAL DEFAULT 0,
                total_roi REAL DEFAULT 0,
                average_confidence REAL DEFAULT 0,

                -- Signal Effectiveness
                sentiment_weight_avg REAL DEFAULT 0,
                odds_weight_avg REAL DEFAULT 0,
                best_signal_type TEXT,
                worst_signal_type TEXT,

                -- Specialty Performance
                specialty_success_rate REAL DEFAULT 0,
                primary_signal_success_rate REAL DEFAULT 0,

                -- Learning Adjustments
                recommended_sentiment_weight REAL,
                recommended_odds_weight REAL,
                confidence_reliability REAL,

                -- Metadata
                last_updated TEXT NOT NULL,
                sample_size_adequate INTEGER DEFAULT 0
            )
        """
        )

        # Learning metrics table
        await self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_metrics (
                metric_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata TEXT, -- JSON for additional context
                timestamp TEXT NOT NULL
            )
        """
        )

        # Create performance indexes
        await self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_game_id ON ai_predictions(game_id)"
        )
        await self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_date ON ai_predictions(date)"
        )
        await self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_model ON ai_predictions(ai_model)"
        )
        await self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_run_id ON ai_predictions(run_id)"
        )
        await self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_consensus_date ON ensemble_consensus(date)"
        )
        await self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_date ON daily_portfolios(date)"
        )

        await self.conn.commit()
        logger.info("✅ All database tables and indexes created successfully")

    async def save_ai_predictions_from_personalities_data(
        self, run_id: str, personalities_data: dict[str, Any]
    ) -> int:
        """Convert personalities data format to structured predictions and save."""
        predictions = []

        for personality_name, personality_info in personalities_data.get(
            "personalities", {}
        ).items():
            role = personality_info.get("role", "Unknown")
            specialty = personality_info.get("specialty", "General")

            for prediction in personality_info.get("predictions", []):
                prediction_id = f"{run_id}_{prediction.get('game_id', 'unknown')}_{personality_name}"

                pred_data = AIPredictionData(
                    prediction_id=prediction_id,
                    run_id=run_id,
                    date=personalities_data["meta"]["date"],
                    game_id=prediction.get("game_id", ""),
                    timestamp=prediction.get("timestamp", datetime.now().isoformat()),
                    # AI Personality
                    ai_model=personality_name,
                    analyst_role=role,
                    specialty=specialty,
                    analyst_type=prediction.get("analyst_type", "unknown"),
                    # Prediction
                    pick=prediction.get("pick", ""),
                    confidence=prediction.get("confidence", 0),
                    stake_units=prediction.get("stake_units", 1),
                    rationale=prediction.get("rationale", ""),
                    primary_signal=prediction.get("primary_signal", ""),
                    # Weights
                    sentiment_weight=prediction.get("signal_weights", {}).get(
                        "sentiment_weight", 0.5
                    ),
                    odds_weight=prediction.get("signal_weights", {}).get(
                        "odds_weight", 0.5
                    ),
                    # Advanced metrics
                    expected_value=prediction.get("expected_value"),
                    fade_strength=prediction.get("fade_strength"),
                    situational_edge=prediction.get("situational_edge"),
                    signal_convergence=prediction.get("signal_convergence"),
                    token_estimate=prediction.get("token_estimate", 0),
                )

                predictions.append(pred_data)

        await self.save_ai_predictions(predictions)
        return len(predictions)

    async def save_ai_predictions(self, predictions: list[AIPredictionData]):
        """Saves a batch of AI personality predictions with rich metadata."""
        if not predictions:
            return

        # Convert dataclasses to tuples for database insertion
        records = []
        for pred in predictions:
            record = (
                pred.prediction_id,
                pred.run_id,
                pred.date,
                pred.game_id,
                pred.timestamp,
                pred.ai_model,
                pred.analyst_role,
                pred.specialty,
                pred.analyst_type,
                pred.pick,
                pred.confidence,
                pred.stake_units,
                pred.rationale,
                pred.primary_signal,
                pred.sentiment_weight,
                pred.odds_weight,
                pred.expected_value,
                pred.fade_strength,
                pred.situational_edge,
                pred.signal_convergence,
                pred.token_estimate,
                pred.actual_outcome,
                pred.was_correct,
                pred.roi,
                pred.pnl,
                pred.prompt_type,
                pred.learning_phase,
            )
            records.append(record)

        await self.cursor.executemany(
            """
            INSERT OR REPLACE INTO ai_predictions (
                prediction_id, run_id, date, game_id, timestamp,
                ai_model, analyst_role, specialty, analyst_type,
                pick, confidence, stake_units, rationale, primary_signal,
                sentiment_weight, odds_weight,
                expected_value, fade_strength, situational_edge,
                signal_convergence, token_estimate,
                actual_outcome, was_correct, roi, pnl,
                prompt_type, learning_phase
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            records,
        )

        await self.conn.commit()
        logger.info(f"💾 Saved {len(records)} AI predictions to database")

    async def save_ensemble_consensus(self, consensus_data: EnsembleConsensusData):
        """Saves a single ensemble consensus result."""
        # Convert lists to JSON strings for storage
        data_dict = asdict(consensus_data)
        data_dict["supporting_analysts"] = json.dumps(data_dict["supporting_analysts"])
        data_dict["conflicting_analysts"] = json.dumps(
            data_dict["conflicting_analysts"]
        )
        data_dict["unanimous_agreement"] = int(data_dict["unanimous_agreement"])
        data_dict["majority_agreement"] = int(data_dict["majority_agreement"])

        await self.cursor.execute(
            """
            INSERT OR REPLACE INTO ensemble_consensus (
                consensus_id, run_id, date, game_id, timestamp,
                total_predictions, consensus_pick, consensus_confidence, consensus_stake_units,
                unanimous_agreement, majority_agreement, signal_agreement_score,
                supporting_analysts, conflicting_analysts,
                recommended_bet, recommended_stake, edge_estimate,
                final_outcome, consensus_roi, consensus_accuracy
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            tuple(data_dict.values()),
        )

        await self.conn.commit()
        return consensus_data.consensus_id

    async def save_daily_portfolio(self, portfolio_data: DailyPortfolioData):
        """Save daily portfolio data."""
        record = (
            portfolio_data.portfolio_id,
            portfolio_data.run_id,
            portfolio_data.date,
            portfolio_data.timestamp,
            portfolio_data.total_bets,
            portfolio_data.total_stake,
            portfolio_data.total_units,
            portfolio_data.estimated_edge,
            portfolio_data.quant_bets,
            portfolio_data.sentiment_bets,
            portfolio_data.news_bets,
            portfolio_data.synthesizer_bets,
            portfolio_data.actual_roi,
            portfolio_data.actual_pnl,
            portfolio_data.winning_bets,
            portfolio_data.portfolio_accuracy,
            portfolio_data.bets_json,
        )

        await self.cursor.execute(
            """
            INSERT OR REPLACE INTO daily_portfolios VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            record,
        )

        await self.conn.commit()
        return portfolio_data.portfolio_id

    async def get_ai_performance_summary(self, days: int = 30) -> dict[str, Any]:
        """Get performance summary for all AI personalities."""
        await self.cursor.execute(
            """
            SELECT ai_model, analyst_role,
                   COUNT(*) as total_predictions,
                   AVG(confidence) as avg_confidence,
                   AVG(sentiment_weight) as avg_sentiment_weight,
                   AVG(odds_weight) as avg_odds_weight,
                   COUNT(CASE WHEN was_correct = 1 THEN 1 END) as correct_predictions,
                   AVG(roi) as avg_roi
            FROM ai_predictions
            WHERE date >= date('now', '-{} days')
            GROUP BY ai_model, analyst_role
        """.format(
                days
            )
        )

        results = {}
        async for row in self.cursor:
            model = row[0]
            results[model] = {
                "role": row[1],
                "total_predictions": row[2],
                "avg_confidence": row[3] or 0,
                "avg_sentiment_weight": row[4] or 0,
                "avg_odds_weight": row[5] or 0,
                "correct_predictions": row[6] or 0,
                "accuracy": (row[6] or 0) / max(row[2], 1),
                "avg_roi": row[7] or 0,
            }

        return results

    async def update_prediction_results(
        self, prediction_id: str, was_correct: bool, roi: float, pnl: float
    ):
        """Update prediction with actual results."""
        await self.cursor.execute(
            """
            UPDATE ai_predictions
            SET was_correct = ?, roi = ?, pnl = ?, actual_outcome = ?
            WHERE prediction_id = ?
        """,
            (
                int(was_correct),
                roi,
                pnl,
                "win" if was_correct else "loss",
                prediction_id,
            ),
        )

        await self.conn.commit()

    async def get_learning_insights(self) -> dict[str, Any]:
        """Generate insights for the learning system."""
        insights = {}

        # Best performing AI by accuracy
        await self.cursor.execute(
            """
            SELECT ai_model, analyst_role,
                   AVG(CASE WHEN was_correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
                   COUNT(*) as sample_size
            FROM ai_predictions
            WHERE was_correct IS NOT NULL
            GROUP BY ai_model
            HAVING sample_size >= 10
            ORDER BY accuracy DESC
        """
        )

        best_performers = []
        async for row in self.cursor:
            best_performers.append(
                {
                    "model": row[0],
                    "role": row[1],
                    "accuracy": row[2],
                    "sample_size": row[3],
                }
            )
        insights["best_performers"] = best_performers

        # Signal effectiveness
        await self.cursor.execute(
            """
            SELECT primary_signal,
                   AVG(CASE WHEN was_correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
                   COUNT(*) as frequency
            FROM ai_predictions
            WHERE was_correct IS NOT NULL
            GROUP BY primary_signal
            ORDER BY accuracy DESC
        """
        )

        signal_effectiveness = []
        async for row in self.cursor:
            signal_effectiveness.append(
                {"signal": row[0], "accuracy": row[1], "frequency": row[2]}
            )
        insights["signal_effectiveness"] = signal_effectiveness

        return insights


# --- Example Usage Demonstration ---
async def main():
    """Demonstrates how to use the UltimateDatabaseManager."""
    logging.basicConfig(level=logging.INFO)

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Create some sample prediction data
    sample_preds = [
        AIPredictionData(
            prediction_id=f"{run_id}_game123_claude",
            run_id=run_id,
            date=today_str,
            game_id="game123",
            timestamp=datetime.now().isoformat(),
            ai_model="claude4",
            analyst_role="The Quantitative Analyst",
            specialty="Market inefficiencies and line value",
            analyst_type="quantitative",
            pick="Yankees ML (-145)",
            confidence=72,
            stake_units=3,
            rationale="Line movement from -150 to -145 indicates sharp money on Yankees despite 67% public on Red Sox.",
            primary_signal="line_value",
            sentiment_weight=0.4,
            odds_weight=0.6,
        ),
        AIPredictionData(
            prediction_id=f"{run_id}_game123_gpt4",
            run_id=run_id,
            date=today_str,
            game_id="game123",
            timestamp=datetime.now().isoformat(),
            ai_model="gpt4",
            analyst_role="The Sentiment Trader",
            specialty="Public betting patterns and narrative analysis",
            analyst_type="sentiment",
            pick="Red Sox +150",
            confidence=68,
            stake_units=2,
            rationale="Classic fade spot: 67% public money on Red Sox with heavy social media hype.",
            primary_signal="contrarian",
            sentiment_weight=0.7,
            odds_weight=0.3,
        ),
    ]

    # 2. Use the Database Manager to save the data
    async with UltimateDatabaseManager() as db:
        await db.save_ai_predictions(sample_preds)

        # Example consensus data
        consensus = EnsembleConsensusData(
            consensus_id=f"{run_id}_consensus",
            run_id=run_id,
            date=today_str,
            game_id="game123",
            timestamp=datetime.now().isoformat(),
            total_predictions=2,
            consensus_pick="Yankees ML",
            consensus_confidence=0.70,
            consensus_stake_units=3,
            unanimous_agreement=False,
            majority_agreement=True,
            signal_agreement_score=0.8,
            supporting_analysts=["claude4"],
            conflicting_analysts=["gpt4"],
            recommended_bet="Yankees ML (-145)",
            recommended_stake=30.0,
            edge_estimate=0.08,
        )
        await db.save_ensemble_consensus(consensus)

    print(
        "\n✅ Demonstration complete. Check the 'data/ultimate_betting_system.db' file."
    )


if __name__ == "__main__":
    asyncio.run(main())
