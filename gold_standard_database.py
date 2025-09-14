import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional  # Added Dict import

logger = logging.getLogger(__name__)


class GoldStandardDatabase:
    """Database handler for Gold Standard MLB betting system."""

    def __init__(self, db_path: str = "gold_standard_betting.db"):
        self.db_path = db_path
        self.ensure_database_exists()

    def ensure_database_exists(self):
        """Ensure database and tables exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create recommendations table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        game_id TEXT,
                        bet_type TEXT,
                        selection TEXT,
                        odds REAL,
                        bookmaker TEXT,
                        win_probability REAL,
                        confidence REAL,
                        kelly_fraction REAL,
                        bet_size REAL,
                        units REAL,
                        expected_value REAL,
                        risk_level TEXT,
                        recommendation TEXT,
                        raw_data TEXT
                    )
                """
                )

                # Create analysis_history table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_opportunities INTEGER,
                        recommendations_count INTEGER,
                        avg_confidence REAL,
                        total_expected_value REAL,
                        session_data TEXT
                    )
                """
                )

                conn.commit()
                logger.info(f"✅ Database initialized: {self.db_path}")

        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
            raise

    def save_recommendations(
        self, recommendations: list[dict]
    ):  # Fixed: Dict is now imported
        """Save betting recommendations to database."""
        if not recommendations:
            logger.warning("No recommendations to save")
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for rec in recommendations:
                    opportunity = rec.get("opportunity", {})
                    cursor.execute(
                        """
                        INSERT INTO recommendations (
                            timestamp, game_id, bet_type, selection, odds, bookmaker,
                            win_probability, confidence, kelly_fraction, bet_size, units,
                            expected_value, risk_level, recommendation, raw_data
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            datetime.now().isoformat(),
                            opportunity.get("game_id"),
                            opportunity.get("bet_type"),
                            opportunity.get("selection"),
                            opportunity.get("odds"),
                            opportunity.get("bookmaker"),
                            rec.get("win_probability"),
                            rec.get("confidence"),
                            rec.get("kelly_fraction"),
                            rec.get("bet_size"),
                            rec.get("units"),
                            rec.get("expected_value"),
                            rec.get("risk_level"),
                            rec.get("recommendation"),
                            json.dumps(rec),
                        ),
                    )

                conn.commit()
                logger.info(
                    f"💾 Saved {len(recommendations)} recommendations to database"
                )

        except Exception as e:
            logger.error(f"❌ Error saving recommendations: {e}")
            raise

    def save_analysis_session(self, session_data: dict):
        """Save analysis session summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO analysis_history (
                        timestamp, total_opportunities, recommendations_count,
                        avg_confidence, total_expected_value, session_data
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        datetime.now().isoformat(),
                        session_data.get("total_opportunities", 0),
                        session_data.get("recommendations_count", 0),
                        session_data.get("avg_confidence", 0.0),
                        session_data.get("total_expected_value", 0.0),
                        json.dumps(session_data),
                    ),
                )

                conn.commit()
                logger.info("📊 Analysis session saved to database")

        except Exception as e:
            logger.error(f"❌ Error saving session: {e}")

    def get_recent_recommendations(self, limit: int = 50) -> list[dict]:
        """Get recent recommendations from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM recommendations
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (limit,),
                )

                columns = [desc[0] for desc in cursor.description]
                results = []

                for row in cursor.fetchall():
                    rec = dict(zip(columns, row))
                    if rec.get("raw_data"):
                        try:
                            rec["parsed_data"] = json.loads(rec["raw_data"])
                        except:
                            pass
                    results.append(rec)

                return results

        except Exception as e:
            logger.error(f"❌ Error getting recommendations: {e}")
            return []

    def get_database_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Count recommendations
                cursor.execute("SELECT COUNT(*) FROM recommendations")
                total_recommendations = cursor.fetchone()[0]

                # Count sessions
                cursor.execute("SELECT COUNT(*) FROM analysis_history")
                total_sessions = cursor.fetchone()[0]

                # Recent session stats
                cursor.execute(
                    """
                    SELECT AVG(avg_confidence), SUM(total_expected_value)
                    FROM analysis_history
                    WHERE timestamp > datetime('now', '-7 days')
                """
                )
                recent_stats = cursor.fetchone()

                return {
                    "total_recommendations": total_recommendations,
                    "total_sessions": total_sessions,
                    "recent_avg_confidence": recent_stats[0] or 0.0,
                    "recent_total_ev": recent_stats[1] or 0.0,
                    "database_path": self.db_path,
                    "database_size_mb": round(
                        os.path.getsize(self.db_path) / 1024 / 1024, 2
                    ),
                }

        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}


if __name__ == "__main__":
    # Test database functionality
    db = GoldStandardDatabase()
    stats = db.get_database_stats()
    print("📊 Database Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
