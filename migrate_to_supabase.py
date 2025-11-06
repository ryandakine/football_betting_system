#!/usr/bin/env python3
"""
MLB Betting System Migration to Supabase
========================================
Migrates existing SQLite data to Supabase cloud storage.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from supabase_client import MLBSupabaseClient

logger = logging.getLogger(__name__)


class MLBMigrationToSupabase:
    """Migrates MLB betting system data from SQLite to Supabase."""

    def __init__(self):
        self.supabase_client = MLBSupabaseClient()
        self.migration_stats = {
            "ai_predictions": 0,
            "recommendations": 0,
            "professional_bets": 0,
            "unit_bets": 0,
            "odds_data": 0,
            "sentiment_data": 0,
            "metrics": 0,
            "results": 0,
            "analysis_history": 0,
        }

    def get_sqlite_connection(self, db_path: str) -> sqlite3.Connection:
        """Get SQLite connection."""
        if not Path(db_path).exists():
            logger.warning(f"Database file not found: {db_path}")
            return None
        return sqlite3.connect(db_path)

    async def migrate_ai_predictions(
        self, db_path: str = "data/ultimate_betting_system.db"
    ):
        """Migrate AI predictions from SQLite to Supabase."""
        conn = self.get_sqlite_connection(db_path)
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ai_predictions")
            rows = cursor.fetchall()

            if not rows:
                logger.info("No AI predictions found to migrate")
                return

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Convert to list of dictionaries
            predictions = []
            for row in rows:
                prediction = dict(zip(columns, row))
                # Convert any JSON strings to proper format
                if "rationale" in prediction and prediction["rationale"]:
                    try:
                        prediction["rationale"] = json.loads(prediction["rationale"])
                    except:
                        pass  # Keep as string if not valid JSON
                predictions.append(prediction)

            # Save to Supabase
            count = await self.supabase_client.save_ai_predictions(predictions)
            self.migration_stats["ai_predictions"] = count
            logger.info(f"✅ Migrated {count} AI predictions")

        except Exception as e:
            logger.error(f"❌ Error migrating AI predictions: {e}")
        finally:
            conn.close()

    async def migrate_recommendations(self, db_path: str = "gold_standard_betting.db"):
        """Migrate recommendations from SQLite to Supabase."""
        conn = self.get_sqlite_connection(db_path)
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM recommendations")
            rows = cursor.fetchall()

            if not rows:
                logger.info("No recommendations found to migrate")
                return

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Convert to list of dictionaries
            recommendations = []
            for row in rows:
                rec = dict(zip(columns, row))
                # Convert raw_data JSON string
                if "raw_data" in rec and rec["raw_data"]:
                    try:
                        rec["raw_data"] = json.loads(rec["raw_data"])
                    except:
                        pass  # Keep as string if not valid JSON
                recommendations.append(rec)

            # Save to Supabase
            count = await self.supabase_client.save_recommendations(recommendations)
            self.migration_stats["recommendations"] = count
            logger.info(f"✅ Migrated {count} recommendations")

        except Exception as e:
            logger.error(f"❌ Error migrating recommendations: {e}")
        finally:
            conn.close()

    async def migrate_professional_bets(
        self, db_path: str = "professional_betting_history.db"
    ):
        """Migrate professional bets from SQLite to Supabase."""
        conn = self.get_sqlite_connection(db_path)
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM professional_bets")
            rows = cursor.fetchall()

            if not rows:
                logger.info("No professional bets found to migrate")
                return

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Convert to list of dictionaries
            bets = []
            for row in rows:
                bet = dict(zip(columns, row))
                bets.append(bet)

            # Save to Supabase
            count = 0
            for bet in bets:
                await self.supabase_client.save_professional_bet(bet)
                count += 1

            self.migration_stats["professional_bets"] = count
            logger.info(f"✅ Migrated {count} professional bets")

        except Exception as e:
            logger.error(f"❌ Error migrating professional bets: {e}")
        finally:
            conn.close()

    async def migrate_unit_bets(self, db_path: str = "unit_betting_history.db"):
        """Migrate unit bets from SQLite to Supabase."""
        conn = self.get_sqlite_connection(db_path)
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM unit_bets")
            rows = cursor.fetchall()

            if not rows:
                logger.info("No unit bets found to migrate")
                return

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Convert to list of dictionaries
            bets = []
            for row in rows:
                bet = dict(zip(columns, row))
                bets.append(bet)

            # Save to Supabase
            count = 0
            for bet in bets:
                await self.supabase_client.save_unit_bet(bet)
                count += 1

            self.migration_stats["unit_bets"] = count
            logger.info(f"✅ Migrated {count} unit bets")

        except Exception as e:
            logger.error(f"❌ Error migrating unit bets: {e}")
        finally:
            conn.close()

    async def migrate_metrics(self, db_path: str = "db/bets.sqlite"):
        """Migrate metrics from SQLite to Supabase."""
        conn = self.get_sqlite_connection(db_path)
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metrics")
            rows = cursor.fetchall()

            if not rows:
                logger.info("No metrics found to migrate")
                return

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Convert to list of dictionaries
            metrics = []
            for row in rows:
                metric = dict(zip(columns, row))
                # Convert notes_json
                if "notes_json" in metric and metric["notes_json"]:
                    try:
                        metric["notes_json"] = json.loads(metric["notes_json"])
                    except:
                        metric["notes_json"] = {}
                metrics.append(metric)

            # Save to Supabase
            count = 0
            for metric in metrics:
                await self.supabase_client.save_metrics(metric)
                count += 1

            self.migration_stats["metrics"] = count
            logger.info(f"✅ Migrated {count} metrics")

        except Exception as e:
            logger.error(f"❌ Error migrating metrics: {e}")
        finally:
            conn.close()

    async def migrate_results(self, db_path: str = "db/bets.sqlite"):
        """Migrate results from SQLite to Supabase."""
        conn = self.get_sqlite_connection(db_path)
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM results")
            rows = cursor.fetchall()

            if not rows:
                logger.info("No results found to migrate")
                return

            # Get column names
            columns = [description[0] for description in cursor.description]

            # Convert to list of dictionaries
            results = []
            for row in rows:
                result = dict(zip(columns, row))
                results.append(result)

            # Save to Supabase
            count = 0
            for result in results:
                await self.supabase_client.save_result(result)
                count += 1

            self.migration_stats["results"] = count
            logger.info(f"✅ Migrated {count} results")

        except Exception as e:
            logger.error(f"❌ Error migrating results: {e}")
        finally:
            conn.close()

    async def migrate_json_files(self):
        """Migrate JSON files to Supabase."""
        # Migrate odds data from JSON files
        odds_dir = Path("data/odds")
        if odds_dir.exists():
            count = 0
            for odds_file in odds_dir.glob("*.json"):
                try:
                    with open(odds_file) as f:
                        odds_data = json.load(f)

                    # Add date from filename
                    date_str = odds_file.stem
                    for odds in odds_data:
                        odds["date"] = date_str
                        odds["source"] = "migrated_json"

                    await self.supabase_client.save_odds_data(odds_data)
                    count += len(odds_data)

                except Exception as e:
                    logger.error(f"❌ Error migrating odds file {odds_file}: {e}")

            self.migration_stats["odds_data"] = count
            logger.info(f"✅ Migrated {count} odds records from JSON files")

        # Migrate sentiment data from JSON files
        sentiment_dir = Path("sentiment")
        if sentiment_dir.exists():
            count = 0
            for sentiment_file in sentiment_dir.glob("*.json"):
                try:
                    with open(sentiment_file) as f:
                        sentiment_data = json.load(f)

                    # Add metadata
                    sentiment_record = {
                        "date": sentiment_file.stem,
                        "source": "youtube_sentiment",
                        "data": sentiment_data,
                        "generated_at": datetime.now().isoformat(),
                    }

                    await self.supabase_client.save_sentiment_data(sentiment_record)
                    count += 1

                except Exception as e:
                    logger.error(
                        f"❌ Error migrating sentiment file {sentiment_file}: {e}"
                    )

            self.migration_stats["sentiment_data"] = count
            logger.info(f"✅ Migrated {count} sentiment records from JSON files")

    async def run_full_migration(self):
        """Run complete migration from SQLite to Supabase."""
        logger.info("🚀 Starting full migration to Supabase...")

        # Test connection first
        if not await self.supabase_client.test_connection():
            logger.error("❌ Cannot connect to Supabase. Check your credentials.")
            return

        # Run all migrations
        await self.migrate_ai_predictions()
        await self.migrate_recommendations()
        await self.migrate_professional_bets()
        await self.migrate_unit_bets()
        await self.migrate_metrics()
        await self.migrate_results()
        await self.migrate_json_files()

        # Print migration summary
        logger.info("📊 Migration Summary:")
        total_migrated = sum(self.migration_stats.values())
        for table, count in self.migration_stats.items():
            logger.info(f"  {table}: {count} records")
        logger.info(f"  Total: {total_migrated} records migrated")
        logger.info("✅ Migration completed!")


async def main():
    """Main migration function."""
    migration = MLBMigrationToSupabase()
    await migration.run_full_migration()


if __name__ == "__main__":
    asyncio.run(main())
