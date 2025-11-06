import logging
import os
from datetime import datetime

import polars as pl

logger = logging.getLogger(__name__)


def _load_data_safe(path: str, schema: dict = None) -> pl.DataFrame:
    """Safely loads a Polars DataFrame, returns empty DataFrame if not found/error."""
    if os.path.exists(path):
        try:
            df = pl.read_parquet(path)
            logger.debug(f"Successfully loaded data from {path} (rows: {df.height})")
            return df
        except Exception as e:
            logger.error(
                f"Error loading data from {path}: {e}. Returning empty DataFrame."
            )
            return pl.DataFrame(schema=schema if schema else {})
    else:
        logger.warning(f"Data file not found: {path}. Returning empty DataFrame.")
        return pl.DataFrame(schema=schema if schema else {})


def run_backtest(
    recommendations_path: str,
    actual_results_path: str,
    evaluation_date: datetime,
    output_dir: str,
) -> None:
    """
    Performs backtesting of betting recommendations against actual game results.
    Calculates profit/loss and updates betting history.

    Args:
        recommendations_path (str): Path to parquet file with betting recommendations.
        actual_results_path (str): Path to parquet file with actual game results.
        evaluation_date (datetime): Date for which results are evaluated.
        output_dir (str): Directory to save bet history and summary results.
    """
    logger.info(
        f"Starting backtesting for recommendations on {evaluation_date.strftime('%Y-%m-%d')}..."
    )

    recommendations_df = _load_data_safe(recommendations_path)
    actuals_df = _load_data_safe(actual_results_path)

    if recommendations_df.is_empty():
        logger.warning("No recommendations found to backtest.")
        return
    if actuals_df.is_empty():
        logger.warning("No actual results found for backtesting. Skipping evaluation.")
        return

    actuals_for_join = actuals_df.select(
        [
            pl.col("game_id").cast(pl.Utf8),
            pl.col("Date")
            .str.strptime(pl.Date, format="%Y-%m-%d")
            .alias("actual_date"),
            pl.col("Home Score").cast(pl.Int32).alias("actual_home_score"),
            pl.col("Away Score").cast(pl.Int32).alias("actual_away_score"),
        ]
    ).unique(subset=["game_id"])

    player_actuals_for_join = actuals_df.select(
        [
            pl.col("game_id").cast(pl.Utf8),
            pl.col("player_id").cast(pl.Utf8),
            pl.col("actual_home_runs").cast(pl.Int32, strict=False).fill_null(0),
            pl.col("actual_strikeouts").cast(pl.Int32, strict=False).fill_null(0),
        ]
    ).unique(subset=["game_id", "player_id"])

    merged_df = recommendations_df.join(
        actuals_for_join, on="game_id", how="left"
    ).join(
        player_actuals_for_join,
        on=["game_id", "player_id"],
        how="left",
        suffix="_player_actual",
    )

    def calculate_payout(row):
        bet_type = row["bet_type"]
        wager = row["wager_amount"]
        odds = row["odds"]

        if row.get("actual_date") is None:
            return None

        if "moneyline" in bet_type:
            actual_home_wins = row["actual_home_team_wins"]
            recommended_team = row.get("recommended_team")
            if recommended_team:
                if recommended_team == row.get("home_team"):
                    return wager * odds if actual_home_wins == 1 else -wager
                elif recommended_team == row.get("away_team"):
                    return wager * odds if actual_home_wins == 0 else -wager
            return None

        elif bet_type == "run_line":
            actual_home_score = row.get("actual_home_score")
            actual_away_score = row.get("actual_away_score")
            recommended_team = row.get("recommended_team")
            if actual_home_score is None or actual_away_score is None:
                return None
            if recommended_team == row.get("home_team"):
                return (
                    wager * odds
                    if (actual_home_score - actual_away_score) > 1.5
                    else -wager
                )
            else:
                return (
                    wager * odds
                    if (actual_home_score - actual_away_score) < 1.5
                    else -wager
                )

        elif "total_" in bet_type:
            actual_total_runs = row.get("actual_home_score", 0) + row.get(
                "actual_away_score", 0
            )
            total_line = row.get("total_line")
            if total_line is None:
                return None
            if "total_over" in bet_type:
                return wager * odds if actual_total_runs > total_line else -wager
            elif "total_under" in bet_type:
                return wager * odds if actual_total_runs < total_line else -wager
            return None

        elif "player_home_runs_over" == bet_type:
            actual_hrs = row.get("actual_home_runs")
            prop_line = row.get("prop_line")
            if actual_hrs is None or prop_line is None:
                return None
            return wager * odds if actual_hrs > prop_line else -wager

        elif "player_strikeouts_over" == bet_type:
            actual_ks = row.get("actual_strikeouts")
            prop_line = row.get("prop_line")
            if actual_ks is None or prop_line is None:
                return None
            return wager * odds if actual_ks > prop_line else -wager

        return None

    if not merged_df.is_empty():
        merged_df = merged_df.with_columns(
            pl.struct(pl.all()).apply(calculate_payout).alias("payout")
        ).with_columns(
            pl.when(pl.col("payout") > 0)
            .then(pl.lit("win"))
            .when(pl.col("payout") < 0)
            .then(pl.lit("loss"))
            .otherwise(pl.lit("unsettled"))
            .alias("bet_outcome")
        )
    else:
        logger.info(
            "No recommendations found with sufficient data to calculate payouts."
        )
        merged_df = recommendations_df.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("payout"),
                pl.lit("unsettled").alias("bet_outcome"),
            ]
        )

    all_evaluated_bets_df = (
        merged_df.with_columns(
            [
                pl.col("date").cast(pl.Utf8),
                pl.lit(evaluation_date.strftime("%Y-%m-%d")).alias("evaluation_date"),
                pl.lit("evaluated").alias("bet_status"),
            ]
        )
        .select(
            recommendations_df.columns
            + ["payout", "bet_outcome", "evaluation_date", "bet_status"]
        )
        .unique(subset=["game_id", "date", "bet_type"])
    )

    bet_history_path = os.path.join(output_dir, "bet_history.parquet")
    existing_bet_history = _load_data_safe(bet_history_path)

    if not existing_bet_history.is_empty():
        existing_bet_history = existing_bet_history.filter(
            pl.col("evaluation_date") != evaluation_date.strftime("%Y-%m-%d")
        )
        updated_bet_history = pl.concat(
            [existing_bet_history, all_evaluated_bets_df], how="vertical"
        )
    else:
        updated_bet_history = all_evaluated_bets_df

    updated_bet_history.write_parquet(bet_history_path, compression="zstd")
    logger.info(f"Updated bet history saved to {bet_history_path}.")

    daily_evaluated_bets = all_evaluated_bets_df.filter(
        pl.col("evaluation_date") == evaluation_date.strftime("%Y-%m-%d")
    )
    daily_p_l = daily_evaluated_bets.filter(pl.col("payout").is_not_null())[
        "payout"
    ].sum()
    daily_wagered = daily_evaluated_bets.filter(pl.col("payout").is_not_null())[
        "wager_amount"
    ].sum()
    daily_roi = (daily_p_l / daily_wagered) if daily_wagered > 0 else 0.0

    total_system_profit_loss = updated_bet_history.filter(
        pl.col("payout").is_not_null()
    )["payout"].sum()
    total_system_wagered = updated_bet_history.filter(pl.col("payout").is_not_null())[
        "wager_amount"
    ].sum()
    total_system_roi = (
        (total_system_profit_loss / total_system_wagered)
        if total_system_wagered > 0
        else 0.0
    )

    logger.info(f"\nBacktesting Summary for {evaluation_date.strftime('%Y-%m-%d')}:")
    logger.info(f"  Daily Evaluated Bets: {daily_evaluated_bets.height}")
    logger.info(f"  Daily Profit/Loss: ${daily_p_l:.2f}")
    logger.info(f"  Daily Wagered: ${daily_wagered:.2f}")
    logger.info(f"  Daily ROI: {daily_roi:.2%}")
    logger.info(f"  Total System P/L to date: ${total_system_profit_loss:.2f}")
    logger.info(f"  Total System Wagered to date: ${total_system_wagered:.2f}")
    logger.info(f"  Total System ROI to date: {total_system_roi:.2%}")

    daily_summary_record = pl.DataFrame(
        {
            "date": [evaluation_date.strftime("%Y-%m-%d")],
            "daily_profit_loss": [daily_p_l],
            "daily_wagered": [daily_wagered],
            "daily_roi": [daily_roi],
            "total_system_profit_loss": [total_system_profit_loss],
            "total_system_wagered": [total_system_wagered],
            "total_system_roi": [total_system_roi],
        }
    )

    summary_path = os.path.join(output_dir, "backtest_results_summary.parquet")
    existing_summary = _load_data_safe(summary_path)

    if not existing_summary.is_empty() and "date" in existing_summary.columns:
        existing_summary = existing_summary.filter(
            pl.col("date") != evaluation_date.strftime("%Y-%m-%d")
        )
        final_summary = pl.concat(
            [existing_summary, daily_summary_record], how="vertical"
        )
    else:
        final_summary = daily_summary_record

    final_summary.write_parquet(summary_path, compression="zstd")
    logger.info(f"Backtesting results summary saved to {summary_path}.")
