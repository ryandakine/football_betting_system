#!/usr/bin/env python3
"""
Integration workflow for Thursday Night Football predictions.
Runs the full HRM pipeline: Beast → Sniper → Conspiracy → Fusion
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data/referee_conspiracy")
PREDICTIONS_DIR = DATA_DIR / "predictions"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "tnf_predictions.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], description: str) -> bool:
    """Run a shell command and log the result."""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.debug(f"STDOUT:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Return code: {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed with exception: {e}")
        return False


def verify_data_exists() -> bool:
    """Check if required data files exist."""
    required_files = [
        DATA_DIR / "crew_game_log.parquet",
        DATA_DIR / "penalties_2018_2024.parquet",
        DATA_DIR / "schedules_2018_2024.parquet",
    ]
    
    missing = [f for f in required_files if not f.exists()]
    if missing:
        logger.error("Missing required data files:")
        for f in missing:
            logger.error(f"  - {f}")
        logger.error("Run nfl_referee_conspiracy_engine.py first to generate data.")
        return False
    
    logger.info("✓ All required data files found")
    return True


def fetch_live_odds() -> bool:
    """Fetch live NFL odds for tonight."""
    return run_command(
        ["python", "fetch_tonight_odds.py"],
        "Live Odds Fetch"
    )


def run_beast_model(train: bool = False, int8: bool = True, epochs: int = 5) -> bool:
    """Run the HRM Beast model."""
    cmd = ["python", "referee_style_analysis_hr.py", "--hrm"]
    
    if train:
        cmd.extend(["--train", "--epochs", str(epochs)])
    else:
        cmd.append("--no-train")
    
    if int8:
        cmd.append("--int8")
    
    return run_command(cmd, "HRM Beast Model")


def run_sniper_model() -> bool:
    """Run the HRM Sniper model."""
    return run_command(["python", "hrm_sniper.py"], "HRM Sniper Model")


def run_conspiracy_bot() -> bool:
    """Run the Conspiracy Bot."""
    return run_command(["python", "conspiracy_bot.py"], "Conspiracy Bot")


def run_fusion() -> bool:
    """Run the fusion layer."""
    return run_command(["python", "fuse_hr.py"], "Fusion Layer")


def display_predictions() -> None:
    """Display the final fused predictions."""
    fusion_path = DATA_DIR / "fused_hr_predictions.json"
    
    if not fusion_path.exists():
        logger.error(f"Fusion predictions file not found: {fusion_path}")
        return
    
    try:
        with open(fusion_path) as f:
            predictions = json.load(f)
        
        logger.info("=" * 80)
        logger.info("THURSDAY NIGHT FOOTBALL PREDICTIONS")
        logger.info("=" * 80)
        
        # Sort by game_id to get most recent
        sorted_games = sorted(predictions.items(), key=lambda x: x[0], reverse=True)
        
        # Display up to 5 most recent games
        for game_id, pred in sorted_games[:5]:
            logger.info(f"\nGame: {game_id}")
            logger.info(f"  Script Type: {pred.get('h_module', 'N/A')}")
            logger.info(f"  Under Probability: {pred.get('fused_under', 'N/A')}")
            logger.info(f"  Source Model: {pred.get('source', 'N/A')}")
            logger.info(f"  Beast Confidence: {pred.get('raw_beast_conf', 'N/A')}")
            logger.info(f"  Sniper Confidence: {pred.get('raw_sniper_conf', 'N/A')}")
        
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error reading predictions: {e}")


def archive_predictions() -> None:
    """Archive predictions with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    files_to_archive = [
        DATA_DIR / "hrm_predictions.json",
        DATA_DIR / "sniper_predictions.json",
        DATA_DIR / "conspiracy_predictions.json",
        DATA_DIR / "fused_hr_predictions.json",
    ]
    
    for src in files_to_archive:
        if src.exists():
            dst = PREDICTIONS_DIR / f"{timestamp}_{src.name}"
            dst.write_bytes(src.read_bytes())
            logger.info(f"Archived: {dst.name}")


def main():
    parser = argparse.ArgumentParser(description="Run TNF predictions workflow")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the Beast model (default: use existing checkpoint)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs if --train is set",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Disable int8 quantization",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Archive predictions with timestamp",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("TNF PREDICTIONS WORKFLOW")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 80)
    
    # Step 1: Verify data
    if not verify_data_exists():
        sys.exit(1)
    
    # Step 2: Fetch live odds
    if not fetch_live_odds():
        logger.warning("Live odds fetch failed, continuing with historical data")
    
    # Step 3: Run Beast
    if not run_beast_model(
        train=args.train,
        int8=not args.no_int8,
        epochs=args.epochs,
    ):
        logger.error("Beast model failed, aborting workflow")
        sys.exit(1)
    
    # Step 4: Run Sniper
    if not run_sniper_model():
        logger.error("Sniper model failed, aborting workflow")
        sys.exit(1)
    
    # Step 5: Run Conspiracy Bot
    if not run_conspiracy_bot():
        logger.warning("Conspiracy bot failed, continuing without it")
    
    # Step 6: Run Fusion
    if not run_fusion():
        logger.error("Fusion layer failed, aborting workflow")
        sys.exit(1)
    
    # Step 7: Display predictions
    display_predictions()
    
    # Step 8: Archive if requested
    if args.archive:
        archive_predictions()
    
    logger.info("=" * 80)
    logger.info(f"Workflow completed: {datetime.now()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
