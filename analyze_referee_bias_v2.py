#!/usr/bin/env python3
"""
Step 1: Refactored Referee Bias & Prediction Correlation Analyzer
- Eliminates data leakage (no realized totals in predictions)
- Typed data models for clarity and safety
- Configurable parameters via CLI
- Structured logging with context
- Explicit handling of missing pregame lines
"""

import json
import logging
import argparse
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from statistics import mean, stdev
from datetime import datetime
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# TYPED DATA MODELS
# ============================================================================

@dataclass
class PregameLines:
    """Pregame market lines and odds (NOT realized)."""
    spread: Optional[float] = None  # Home team spread
    total: Optional[float] = None   # Total line
    home_ml_odds: Optional[int] = None
    away_ml_odds: Optional[int] = None
    spread_odds: Optional[int] = None
    total_odds: Optional[int] = None
    spread_model_home_pct: Optional[float] = None  # Model prediction
    total_model_over_pct: Optional[float] = None
    
    def is_complete(self) -> bool:
        """Check if we have sufficient pregame data for reliable prediction."""
        return (
            self.spread is not None and
            self.total is not None and
            self.home_ml_odds is not None and
            self.away_ml_odds is not None
        )
    
    def has_leakage(self, realized_total: float, epsilon: float = 0.1) -> bool:
        """Check if this pregame data was contaminated with realized values."""
        if self.total is None:
            return False
        # If pregame total matches realized total exactly, likely data leakage
        return abs(self.total - realized_total) < epsilon


@dataclass
class GameResult:
    """Final game outcome (no pregame data)."""
    game_id: str
    date: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    total_score: int
    winner: str
    referee: str = "Unknown"
    
    @property
    def spread_result(self) -> int:
        """Realized spread (home team perspective)."""
        return self.home_score - self.away_score
    
    @property
    def home_won(self) -> bool:
        return self.home_score > self.away_score


@dataclass
class PredictionEvaluation:
    """Prediction vs actual outcome."""
    game_id: str
    referee: str
    market: str  # "spread", "total", "moneyline"
    predicted_pick: str
    actual_outcome: str  # "home", "away", "over", "under"
    confidence: float
    is_correct: bool
    has_input_leakage: bool = False  # Flag if inputs had leakage
    degraded: bool = False  # Flag if prediction was in degraded mode


@dataclass
class RefereeStats:
    """Aggregated statistics per referee."""
    name: str
    games: int = 0
    home_wins: int = 0
    away_wins: int = 0
    total_points_per_game: List[float] = field(default_factory=list)
    spread_results: List[int] = field(default_factory=list)
    predictions_total: int = 0
    predictions_correct: int = 0
    prediction_by_market: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "spread": {"total": 0, "correct": 0},
        "total": {"total": 0, "correct": 0},
        "moneyline": {"total": 0, "correct": 0},
    })
    
    @property
    def home_win_pct(self) -> float:
        if self.games == 0:
            return 0.0
        return self.home_wins / self.games
    
    @property
    def prediction_accuracy(self) -> float:
        if self.predictions_total == 0:
            return 0.0
        return self.predictions_correct / self.predictions_total


# ============================================================================
# CONFIGURATION & VALIDATION
# ============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for analysis."""
    data_dir: Path
    input_file: str = "nfl_games_2015_2025.json"
    min_games_per_ref: int = 20
    home_bias_threshold: float = 0.55
    accuracy_threshold: float = 0.55
    output_dir: Path = field(default_factory=lambda: Path("data/analysis_output"))
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MAIN ANALYZER (REFACTORED)
# ============================================================================

class RefereeBiasAnalyzerV2:
    """
    Referee bias analysis with data leakage prevention and typed models.
    
    Key improvements:
    - No use of realized totals in predictions (prevents data leakage)
    - Explicit tracking of missing/leakage in inputs
    - Typed data models for clarity
    - Structured logging with context
    - Configurable parameters
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.referee_stats: Dict[str, RefereeStats] = {}
        self.predictions: List[PredictionEvaluation] = []
        self.input_validation_report = {
            "total_games": 0,
            "games_with_complete_lines": 0,
            "games_with_missing_lines": 0,
            "games_with_leakage": 0,
        }
        
        # Import council here to avoid circular dependency
        from unified_betting_intelligence import NarrativeIntegratedAICouncil
        self.council = NarrativeIntegratedAICouncil()
    
    def load_and_validate_games(self) -> Tuple[List[Dict[str, Any]], str]:
        """
        Load games and compute input data hash for reproducibility.
        Handles both list and dict (keyed) formats.
        
        Returns:
            (games, input_hash)
        """
        input_file = self.config.data_dir / self.config.input_file
        
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            raise FileNotFoundError(input_file)
        
        with open(input_file, "rb") as f:
            content = f.read()
        
        data = json.loads(content)
        input_hash = hashlib.sha256(content).hexdigest()[:12]
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            games = list(data.values())
        elif isinstance(data, list):
            games = data
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")
        
        logger.info(f"Loaded {len(games)} games from {input_file}")
        logger.info(f"Input hash: {input_hash}")
        
        return games, input_hash
    
    def validate_game_inputs(self, game: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate game record for required fields and data quality.
        Referee field is optional (defaults to 'Unknown').
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        required = ["game_id", "home_team", "away_team", "home_score", "away_score"]
        for field in required:
            if field not in game or game[field] is None or (isinstance(game[field], str) and not game[field]):
                errors.append(f"Missing required field: {field}")
        
        return len(errors) == 0, errors
    
    def extract_pregame_lines(self, game: Dict[str, Any]) -> Optional[PregameLines]:
        """
        Extract pregame market data from game record.
        
        If the game lacks pregame lines, return None (not realized total).
        If leakage detected, flag it but continue.
        """
        try:
            realized_total = game.get("total_score", 0)
            
            pregame = PregameLines(
                spread=game.get("spread"),
                total=game.get("total"),
                home_ml_odds=game.get("home_ml_odds"),
                away_ml_odds=game.get("away_ml_odds"),
                spread_odds=game.get("spread_odds"),
                total_odds=game.get("total_odds"),
                spread_model_home_pct=game.get("spread_model_home_pct"),
                total_model_over_pct=game.get("total_model_over_pct"),
            )
            
            # Check for leakage
            if pregame.has_leakage(realized_total):
                logger.warning(
                    f"Possible data leakage in {game.get('game_id')}: "
                    f"pregame_total={pregame.total}, realized_total={realized_total}"
                )
                self.input_validation_report["games_with_leakage"] += 1
            
            if pregame.is_complete():
                self.input_validation_report["games_with_complete_lines"] += 1
                return pregame
            else:
                self.input_validation_report["games_with_missing_lines"] += 1
                return None
        
        except Exception as e:
            logger.error(f"Failed to extract pregame lines from {game.get('game_id')}: {e}")
            return None
    
    def analyze_referee_patterns(self, games: List[Dict[str, Any]]) -> None:
        """
        First pass: Aggregate referee statistics (bias patterns).
        """
        logger.info(f"Analyzing referee bias patterns in {len(games)} games...")
        self.input_validation_report["total_games"] = len(games)
        
        for game in games:
            is_valid, errors = self.validate_game_inputs(game)
            if not is_valid:
                logger.debug(f"Skipping invalid game {game.get('game_id')}: {errors}")
                continue
            
            referee = game.get("referee", "Unknown")
            
            # Initialize referee stats if needed
            if referee not in self.referee_stats:
                self.referee_stats[referee] = RefereeStats(name=referee)
            
            stats = self.referee_stats[referee]
            
            # Aggregate game outcomes
            result = GameResult(
                game_id=game.get("game_id", ""),
                date=game.get("date", ""),
                home_team=game.get("home_team", ""),
                away_team=game.get("away_team", ""),
                home_score=int(game.get("home_score", 0)),
                away_score=int(game.get("away_score", 0)),
                total_score=int(game.get("total_score", 0)),
                winner=game.get("winner", ""),
                referee=referee,
            )
            
            stats.games += 1
            stats.total_points_per_game.append(result.total_score)
            stats.spread_results.append(result.spread_result)
            
            if result.home_won:
                stats.home_wins += 1
            else:
                stats.away_wins += 1
        
        logger.info(f"Analyzed {len(self.referee_stats)} unique referees")
    
    def correlate_with_predictions(self, games: List[Dict[str, Any]]) -> None:
        """
        Second pass: Generate predictions using ONLY pregame lines.
        Skip games without valid pregame data.
        
        Key: NO realized totals are passed to the council.
        """
        logger.info("Correlating predictions with referee data (pregame lines only)...")
        
        skipped_no_lines = 0
        skipped_errors = 0
        processed = 0
        
        for i, game in enumerate(games):
            if i % 500 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(games)} games...")
            
            is_valid, _ = self.validate_game_inputs(game)
            if not is_valid:
                skipped_errors += 1
                continue
            
            # Extract pregame lines (THIS EXCLUDES REALIZED TOTALS)
            pregame = self.extract_pregame_lines(game)
            if pregame is None:
                skipped_no_lines += 1
                continue
            
            processed += 1
            
            try:
                referee = game.get("referee", "Unknown")
                
                # Build prediction input with ONLY pregame data
                game_data = {
                    "game_id": game.get("game_id", ""),
                    "home_team": game.get("home_team", ""),
                    "away_team": game.get("away_team", ""),
                    "spread": pregame.spread,
                    "total": pregame.total,  # PREGAME TOTAL, NOT REALIZED
                    "home_ml_odds": pregame.home_ml_odds,
                    "away_ml_odds": pregame.away_ml_odds,
                    "spread_odds": pregame.spread_odds,
                    "total_odds": pregame.total_odds,
                    "spread_model_home_pct": pregame.spread_model_home_pct or 0.5,
                    "total_model_over_pct": pregame.total_model_over_pct or 0.5,
                }
                
                # Get prediction
                prediction = self.council.make_unified_prediction(game_data)
                
                # Evaluate against actual outcome
                result = GameResult(
                    game_id=game.get("game_id", ""),
                    date=game.get("date", ""),
                    home_team=game.get("home_team", ""),
                    away_team=game.get("away_team", ""),
                    home_score=int(game.get("home_score", 0)),
                    away_score=int(game.get("away_score", 0)),
                    total_score=int(game.get("total_score", 0)),
                    winner=game.get("winner", ""),
                    referee=referee,
                )
                
                # Spread evaluation
                if prediction.spread_prediction:
                    spread_pick = prediction.spread_prediction.pick
                    actual_winner = "home" if result.home_won else "away"
                    is_correct = (spread_pick == actual_winner)
                    
                    eval_record = PredictionEvaluation(
                        game_id=game.get("game_id", ""),
                        referee=referee,
                        market="spread",
                        predicted_pick=spread_pick,
                        actual_outcome=actual_winner,
                        confidence=prediction.spread_prediction.confidence,
                        is_correct=is_correct,
                        degraded=prediction.degraded,
                    )
                    
                    self.predictions.append(eval_record)
                    
                    # Update referee stats
                    stats = self.referee_stats[referee]
                    stats.predictions_total += 1
                    if is_correct:
                        stats.predictions_correct += 1
                    stats.prediction_by_market["spread"]["total"] += 1
                    if is_correct:
                        stats.prediction_by_market["spread"]["correct"] += 1
                
                # Total evaluation (if we have pregame total)
                if prediction.total_prediction and pregame.total is not None:
                    over_under = "over" if result.total_score > pregame.total else "under"
                    total_pick = prediction.total_prediction.pick
                    is_correct = (total_pick == over_under)
                    
                    eval_record = PredictionEvaluation(
                        game_id=game.get("game_id", ""),
                        referee=referee,
                        market="total",
                        predicted_pick=total_pick,
                        actual_outcome=over_under,
                        confidence=prediction.total_prediction.confidence,
                        is_correct=is_correct,
                        degraded=prediction.degraded,
                    )
                    
                    self.predictions.append(eval_record)
                    
                    # Update referee stats
                    stats.prediction_by_market["total"]["total"] += 1
                    if is_correct:
                        stats.prediction_by_market["total"]["correct"] += 1
            
            except Exception as e:
                logger.error(f"Failed to process prediction for {game.get('game_id')}: {e}")
                skipped_errors += 1
        
        logger.info(
            f"Prediction correlation complete: "
            f"processed={processed}, skipped_no_lines={skipped_no_lines}, "
            f"skipped_errors={skipped_errors}"
        )
    
    def generate_report(self, input_hash: str) -> Dict[str, Any]:
        """Generate comprehensive analysis report with configuration snapshot."""
        
        overall_accuracy = 0.0
        if len(self.predictions) > 0:
            overall_accuracy = sum(1 for p in self.predictions if p.is_correct) / len(self.predictions)
        
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "2.0",
                "input_hash": input_hash,
                "config": asdict(self.config),
            },
            "input_validation": self.input_validation_report,
            "global_metrics": {
                "total_predictions": len(self.predictions),
                "correct_predictions": sum(1 for p in self.predictions if p.is_correct),
                "overall_accuracy": overall_accuracy,
                "total_unique_referees": len(self.referee_stats),
                "referees_with_min_games": sum(
                    1 for r in self.referee_stats.values()
                    if r.games >= self.config.min_games_per_ref
                ),
            },
            "referee_stats": {
                ref: asdict(stats)
                for ref, stats in sorted(
                    self.referee_stats.items(),
                    key=lambda x: x[1].games,
                    reverse=True
                )
            },
        }
        
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS REPORT SUMMARY")
        logger.info("="*80)
        logger.info(f"Input Validation:")
        logger.info(f"  Total games: {self.input_validation_report['total_games']}")
        logger.info(f"  With complete lines: {self.input_validation_report['games_with_complete_lines']}")
        logger.info(f"  With missing lines: {self.input_validation_report['games_with_missing_lines']}")
        logger.info(f"  With detected leakage: {self.input_validation_report['games_with_leakage']}")
        logger.info(f"\nPrediction Performance:")
        logger.info(f"  Total predictions: {len(self.predictions)}")
        logger.info(f"  Overall accuracy: {overall_accuracy:.1%}")
        
        return report
    
    def save_report(self, report: Dict[str, Any]) -> Path:
        """Save report to JSON file."""
        output_file = self.config.output_dir / "referee_bias_analysis_v2.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {output_file}")
        return output_file


# ============================================================================
# CLI & MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Referee Bias Analysis (Step 1: Data Leakage Fixes)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/historical_games"),
        help="Directory containing game data",
    )
    parser.add_argument(
        "--input-file",
        default="nfl_games_2015_2025.json",
        help="Input games file name",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=20,
        help="Minimum games per referee to include in analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis_output"),
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    config = AnalysisConfig(
        data_dir=args.data_dir,
        input_file=args.input_file,
        min_games_per_ref=args.min_games,
        output_dir=args.output_dir,
    )
    
    analyzer = RefereeBiasAnalyzerV2(config)
    
    try:
        games, input_hash = analyzer.load_and_validate_games()
        analyzer.analyze_referee_patterns(games)
        analyzer.correlate_with_predictions(games)
        report = analyzer.generate_report(input_hash)
        analyzer.save_report(report)
        
        logger.info("\n✅ Step 1 complete: Referee bias analysis with data leakage fixes")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
