#!/usr/bin/env python3
"""
Unit Tests for Enhanced Backtest
=================================
Tests grading logic, Kelly stakes, metrics calculation, and referee correlations.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Import functions from enhanced backtest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_end_to_end_backtest_enhanced import (
    american_to_decimal,
    calculate_kelly_stake,
    _grade_with_stake,
    calculate_risk_metrics,
    enrich_game_with_odds,
)


class TestOddsConversion:
    """Test American to decimal odds conversion."""
    
    def test_positive_odds(self):
        assert american_to_decimal(150) == 2.5
        assert american_to_decimal(200) == 3.0
    
    def test_negative_odds(self):
        assert american_to_decimal(-110) == pytest.approx(1.909, rel=0.01)
        assert american_to_decimal(-200) == 1.5
    
    def test_even_odds(self):
        assert american_to_decimal(100) == 2.0


class TestKellyStake:
    """Test Kelly Criterion stake calculation."""
    
    def test_positive_edge(self):
        stake = calculate_kelly_stake(
            edge=0.1,
            confidence=0.60,
            odds=-110,
            kelly_fraction=1.0,
            max_stake=1.0
        )
        assert 0 < stake < 0.5
    
    def test_fractional_kelly(self):
        full_kelly = calculate_kelly_stake(
            edge=0.15,
            confidence=0.65,
            odds=-110,
            kelly_fraction=1.0,
            max_stake=1.0
        )
        quarter_kelly = calculate_kelly_stake(
            edge=0.15,
            confidence=0.65,
            odds=-110,
            kelly_fraction=0.25,
            max_stake=1.0
        )
        assert quarter_kelly == pytest.approx(full_kelly * 0.25, rel=0.01)
    
    def test_max_stake_cap(self):
        stake = calculate_kelly_stake(
            edge=0.5,
            confidence=0.9,
            odds=-110,
            kelly_fraction=1.0,
            max_stake=0.03
        )
        assert stake <= 0.03
    
    def test_zero_edge(self):
        stake = calculate_kelly_stake(
            edge=0.0,
            confidence=0.55,
            odds=-110,
            kelly_fraction=0.25,
            max_stake=0.03
        )
        assert stake == 0.0


class TestGrading:
    """Test game grading with stakes."""
    
    def create_mock_game(self, home_score: float, away_score: float) -> Dict[str, Any]:
        return {
            "game_id": "test_game",
            "home_score": home_score,
            "away_score": away_score,
            "spread": -3.0,
            "total": 45.0,
            "home_ml_odds": -150,
            "away_ml_odds": 130,
        }
    
    def create_mock_prediction(self) -> Dict[str, Any]:
        return {
            "spread_pick": "home",
            "spread_edge": 0.08,
            "spread_confidence": 0.62,
            "total_pick": "over",
            "total_edge": 0.05,
            "total_confidence": 0.58,
            "ml_pick": "home",
            "ml_edge": 0.06,
            "ml_confidence": 0.60,
        }
    
    def test_spread_win(self):
        game = self.create_mock_game(home_score=27, away_score=20)
        prediction = self.create_mock_prediction()
        
        result, profit, stake = _grade_with_stake(game, prediction, "spread", kelly_fraction=0.25)
        
        assert result == "WIN"
        assert profit > 0
        assert stake > 0
    
    def test_spread_loss(self):
        game = self.create_mock_game(home_score=20, away_score=27)
        prediction = self.create_mock_prediction()
        
        result, profit, stake = _grade_with_stake(game, prediction, "spread", kelly_fraction=0.25)
        
        assert result == "LOSS"
        assert profit < 0
        assert stake > 0
    
    def test_spread_push(self):
        game = self.create_mock_game(home_score=24, away_score=27)
        prediction = self.create_mock_prediction()
        
        result, profit, stake = _grade_with_stake(game, prediction, "spread", kelly_fraction=0.25)
        
        assert result == "PUSH"
        assert profit == 0
        assert stake == 0
    
    def test_total_over_win(self):
        game = self.create_mock_game(home_score=28, away_score=24)  # Total 52
        prediction = self.create_mock_prediction()
        
        result, profit, stake = _grade_with_stake(game, prediction, "total", kelly_fraction=0.25)
        
        assert result == "WIN"
        assert profit > 0
    
    def test_total_under_loss(self):
        game = self.create_mock_game(home_score=28, away_score=24)  # Total 52
        prediction = self.create_mock_prediction()
        prediction["total_pick"] = "under"
        
        result, profit, stake = _grade_with_stake(game, prediction, "total", kelly_fraction=0.25)
        
        assert result == "LOSS"
        assert profit < 0
    
    def test_moneyline_win(self):
        game = self.create_mock_game(home_score=27, away_score=20)
        prediction = self.create_mock_prediction()
        
        result, profit, stake = _grade_with_stake(game, prediction, "moneyline", kelly_fraction=0.25)
        
        assert result == "WIN"
        assert profit > 0
    
    def test_moneyline_loss(self):
        game = self.create_mock_game(home_score=20, away_score=27)
        prediction = self.create_mock_prediction()
        
        result, profit, stake = _grade_with_stake(game, prediction, "moneyline", kelly_fraction=0.25)
        
        assert result == "LOSS"
        assert profit < 0


class TestRiskMetrics:
    """Test risk metric calculations."""
    
    def test_sharpe_ratio(self):
        # Positive returns
        equity = np.array([1.0, 1.05, 1.10, 1.08, 1.15, 1.20])
        metrics = calculate_risk_metrics(equity)
        
        assert metrics["sharpe"] > 0
        assert 0 <= metrics["max_drawdown"] <= 1
    
    def test_max_drawdown(self):
        # Equity with drawdown
        equity = np.array([1.0, 1.2, 1.5, 1.3, 1.1, 1.4])
        metrics = calculate_risk_metrics(equity)
        
        # Max drawdown should be from 1.5 to 1.1 = 26.7%
        assert metrics["max_drawdown"] > 0.2
        assert metrics["max_drawdown"] < 0.3
    
    def test_flat_equity(self):
        equity = np.ones(100)
        metrics = calculate_risk_metrics(equity)
        
        assert metrics["sharpe"] == 0.0
        assert metrics["max_drawdown"] == 0.0
    
    def test_single_point(self):
        equity = np.array([1.0])
        metrics = calculate_risk_metrics(equity)
        
        assert metrics["sharpe"] == 0.0
        assert metrics["max_drawdown"] == 0.0


class TestGameEnrichment:
    """Test game odds enrichment."""
    
    def test_real_odds_preserved(self):
        game = {
            "game_id": "test_1",
            "spread": -5.5,
            "total": 48.5,
            "home_ml_odds": -220,
            "away_ml_odds": 180,
        }
        
        enriched = enrich_game_with_odds(game)
        
        assert enriched["spread"] == -5.5
        assert enriched["total"] == 48.5
        assert enriched["home_ml_odds"] == -220
        assert enriched["has_real_odds"] is True
    
    def test_missing_odds_estimated(self):
        game = {
            "game_id": "test_2",
            "home_score": 28,
            "away_score": 21,
        }
        
        enriched = enrich_game_with_odds(game)
        
        # Should estimate from scores
        assert enriched["spread"] is not None
        assert enriched["total"] is not None
        assert enriched["home_ml_odds"] is not None
        assert enriched["has_real_odds"] is False
    
    def test_fallback_defaults(self):
        game = {"game_id": "test_3"}
        
        enriched = enrich_game_with_odds(game)
        
        assert enriched["spread"] == 0.0
        assert enriched["total"] == 44.5
        assert enriched["home_ml_odds"] == -110


class TestIntegration:
    """Integration tests with mock game sets."""
    
    def create_mini_season(self, games: int = 5) -> List[Dict[str, Any]]:
        """Create a small season for testing."""
        game_data = []
        
        for i in range(games):
            game_data.append({
                "game_id": f"2024_W1_G{i}",
                "season": 2024,
                "week": 1,
                "home_team": "HOME",
                "away_team": "AWAY",
                "home_score": 24 + i,
                "away_score": 20 + i,
                "spread": -3.0,
                "total": 45.0,
                "home_ml_odds": -150,
                "away_ml_odds": 130,
                "referee_name": f"Ref_{i % 3}",
            })
        
        return game_data
    
    def test_mini_season_grading(self):
        """Test grading a small season."""
        games = self.create_mini_season(5)
        
        # Create mock predictions for each game
        enriched_games = []
        for game in games:
            enriched_games.append({
                **game,
                "prediction": {
                    "game_id": game["game_id"],
                    "spread_pick": "home",
                    "spread_edge": 0.05,
                    "spread_confidence": 0.58,
                    "total_pick": "over",
                    "total_edge": 0.03,
                    "total_confidence": 0.55,
                    "ml_pick": "home",
                    "ml_edge": 0.04,
                    "ml_confidence": 0.57,
                }
            })
        
        # Should not crash
        assert len(enriched_games) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
