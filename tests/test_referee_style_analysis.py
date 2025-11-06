"""
Pytest suite for referee_style_analysis.py
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from referee_style_analysis import (
    _aggregate_penalties,
    validate_df,
    _determine_phase,
)


def test_aggregate_penalties_with_mock_data():
    """Test penalty aggregation with mock data."""
    mock_penalties = pd.DataFrame({
        "season": [2024] * 10,
        "game_id": ["2024_01_KC_BAL"] * 10,
        "play_id": range(10),
        "penalty_yards": [10, 5, 15, 10, 5, 10, 15, 5, 10, 15],
        "score_swing": [0.5, -0.3, 0.8, -0.2, 0.1, 0.4, -0.5, 0.2, 0.3, -0.1],
        "total_plays": [65] * 10,
        "overtime_plays": [0] * 10,
    })
    
    result = _aggregate_penalties(mock_penalties)
    
    assert not result.empty, "Result should not be empty"
    assert "penalties" in result.columns
    assert "penalty_yards" in result.columns
    assert "flag_density" in result.columns
    assert "score_swing_mean_abs" in result.columns
    assert "score_swing_positive_rate" in result.columns
    
    row = result.iloc[0]
    assert row["penalties"] == 10
    assert row["penalty_yards"] == 100
    assert not pd.isna(row["score_swing_mean_abs"])
    assert not pd.isna(row["score_swing_positive_rate"])
    assert row["flag_density"] == pytest.approx(10 / 65, rel=1e-3)
    
    assert row["score_swing_mean_abs"] == pytest.approx(
        np.mean(np.abs([0.5, -0.3, 0.8, -0.2, 0.1, 0.4, -0.5, 0.2, 0.3, -0.1])),
        rel=1e-3
    )
    
    positive_rate = np.mean([1 if x > 0 else 0 for x in [0.5, -0.3, 0.8, -0.2, 0.1, 0.4, -0.5, 0.2, 0.3, -0.1]])
    assert row["score_swing_positive_rate"] == pytest.approx(positive_rate, rel=1e-3)


def test_aggregate_penalties_empty_dataframe():
    """Test penalty aggregation with empty DataFrame."""
    empty_df = pd.DataFrame()
    result = _aggregate_penalties(empty_df)
    
    assert result.empty
    assert "penalties" in result.columns
    assert "penalty_yards" in result.columns
    assert "flag_density" in result.columns


def test_validate_df_success():
    """Test validate_df with valid DataFrame."""
    df = pd.DataFrame({
        "game_id": ["2024_01_KC_BAL"],
        "season": [2024],
        "penalties": [10],
    })
    
    validate_df(df, ["game_id", "season", "penalties"], "test_df")


def test_validate_df_missing_columns():
    """Test validate_df raises ValueError on missing columns."""
    df = pd.DataFrame({
        "game_id": ["2024_01_KC_BAL"],
        "season": [2024],
    })
    
    with pytest.raises(ValueError) as exc_info:
        validate_df(df, ["game_id", "season", "penalties"], "test_df")
    
    assert "test_df" in str(exc_info.value)
    assert "penalties" in str(exc_info.value)


def test_determine_phase_early():
    """Test phase determination for early season."""
    assert _determine_phase(1, "REG") == "early"
    assert _determine_phase(6, "REG") == "early"


def test_determine_phase_mid():
    """Test phase determination for mid season."""
    assert _determine_phase(7, "REG") == "mid"
    assert _determine_phase(12, "REG") == "mid"


def test_determine_phase_late():
    """Test phase determination for late season."""
    assert _determine_phase(13, "REG") == "late"
    assert _determine_phase(18, "REG") == "late"
    assert _determine_phase(None, "WC") == "late"
    assert _determine_phase(None, "DIV") == "late"


def test_determine_phase_none():
    """Test phase determination with None values."""
    assert _determine_phase(None, "REG") == "mid"
    assert _determine_phase(None, None) == "mid"
