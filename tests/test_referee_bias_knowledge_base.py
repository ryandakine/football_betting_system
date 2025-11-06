import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from referee_bias_knowledge_base import RefereeBiasKnowledgeBase


def test_lookup_returns_expected_record():
    kb = RefereeBiasKnowledgeBase()
    record = kb.lookup_game(2024, 7, "Minnesota Vikings", "Philadelphia Eagles")
    assert record is not None, "Week 7 Vikings vs Eagles record should exist"
    assert record["referee"] == "Bill Vinovich"
    assert record["anti_home"] is True
    assert pytest.approx(record["total"], rel=1e-6) == 43.5


def test_evaluate_game_triggers_auto_under_for_bengals():
    kb = RefereeBiasKnowledgeBase()
    game_data = {
        "season": 2024,
        "week": 7,
        "home_team": "Cincinnati Bengals",
        "away_team": "Pittsburgh Steelers",
    }
    result = kb.evaluate_game(game_data, base_probability=0.47, current_confidence=0.63)
    assert result is not None
    assert result["trigger_auto_under"] is True
    assert result["anti_home"] is True
    assert result["probability"] < 0.47
    assert result["confidence"] <= 0.63
    assert "Clay Martin" in result["alert"]
    plays_blob = " ".join(result.get("recommended_plays", []))
    assert "Under" in plays_blob
