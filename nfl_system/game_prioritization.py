#!/usr/bin/env python3
"""
NFL Game Prioritization
======================

Heuristic prioritizer that orders NFL games for downstream AI analysis.
This mirrors the college football prioritizer but adapts the weighting
scheme to professional football schedules, primetime volatility, and the
conference structure of the NFL.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from nfl_system.gold_standard_nfl_config import NFLGoldStandardConfig, get_nfl_config

logger = logging.getLogger(__name__)


@dataclass
class PrioritizedNFLGame:
    """Lightweight container describing a prioritized NFL matchup."""

    game: Dict[str, Any]
    priority: float
    reasons: List[str]

    @property
    def matchup(self) -> str:
        home = self.game.get("home_team", "HOME")
        away = self.game.get("away_team", "AWAY")
        return f"{away} @ {home}"


class NFLGamePrioritizer:
    """NFL-specific game ordering heuristics."""

    def __init__(self, config: Optional[NFLGoldStandardConfig] = None) -> None:
        self.config = config or get_nfl_config()

        # Baseline conference weights; these can be tuned via backtesting.
        self.conference_weights: Dict[str, float] = {
            "afc": 0.9,
            "nfc": 0.9,
            "afc east": 1.1,
            "afc north": 1.05,
            "afc south": 0.95,
            "afc west": 1.0,
            "nfc east": 1.05,
            "nfc north": 1.0,
            "nfc south": 0.95,
            "nfc west": 1.02,
            "international": 1.2,
        }

        pattern = self.config.pattern_weights
        self.pattern_weights = {
            "thursday": pattern.thursday,
            "sunday_early": pattern.sunday_early,
            "sunday_late": pattern.sunday_late,
            "sunday_night": pattern.sunday_night,
            "monday": pattern.monday,
        }
        self.weather_bonus = pattern.weather_bonus
        self.obscure_bonus = pattern.obscure_bonus

        self.prioritization_log: List[PrioritizedNFLGame] = []
        self.dynamic_weights: Dict[str, float] = {}
        self.weights_storage_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "football"
            / "nfl"
            / "conference_weights.json"
        )
        self._load_dynamic_weights()

    def _load_dynamic_weights(self) -> None:
        """Load persisted conference weighting overrides when present."""
        try:
            if not self.weights_storage_path.is_file():
                return

            with self.weights_storage_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if not isinstance(payload, dict):
                logger.warning("NFL conference weights file malformed: %s", self.weights_storage_path)
                return

            overrides: Dict[str, float] = {}
            for key, value in payload.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue

                normalized = key.strip().lower()
                if not normalized:
                    continue

                overrides[normalized] = self._clamp_weight(numeric)

            if overrides:
                self.dynamic_weights = overrides
                self.conference_weights.update(overrides)
                logger.info("Loaded %d NFL conference weight overrides", len(overrides))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read NFL conference weight overrides: %s", exc)

    @staticmethod
    def _clamp_weight(value: float, *, low: float = 0.3, high: float = 1.5) -> float:
        return max(low, min(high, float(value)))

    def optimize_processing_order(self, games: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return games sorted by priority descending."""
        prioritized = self._prioritize_games(games)
        return [entry.game for entry in prioritized]

    def _prioritize_games(self, games: Sequence[Dict[str, Any]]) -> List[PrioritizedNFLGame]:
        self.prioritization_log = []
        for game in games:
            score, reasons = self._score_game(game)
            self.prioritization_log.append(PrioritizedNFLGame(game=game, priority=score, reasons=reasons))

        self.prioritization_log.sort(key=lambda entry: entry.priority, reverse=True)
        return self.prioritization_log

    def _score_game(self, game: Dict[str, Any]) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        score = 1.0

        # Conference / division weighting
        conference = str(game.get("conference") or "").strip().lower()
        if not conference and game.get("home_team"):
            conference = self._infer_conference(game.get("home_team"))

        if conference:
            conf_weight = self.conference_weights.get(conference, 0.85)
            score *= conf_weight
            reasons.append(f"conference weight {conf_weight:.2f}")

        # Edge potential heuristics
        edge_tag = str(game.get("edge_potential") or "").upper()
        edge_weight = {
            "ULTRA": 1.25,
            "HIGH": 1.15,
            "MEDIUM": 1.0,
            "LOW": 0.85,
        }.get(edge_tag, 0.95)
        score *= edge_weight
        reasons.append(f"edge tag {edge_tag or 'UNKNOWN'}={edge_weight:.2f}")

        # Market disparity between public and sharp percentages
        public_pct = self._safe_float(game.get("public_percentage"), default=0.5)
        sharp_pct = self._safe_float(game.get("sharp_percentage"), default=0.5)
        delta = abs(sharp_pct - public_pct)
        market_weight = 1.0 + min(delta * 1.5, 0.35)
        score *= market_weight
        reasons.append(f"market delta {delta:.2f} -> {market_weight:.2f}")

        # Line movement consideration
        line_move = self._safe_float(game.get("line_movement"), default=0.0)
        if line_move:
            move_weight = 1.0 + min(abs(line_move) / 3.5, 0.25)
            score *= move_weight
            reasons.append(f"line move {line_move:+.1f} -> {move_weight:.2f}")

        # Primetime / day of week boost
        kickoff_bucket = self._bucket_game_time(game)
        if kickoff_bucket and kickoff_bucket in self.pattern_weights:
            time_weight = 1.0 + (self.pattern_weights[kickoff_bucket] - 0.5)
            score *= time_weight
            reasons.append(f"{kickoff_bucket} boost {time_weight:.2f}")

        # Weather considerations
        if str(game.get("weather_risk", "")).lower() in {"high", "severe"}:
            score *= 1.0 + self.weather_bonus
            reasons.append("weather volatility bonus")

        # Hidden gems (lower liquidity but higher edge)
        if conference in {c.lower() for c in self.config.conferences.hidden_gems}:
            score *= 1.0 + self.obscure_bonus
            reasons.append("hidden gem bonus")

        return score, reasons

    def _bucket_game_time(self, game: Dict[str, Any]) -> Optional[str]:
        """
        Bucket the start time into schedule categories:
        thursday, sunday_early/late/night, monday.
        """
        kickoff_raw = game.get("commence_time") or game.get("kickoff") or game.get("game_time")
        if not kickoff_raw:
            return None

        try:
            if isinstance(kickoff_raw, (int, float)):
                kickoff = datetime.fromtimestamp(float(kickoff_raw))
            else:
                kickoff = datetime.fromisoformat(str(kickoff_raw).replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return None

        weekday = kickoff.weekday()  # Monday=0 ... Sunday=6
        hour = kickoff.hour

        if weekday == 3:  # Thursday
            return "thursday"
        if weekday == 0:  # Monday
            return "monday"
        if weekday == 6:  # Sunday
            if hour < 15:
                return "sunday_early"
            if hour < 19:
                return "sunday_late"
            return "sunday_night"
        if weekday == 5:  # Saturday international / late season
            return "sunday_late"
        return None

    def _infer_conference(self, team_name: str) -> str:
        name = str(team_name or "").lower()
        mapping = {
            "patriots": "afc east",
            "bills": "afc east",
            "dolphins": "afc east",
            "jets": "afc east",
            "ravens": "afc north",
            "bengals": "afc north",
            "steelers": "afc north",
            "browns": "afc north",
            "texans": "afc south",
            "colts": "afc south",
            "jaguars": "afc south",
            "titans": "afc south",
            "chiefs": "afc west",
            "chargers": "afc west",
            "broncos": "afc west",
            "raiders": "afc west",
            "cowboys": "nfc east",
            "giants": "nfc east",
            "eagles": "nfc east",
            "commanders": "nfc east",
            "packers": "nfc north",
            "vikings": "nfc north",
            "bears": "nfc north",
            "lions": "nfc north",
            "buccaneers": "nfc south",
            "falcons": "nfc south",
            "panthers": "nfc south",
            "saints": "nfc south",
            "49ers": "nfc west",
            "seahawks": "nfc west",
            "cardinals": "nfc west",
            "rams": "nfc west",
        }
        return mapping.get(name, "")

    @staticmethod
    def _safe_float(value: Any, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_prioritization_log(self) -> List[PrioritizedNFLGame]:
        """Return the most recent prioritization breakdown."""
        return list(self.prioritization_log)


__all__ = ["NFLGamePrioritizer", "PrioritizedNFLGame"]
