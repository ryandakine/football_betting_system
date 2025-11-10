#!/usr/bin/env python3
"""
WNBA Game Prioritization
=======================

Provides heuristics for ordering WNBA games before running
expensive AI analysis. The prioritizer promotes high-leverage matchups.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass
class PrioritizedGame:
    """Lightweight container for prioritization metadata."""

    game: Dict[str, Any]
    priority: float
    reasons: List[str]

    @property
    def matchup(self) -> str:
        return f"{self.game.get('away_team', 'TBD')} @ {self.game.get('home_team', 'TBD')}"


class GamePrioritizer:
    """Heuristic prioritizer for WNBA game analysis."""

    def __init__(self) -> None:
        # Team weighting based on market presence and performance
        self.team_weights = {
            "las vegas aces": 0.95,
            "new york liberty": 0.92,
            "minnesota lynx": 0.88,
            "connecticut sun": 0.85,
            "seattle storm": 0.88,
            "chicago sky": 0.80,
            "atlanta dream": 0.75,
            "phoenix mercury": 0.82,
            "dallas wings": 0.78,
            "washington mystics": 0.80,
            "indiana fever": 0.75,
            "los angeles sparks": 0.78,
        }

        self.edge_potential_weights = {
            "ULTRA": 0.85,
            "HIGH": 0.70,
            "MEDIUM": 0.45,
            "LOW": 0.20,
        }

        self.prioritization_log: List[PrioritizedGame] = []
        self.dynamic_weights: Dict[str, float] = {}
        self.weights_storage_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "basketball"
            / "wnba"
            / "team_weights.json"
        )
        self._load_dynamic_weights()

    def _load_dynamic_weights(self) -> None:
        """Load persisted team weighting overrides, if present."""
        try:
            if not self.weights_storage_path.is_file():
                return

            with self.weights_storage_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if not isinstance(payload, dict):
                logger.warning("Team weights file is not a dictionary: %s", self.weights_storage_path)
                return

            loaded: Dict[str, float] = {}
            for key, value in payload.items():
                try:
                    weight = float(value)
                except (TypeError, ValueError):
                    logger.debug("Skipping non-numeric weight override for %s", key)
                    continue

                normalized_key = key.strip().lower()
                if not normalized_key:
                    continue

                loaded[normalized_key] = self._clamp_weight(weight)

            if loaded:
                self.dynamic_weights = loaded
                self.team_weights.update(loaded)
                logger.info("Loaded %d dynamic team weight overrides", len(loaded))
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning("Failed to load team weights from %s: %s", self.weights_storage_path, exc)

    @staticmethod
    def _clamp_weight(weight: float) -> float:
        """Ensure weights remain within stable bounds."""
        return max(0.25, min(1.25, round(weight, 4)))

    def optimize_processing_order(self, games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return the list of games ordered by descending priority."""
        if not games:
            self.prioritization_log = []
            return []

        prioritized: List[PrioritizedGame] = []
        for game in games:
            priority, reasons = self._score_game(game)
            prioritized.append(PrioritizedGame(game=game, priority=priority, reasons=reasons))

        prioritized.sort(key=lambda item: item.priority, reverse=True)
        self.prioritization_log = prioritized

        return [item.game for item in prioritized]

    def get_prioritization_log(self) -> List[Dict[str, Any]]:
        """Expose prioritization metadata for logging and reporting."""
        return [
            {
                "matchup": item.matchup,
                "priority": item.priority,
                "reasons": item.reasons,
                "home_team": item.game.get("home_team"),
                "away_team": item.game.get("away_team"),
            }
            for item in self.prioritization_log
        ]

    def _score_game(self, game: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Compute a heuristic priority score for a game."""
        reasons: List[str] = []

        home_team = (game.get("home_team") or "").lower()
        away_team = (game.get("away_team") or "").lower()
        edge_potential = (game.get("edge_potential") or "MEDIUM").upper()
        spread = abs(float(game.get("spread") or 0.0))
        total = float(game.get("total") or 0.0)
        public_pct = float(game.get("public_percentage") or 0.5)
        sharp_pct = float(game.get("sharp_percentage") or 0.5)
        line_move = abs(float(game.get("line_movement") or 0.0))

        # Base edge proxy using edge potential tag.
        edge_weight = self.edge_potential_weights.get(edge_potential, 0.35)
        reasons.append(f"Edge potential weight: {edge_weight:.3f} ({edge_potential})")

        # Team weighting
        home_weight = self.team_weights.get(home_team, 0.70)
        away_weight = self.team_weights.get(away_team, 0.70)
        team_weight = (home_weight + away_weight) / 2
        reasons.append(f"Team weight: {team_weight:.3f}")

        # Market signal: sharp/public discrepancy.
        sharp_delta = max(sharp_pct - public_pct, 0.0)
        sharp_bonus = sharp_delta * 0.5  # WNBA sharp money is valuable
        if sharp_bonus > 0:
            reasons.append(f"Sharp money bonus: +{sharp_bonus:.3f} (delta={sharp_delta:.2f})")

        # Market movement
        line_bonus = min(line_move * 0.10, 0.10)
        if line_bonus > 0:
            reasons.append(f"Line movement bonus: +{line_bonus:.3f} (|Δ|={line_move:.2f})")

        # Tighter spreads in WNBA
        spread_bonus = max(0.0, 1.0 - (spread / 18.0)) * 0.06
        if spread_bonus > 0:
            reasons.append(f"Spread tightness bonus: +{spread_bonus:.3f} (spread={spread:.1f})")

        # Totals around 160-170
        total_bonus = 0.0
        if 158.0 <= total <= 172.0:
            total_bonus = 0.04
            reasons.append(f"Key total band bonus: +{total_bonus:.3f} (total={total:.1f})")

        # Estimate pre-analysis edge
        raw_edge = game.get("edge_estimate") or game.get("projected_edge") or game.get("ai_edge")
        if raw_edge is None:
            raw_edge = {
                "ULTRA": 0.25,
                "HIGH": 0.18,
                "MEDIUM": 0.10,
                "LOW": 0.04,
            }.get(edge_potential, 0.08)
        edge_component = float(raw_edge)

        base_confidence = float(game.get("confidence") or 0.55)
        base_confidence = max(0.35, min(base_confidence, 0.95))

        base_score = edge_component * base_confidence * team_weight
        reasons.append(f"Base edge score: {base_score:.3f}")

        priority = (
            base_score
            + edge_weight * 0.12
            + sharp_bonus
            + line_bonus
            + spread_bonus
            + total_bonus
        )

        return round(priority, 6), reasons


__all__ = ["GamePrioritizer"]
