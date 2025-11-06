#!/usr/bin/env python3
"""
College Football Game Prioritization
===================================

Provides heuristics for ordering college football games before running
expensive AI analysis. The prioritizer promotes high-leverage matchups
while still surfacing obscure edges in smaller conferences.
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
    """Heuristic prioritizer for college football game analysis."""

    def __init__(self) -> None:
        # Conference weighting derived from market liquidity and variance.
        self.conference_weights = {
            "sec": 0.90,
            "big ten": 0.95,
            "big 12": 0.80,
            "acc": 0.75,
            "pac-12": 0.70,
            "pac 12": 0.70,
            "american": 0.55,
            "sun belt": 0.45,
            "c-usa": 0.40,
            "conference usa": 0.40,
            "mountain west": 0.42,
            "mwc": 0.42,
            "mac": 0.38,
            "ivy league": 0.25,
            "fcs": 0.30,
        }

        self.edge_potential_weights = {
            "ULTRA": 0.85,
            "HIGH": 0.70,
            "MEDIUM": 0.45,
            "LOW": 0.20,
        }

        self.group_of_five_aliases = {
            "american",
            "sun belt",
            "c-usa",
            "conference usa",
            "mountain west",
            "mwc",
            "mac",
        }

        self.prioritization_log: List[PrioritizedGame] = []
        self.dynamic_weights: Dict[str, float] = {}
        self.weights_storage_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "football"
            / "ncaaf"
            / "conference_weights.json"
        )
        self._load_dynamic_weights()

    def _load_dynamic_weights(self) -> None:
        """Load persisted conference weighting overrides, if present."""
        try:
            if not self.weights_storage_path.is_file():
                return

            with self.weights_storage_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if not isinstance(payload, dict):
                logger.warning("Conference weights file is not a dictionary: %s", self.weights_storage_path)
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
                self.conference_weights.update(loaded)
                logger.info("Loaded %d dynamic conference weight overrides", len(loaded))
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning("Failed to load conference weights from %s: %s", self.weights_storage_path, exc)

    def save_conference_weights(self, output_path: Optional[Path] = None) -> None:
        """Persist dynamic conference weight overrides to disk."""
        path = Path(output_path) if output_path else self.weights_storage_path
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {key: round(weight, 4) for key, weight in sorted(self.dynamic_weights.items())}

        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(serializable, handle, indent=2, sort_keys=True)
            logger.info("Saved %d dynamic conference weights to %s", len(serializable), path)
        except Exception as exc:
            logger.error("Failed to persist conference weights to %s: %s", path, exc)

    def update_conference_weights(
        self,
        new_weights: Dict[str, float],
        *,
        persist: bool = True,
        output_path: Optional[Path] = None,
    ) -> None:
        """Update the active conference weights, optionally persisting overrides."""
        updated = False
        for conference, weight in new_weights.items():
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError):
                logger.debug("Skipping invalid weight for %s: %r", conference, weight)
                continue

            key = conference.strip().lower()
            if not key:
                continue

            clamped_weight = self._clamp_weight(numeric_weight)
            self.dynamic_weights[key] = clamped_weight
            self.conference_weights[key] = clamped_weight
            updated = True

        if updated and persist:
            self.save_conference_weights(output_path=output_path)

    def derive_weights_from_backtest(
        self,
        conference_breakdown: Dict[str, Dict[str, Any]],
        *,
        min_samples: int = 12,
    ) -> Dict[str, float]:
        """
        Generate conference weights from backtest performance breakdown.

        Args:
            conference_breakdown: Aggregated metrics keyed by conference.
            min_samples: Minimum number of bets required for a conference to qualify.

        Returns:
            Mapping of conference -> tuned weight (clamped 0.25-1.25).
        """
        tuned_weights: Dict[str, float] = {}

        for conference, metrics in conference_breakdown.items():
            key = conference.strip().lower()
            if not key or key == "unknown":
                continue

            sample_size = int(metrics.get("total_bets", 0) or 0)
            if sample_size < min_samples:
                logger.debug("Skipping %s: only %d samples (min=%d)", key, sample_size, min_samples)
                continue

            baseline = self.conference_weights.get(key, 0.55)
            roi = float(metrics.get("roi_percentage", 0.0) or 0.0)
            win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
            volume_score = min(sample_size / 75.0, 1.0)

            roi_score = max(min(roi, 40.0), -30.0) / 40.0  # [-0.75, 1.0]
            win_score = max(min(win_rate - 0.5, 0.25), -0.25) / 0.25  # [-1.0, 1.0]
            volume_adjustment = (volume_score - 0.5) * 0.1  # Bias toward larger samples

            adjustment = (0.22 * roi_score) + (0.14 * win_score) + volume_adjustment

            if roi < 0 and win_rate < 0.5:
                adjustment -= 0.05  # Penalize consistently poor performers

            tuned_weight = self._clamp_weight(baseline + adjustment)
            tuned_weights[key] = round(tuned_weight, 4)

        return tuned_weights

    def apply_backtest_breakdown(
        self,
        conference_breakdown: Dict[str, Dict[str, Any]],
        *,
        min_samples: int = 12,
        persist: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Apply tuned weights derived from a backtest breakdown.

        Returns the mapping of conferences that were updated.
        """
        tuned_weights = self.derive_weights_from_backtest(conference_breakdown, min_samples=min_samples)

        if not tuned_weights:
            logger.warning("No conference weights derived from backtest data")
            return {}

        self.update_conference_weights(tuned_weights, persist=persist, output_path=output_path)
        return tuned_weights

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
                "conference": item.game.get("conference"),
            }
            for item in self.prioritization_log
        ]

    def _score_game(self, game: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Compute a heuristic priority score for a game."""
        reasons: List[str] = []

        conference = (game.get("conference") or "").lower()
        edge_potential = (game.get("edge_potential") or "MEDIUM").upper()
        spread = abs(float(game.get("spread") or 0.0))
        total = float(game.get("total") or 0.0)
        public_pct = float(game.get("public_percentage") or 0.5)
        sharp_pct = float(game.get("sharp_percentage") or 0.5)
        line_move = abs(float(game.get("line_movement") or 0.0))

        # Base edge proxy using edge potential tag.
        edge_weight = self.edge_potential_weights.get(edge_potential, 0.35)
        reasons.append(f"Edge potential weight: {edge_weight:.3f} ({edge_potential})")

        # Conference weighting balances liquidity vs. stagnation.
        conf_weight = self.conference_weights.get(conference, 0.55)
        if conference in self.group_of_five_aliases:
            reasons.append("Group of Five bonus applied")
        elif "fcs" in conference or game.get("game_type", "").lower().startswith("fcs"):
            reasons.append("FCS volatility bonus applied")

        # Market signal: sharp/public discrepancy.
        sharp_delta = max(sharp_pct - public_pct, 0.0)
        sharp_bonus = sharp_delta * 0.4  # amplify sharp action edges.
        if sharp_bonus > 0:
            reasons.append(f"Sharp money bonus: +{sharp_bonus:.3f} (delta={sharp_delta:.2f})")

        # Market movement: early strong line moves get attention.
        line_bonus = min(line_move * 0.08, 0.08)
        if line_bonus > 0:
            reasons.append(f"Line movement bonus: +{line_bonus:.3f} (|Δ|={line_move:.2f})")

        # Tighter spreads -> higher volatility -> more opportunity.
        spread_bonus = max(0.0, 1.0 - (spread / 28.0)) * 0.05
        if spread_bonus > 0:
            reasons.append(f"Spread tightness bonus: +{spread_bonus:.3f} (spread={spread:.1f})")

        # Totals around key numbers (45-55) create additional upside.
        total_bonus = 0.0
        if 42.0 <= total <= 55.0:
            total_bonus = 0.03
            reasons.append(f"Key total band bonus: +{total_bonus:.3f} (total={total:.1f})")

        # Obscure conferences get a baseline look for hidden value.
        obscure_bonus = 0.0
        if conference in self.group_of_five_aliases or "fcs" in conference:
            obscure_bonus = 0.02
            reasons.append(f"Obscure conference bonus: +{obscure_bonus:.3f}")

        # Estimate pre-analysis edge if present in payload.
        raw_edge = game.get("edge_estimate") or game.get("projected_edge") or game.get("ai_edge")
        if raw_edge is None:
            # Fall back to edge potential mapping for initial priority.
            raw_edge = {
                "ULTRA": 0.25,
                "HIGH": 0.18,
                "MEDIUM": 0.10,
                "LOW": 0.04,
            }.get(edge_potential, 0.08)
        edge_component = float(raw_edge)

        # Confidence heuristics.
        base_confidence = float(game.get("confidence") or 0.55)
        base_confidence = max(0.35, min(base_confidence, 0.95))

        base_score = edge_component * base_confidence * conf_weight
        reasons.append(f"Base edge score: {base_score:.3f} (edge={edge_component:.3f} * conf={base_confidence:.2f} * weight={conf_weight:.2f})")

        priority = (
            base_score
            + edge_weight * 0.12
            + sharp_bonus
            + line_bonus
            + spread_bonus
            + total_bonus
            + obscure_bonus
        )

        # Slight boost for Group of Five matchups with higher perceived edge.
        if conference in self.group_of_five_aliases and edge_component >= 0.12:
            priority += 0.015
            reasons.append("Hidden gem boost: +0.015 (mid-major with promising edge)")

        return round(priority, 6), reasons


__all__ = ["GamePrioritizer"]
