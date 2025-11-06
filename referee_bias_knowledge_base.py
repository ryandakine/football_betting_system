"""
Referee bias knowledge base.

Loads manually curated weekly referee bias snapshots (anti-home tendencies, public splits,
recommended plays) and exposes small helper utilities so downstream systems can blend the
context into prompts, meta-features, or probability heuristics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BiasRiskProfile:
    """Numeric heuristics for translating qualitative risk labels."""

    probability_shift: float
    confidence_penalty: float
    script_weight: float
    boring_score: float


RISK_LOOKUP: Dict[str, BiasRiskProfile] = {
    "high": BiasRiskProfile(probability_shift=-0.04, confidence_penalty=0.07, script_weight=0.22, boring_score=0.92),
    "mediumhigh": BiasRiskProfile(probability_shift=-0.03, confidence_penalty=0.05, script_weight=0.18, boring_score=0.78),
    "medium": BiasRiskProfile(probability_shift=-0.02, confidence_penalty=0.04, script_weight=0.14, boring_score=0.63),
    "mediumlow": BiasRiskProfile(probability_shift=-0.015, confidence_penalty=0.03, script_weight=0.1, boring_score=0.52),
    "lowmedium": BiasRiskProfile(probability_shift=-0.01, confidence_penalty=0.02, script_weight=0.08, boring_score=0.4),
    "low": BiasRiskProfile(probability_shift=-0.005, confidence_penalty=0.01, script_weight=0.04, boring_score=0.25),
}


class RefereeBiasKnowledgeBase:
    """Loads curated week-level referee bias snapshots and exposes lookup helpers."""

    DEFAULT_PATH = Path("data/referee_conspiracy/referee_bias_snapshots.json")

    def __init__(self, data_path: Optional[Path | str] = None) -> None:
        self.data_path = Path(data_path) if data_path else self.DEFAULT_PATH
        self._loaded = False
        self._raw: Dict[str, Any] = {}
        self._game_index: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Loading / indexing
    # ------------------------------------------------------------------

    def reload(self) -> None:
        self._loaded = False
        self._raw = {}
        self._game_index = {}
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.data_path.exists():
            logger.debug("Referee bias snapshot file missing: %s", self.data_path)
            self._raw = {}
            self._game_index = {}
            self._loaded = True
            return
        try:
            payload = json.loads(self.data_path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse referee bias snapshot JSON %s: %s", self.data_path, exc)
            self._raw = {}
            self._game_index = {}
            self._loaded = True
            return

        self._raw = payload if isinstance(payload, dict) else {}
        self._build_game_index()
        self._loaded = True

    def _build_game_index(self) -> None:
        self._game_index = {}
        season = self._safe_int(self._raw.get("season"))
        if not season:
            return
        weeks = self._raw.get("weeks") or {}
        for week_key, week_payload in weeks.items():
            week = self._safe_int(week_key)
            if not week:
                continue
            games = week_payload.get("games") or []
            for game in games:
                if not isinstance(game, dict):
                    continue
                record = game.copy()
                record["season"] = season
                record["week"] = week
                home_aliases = self._normalize_aliases(record.get("home_aliases"), record.get("home_team"))
                away_aliases = self._normalize_aliases(record.get("away_aliases"), record.get("away_team"))
                if not home_aliases or not away_aliases:
                    continue
                indexed_entry = {
                    "record": record,
                    "home_aliases": home_aliases,
                    "away_aliases": away_aliases,
                }
                self._game_index.setdefault((season, week), []).append(indexed_entry)

    @staticmethod
    def _normalize_aliases(aliases: Optional[Iterable[str]], fallback: Optional[str]) -> List[str]:
        alias_list: List[str] = []
        if aliases:
            alias_list.extend(str(alias) for alias in aliases if alias)
        if fallback:
            alias_list.append(str(fallback))
        normalized = {RefereeBiasKnowledgeBase._normalize(label) for label in alias_list if label}
        return sorted({entry for entry in normalized if entry})

    @staticmethod
    def _normalize(label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        return "".join(ch for ch in str(label).lower() if ch.isalnum())

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Public lookups
    # ------------------------------------------------------------------

    def get_week_snapshot(self, season: int, week: int) -> Dict[str, Any]:
        self._ensure_loaded()
        weeks = self._raw.get("weeks") or {}
        return weeks.get(str(week), {})

    def lookup_game(
        self,
        season: Optional[int],
        week: Optional[int],
        home_label: Optional[str],
        away_label: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        if not self._game_index:
            return None

        season_val = season or self._safe_int(self._raw.get("season"))
        week_val = week
        if week_val is None:
            return None
        try:
            week_val = int(week_val)
        except (TypeError, ValueError):
            return None
        if season_val is None:
            return None

        entries = self._game_index.get((int(season_val), int(week_val)))
        if not entries:
            return None

        home_norm = self._normalize(home_label)
        away_norm = self._normalize(away_label)
        if not home_norm or not away_norm:
            return None

        for entry in entries:
            if home_norm in entry["home_aliases"] and away_norm in entry["away_aliases"]:
                return entry["record"]
        return None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def evaluate_game(
        self,
        game_data: Dict[str, Any],
        base_probability: float,
        current_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Blend bias metadata with the current game context.

        Returns None when the matchup is not present in the knowledge base.
        """
        self._ensure_loaded()
        if not self._game_index:
            return None

        season = self._safe_int(game_data.get("season")) or self._safe_int(self._raw.get("season"))
        week = (
            self._safe_int(game_data.get("week"))
            or self._safe_int(game_data.get("game_week"))
            or self._safe_int(game_data.get("match_week"))
        )
        home = game_data.get("home_team")
        away = game_data.get("away_team")

        record = self.lookup_game(season, week, home, away)
        if not record:
            return None

        risk_label_raw = str(record.get("boring_risk", "medium")).lower()
        normalized_key = risk_label_raw.replace("-", "").replace(" ", "")
        risk_profile = RISK_LOOKUP.get(normalized_key, RISK_LOOKUP["medium"])

        probability = self._clamp_probability(base_probability + risk_profile.probability_shift)
        confidence = self._clamp_probability(current_confidence - risk_profile.confidence_penalty)

        spread_info = record.get("spread") or {}
        home_spread = self._safe_float(spread_info.get("home"))
        away_spread = self._safe_float(spread_info.get("away"))

        trigger_auto_under = bool(
            record.get("anti_home")
            and (home_spread or 0.0) >= 3.0
            and (record.get("public_away_pct") or 0.0) >= 65
        )

        alert_parts: List[str] = [
            f"Ref bias ({record.get('referee')}): {risk_label_raw} boring-risk window."
        ]
        if record.get("anti_home"):
            alert_parts.append("Anti-home: YES.")
        if record.get("public_away_pct") is not None:
            alert_parts.append(f"Public on road: {record['public_away_pct']}%.")
        if trigger_auto_under:
            alert_parts.append("Trigger: auto-under signal satisfied.")
        elif record.get("script_triggers"):
            alert_parts.append(f"Triggers: {', '.join(record.get('script_triggers', []))}.")
        if record.get("recommended_plays"):
            alert_parts.append(f"Plays: {', '.join(record['recommended_plays'])}.")

        return {
            "season": record.get("season"),
            "week": record.get("week"),
            "referee": record.get("referee"),
            "anti_home": bool(record.get("anti_home")),
            "boring_risk": risk_label_raw,
            "boring_score": risk_profile.boring_score,
            "public_away_pct": self._safe_float(record.get("public_away_pct")),
            "home_flag_pct": self._safe_float(record.get("home_flag_pct")),
            "home_ats_pct": self._safe_float(record.get("home_ats_pct")),
            "home_spread": home_spread,
            "away_spread": away_spread,
            "total": self._safe_float(record.get("total")),
            "probability_shift": risk_profile.probability_shift,
            "probability": probability,
            "confidence_penalty": risk_profile.confidence_penalty,
            "confidence": confidence,
            "script_weight": risk_profile.script_weight,
            "tags": list(record.get("tags") or []),
            "alert": " ".join(alert_parts),
            "recommended_plays": list(record.get("recommended_plays") or []),
            "notes": record.get("notes"),
            "trigger_auto_under": trigger_auto_under,
            "script_triggers": list(record.get("script_triggers") or []),
            "record": record,
        }

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clamp_probability(value: float) -> float:
        return max(0.0, min(1.0, float(value)))


__all__ = ["RefereeBiasKnowledgeBase"]
