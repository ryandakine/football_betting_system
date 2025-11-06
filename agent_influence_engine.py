#!/usr/bin/env python3
"""
Agent Influence Engine
======================

Reads optional agent/relationship datasets and produces adjustment signals
used to tweak model confidence and edge based on potential conflicts of
interest (coach–player–agent ties, coordinator overlap, ref linkages,
ownership broadcast influences, etc.).

No synthetic data is created. If datasets are missing, returns neutral
adjustments (no-ops).

Expected optional CSV files (place under data/agents/):
- coach_agents.csv          : team,coach_name,agent_name
- player_agents.csv         : team,player_name,position,is_starter,agent_name
- coordinator_agents.csv    : team,role,coach_name,agent_name  (role in {OC,DC})
- ref_agent_conflicts.csv   : year,week,crew_chief,agent_name,team  (if known)
- ownership_broadcast.csv   : team,owner_name,is_broadcaster

Returned adjustments (common):
- edge_multiplier (float): multiplies current computed edge (default 1.0)
- confidence_delta (float): additive change to confidence in [−1, 1] (default 0.0)
- rules_triggered (list[str])
- strategy_signals (list[str])

College (NCAA FBS) optional datasets (place under data/agents/college/):
- coach_agents.csv          : team,coach_name,agent_name,agency_name
- player_agents.csv         : team,player_name,position,is_starter,agent_name,agency_name
- coordinator_agents.csv    : team,role,coach_name,agent_name,agency_name
- ref_agent_conflicts.csv   : year,week,crew_chief,agent_name,team
- nil_agencies.csv          : agency_name,notes,is_flagged

Additional college-only outputs:
- penalty_bias_home (float): multiplier for expected offensive penalties against visitor (default 1.0)
- phantom_flag_probability (float): Monte Carlo estimate of phantom flag incidence (0..1)
- scandal_score (float): heuristic score 0..1 of agent-related conflict
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AgentInfluenceResult:
    edge_multiplier: float = 1.0
    confidence_delta: float = 0.0
    rules_triggered: List[str] = None
    strategy_signals: List[str] = None

    def to_dict(self) -> Dict:
        return {
            "edge_multiplier": self.edge_multiplier,
            "confidence_delta": self.confidence_delta,
            "rules_triggered": list(self.rules_triggered or []),
            "strategy_signals": list(self.strategy_signals or []),
        }


class AgentInfluenceEngine:
    def __init__(self, base_dir: str = "data/agents"):
        self.base = Path(base_dir)
        self._coach_agents = self._read_csv("coach_agents.csv")
        self._player_agents = self._read_csv("player_agents.csv")
        self._coord_agents = self._read_csv("coordinator_agents.csv")
        self._ref_conflicts = self._read_csv("ref_agent_conflicts.csv")
        self._owner_broadcast = self._read_csv("ownership_broadcast.csv")
        # College
        self.base_cfb = self.base / "college"
        self._coach_agents_cfb = self._read_csv("college/coach_agents.csv")
        self._player_agents_cfb = self._read_csv("college/player_agents.csv")
        self._coord_agents_cfb = self._read_csv("college/coordinator_agents.csv")
        self._ref_conflicts_cfb = self._read_csv("college/ref_agent_conflicts.csv")
        self._nil_agencies_cfb = self._read_csv("college/nil_agencies.csv")

    def _read_csv(self, name: str) -> List[Dict]:
        path = self.base / name
        if not path.exists():
            return []
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [dict({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}) for row in reader]
                # Basic validation: ensure required headers exist if any rows present
                if rows and any(v is None for v in rows[0].keys()):
                    return []
                return rows
        except Exception:
            return []

    def compute_adjustments(self, game: Dict) -> Dict:
        """Compute agent-based adjustments for a single game.

        game expects keys: home_team, away_team, spread (home spread, negative favorite)
        """
        res = AgentInfluenceResult(edge_multiplier=1.0, confidence_delta=0.0, rules_triggered=[], strategy_signals=[])

        home = (game.get("home_team") or "").strip()
        away = (game.get("away_team") or "").strip()
        spread = game.get("spread")
        is_home_underdog = (spread is not None) and (spread > 0)

        # Helper lookups
        def team_players(team: str) -> List[Dict]:
            return [r for r in self._player_agents if r.get("team", "").strip().lower() == team.lower()]

        def team_coach_agent(team: str) -> Optional[str]:
            for r in self._coach_agents:
                if r.get("team", "").strip().lower() == team.lower():
                    return r.get("agent_name")
            return None

        def team_coord_agent(team: str, role: str) -> Optional[str]:
            for r in self._coord_agents:
                if r.get("team", "").strip().lower() == team.lower() and r.get("role", "").upper() == role.upper():
                    return r.get("agent_name")
            return None

        # Rule 1: Coach & QB share agent → home underdog boost
        try:
            coach_agent = team_coach_agent(home)
            qb_agent = None
            for p in team_players(home):
                if (p.get("position", "").upper() == "QB") and (p.get("is_starter", "").lower() in {"1", "true", "yes"}):
                    qb_agent = p.get("agent_name")
                    break
            if coach_agent and qb_agent and coach_agent == qb_agent and is_home_underdog:
                # Edge bump ~18%
                res.edge_multiplier *= 1.18
                res.confidence_delta += 0.03
                res.rules_triggered.append("coach_qb_shared_agent_home_dog_boost")
        except Exception:
            pass

        # Rule 2: Same agent across OL starters and DC → defensive nerf
        try:
            dc_agent = team_coord_agent(home, "DC")
            if dc_agent:
                ol_agents = {p.get("agent_name") for p in team_players(home) if p.get("position", "").upper() in {"LT", "LG", "C", "RG", "RT", "OL"}}
                if dc_agent in ol_agents:
                    res.edge_multiplier *= 0.75
                    res.confidence_delta -= 0.04
                    res.rules_triggered.append("ol_dc_same_agent_defense_nerf")
        except Exception:
            pass

        # Rule 3: High inter-team agent overlap → volatility (favor Unders type signal)
        try:
            home_agents = {p.get("agent_name") for p in team_players(home)}
            away_agents = {p.get("agent_name") for p in team_players(away)}
            if home_agents and away_agents:
                overlap = len(home_agents.intersection(away_agents))
                union = len(home_agents.union(away_agents)) or 1
                overlap_ratio = overlap / union
                if overlap_ratio >= 0.15:  # 15%+ overlap threshold
                    res.confidence_delta -= 0.02
                    res.strategy_signals.append("volatility_overlap_under_completions")
                    res.rules_triggered.append("inter_team_agent_overlap")
        except Exception:
            pass

        # Rule 4: Ref conflict (team appears with crew chief agent record) → more flags against offense
        try:
            # Expected fields: year,week,crew_chief,agent_name,team
            home_conflict = any(r.get("team", "").strip().lower() == home.lower() for r in self._ref_conflicts)
            if home_conflict:
                res.edge_multiplier *= 0.85
                res.rules_triggered.append("ref_conflict_home_penalty_bias")
        except Exception:
            pass

        # Rule 5: Ownership broadcast influence (e.g., minority owner broadcaster) → slight home bias
        try:
            flagged = any((r.get("team", "").strip().lower() == home.lower()) and (r.get("is_broadcaster", "").lower() in {"1", "true", "yes"}) for r in self._owner_broadcast)
            if flagged:
                res.edge_multiplier *= 1.05
                res.rules_triggered.append("ownership_broadcast_bias")
        except Exception:
            pass

        # Clamp confidence delta
        res.confidence_delta = max(-0.2, min(0.2, res.confidence_delta))

        return res.to_dict()

    # College-only extended adjustments with Monte Carlo scandal simulation
    def compute_adjustments_cfb(self, game: Dict, simulations: int = 2000) -> Dict:
        res = AgentInfluenceResult(edge_multiplier=1.0, confidence_delta=0.0, rules_triggered=[], strategy_signals=[])
        home = (game.get("home_team") or "").strip()
        away = (game.get("away_team") or "").strip()
        spread = game.get("spread")
        is_home_underdog = (spread is not None) and (spread > 0)

        def team_players_cfb(team: str) -> List[Dict]:
            return [r for r in self._player_agents_cfb if r.get("team", "").strip().lower() == team.lower()]

        def team_coach_agent_cfb(team: str) -> Optional[Dict]:
            for r in self._coach_agents_cfb:
                if r.get("team", "").strip().lower() == team.lower():
                    return r
            return None

        # Coach-starter shared agent/agency → home dog boost
        try:
            coach_row = team_coach_agent_cfb(home)
            coach_agent = coach_row.get("agent_name") if coach_row else None
            coach_agency = coach_row.get("agency_name") if coach_row else None
            shared = False
            for p in team_players_cfb(home):
                if p.get("is_starter", "").lower() in {"1", "true", "yes"}:
                    if coach_agent and p.get("agent_name") == coach_agent:
                        shared = True
                        break
                    if coach_agency and p.get("agency_name") == coach_agency:
                        shared = True
                        break
            if shared and is_home_underdog:
                res.edge_multiplier *= 1.18
                res.confidence_delta += 0.03
                res.rules_triggered.append("cfb_coach_starter_shared_agent_home_dog_boost")
        except Exception:
            pass

        # NIL agency flagged → potential penalty bias against visitor (home protected)
        penalty_bias_home = 1.0
        try:
            if coach_row and any((a.get("agency_name") == coach_row.get("agency_name")) and (a.get("is_flagged", "").lower() in {"1","true","yes"}) for a in self._nil_agencies_cfb):
                penalty_bias_home *= 1.12  # ~12% more calls favoring home
                res.rules_triggered.append("cfb_flagged_nil_agency_penalty_bias")
        except Exception:
            pass

        # Monte Carlo: phantom flag probability from overlaps
        import random
        home_agents = {p.get("agent_name") or p.get("agency_name") for p in team_players_cfb(home)}
        away_agents = {p.get("agent_name") or p.get("agency_name") for p in team_players_cfb(away)}
        overlap = len((home_agents - {None}).intersection(away_agents - {None}))
        base_p = 0.02 + 0.05 * min(1.0, overlap / 5.0)
        if coach_row and coach_row.get("agency_name"):
            base_p += 0.03
        if is_home_underdog:
            base_p += 0.01
        base_p = max(0.0, min(0.4, base_p))
        phantom_count = 0
        for _ in range(simulations):
            if random.random() < base_p:
                phantom_count += 1
        phantom_prob = phantom_count / simulations

        scandal_score = min(1.0, base_p * 2.5)
        if scandal_score >= 0.5:
            res.strategy_signals.append("cfb_scandal_watch_under_flags")

        d = res.to_dict()
        d.update({
            "penalty_bias_home": penalty_bias_home,
            "phantom_flag_probability": round(phantom_prob, 3),
            "scandal_score": round(scandal_score, 3),
        })
        return d


