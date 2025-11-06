"""
Automated generation of team-specific referee reports for 2018-2024.

Each report surfaces:
  - Crew rotation timelines by season.
  - Crew style impacts on the team (penalties, margins, odds deltas).
  - Narrative beats when officiating alignments coincided with dramatic swings.
  - Broadcast context comparisons (primetime vs low-visibility windows).

The output is written to `reports/referee_conspiracy/{team}.md`.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from referee_style_analysis import (
    build_game_level_table,
    build_team_penalty_table,
    compute_crew_features,
    label_crews,
    TEAM_ALIAS,
)

OUTPUT_DIR = Path("reports/referee_conspiracy")
MIN_GAMES_FOR_CREW_IMPACT = 3
NARRATIVE_JSON_PATH = Path("data/referee_conspiracy/narrative_notes.json")
NARRATIVE_VALIDATION_PATH = Path("data/referee_conspiracy/narrative_validation.json")


def _load_team_metadata() -> pd.DataFrame:
    try:
        from nfl_data_py import import_team_desc
    except Exception:
        return pd.DataFrame()
    return import_team_desc()[["team_abbr", "team_name", "team_nick"]]


def _build_team_games() -> pd.DataFrame:
    games = build_game_level_table()
    team_penalties = build_team_penalty_table()

    home = games[
        [
            "season",
            "week",
            "game_id",
            "gameday",
            "weekday",
            "gametime_hours",
            "is_primetime",
            "is_low_visibility",
            "is_overseas",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "points_total",
            "margin_abs",
            "overtime",
            "penalties",
            "penalty_yards",
            "flag_density",
            "score_swing_mean_abs",
            "spread_line",
            "referee",
            "U",
            "DJ",
            "LJ",
            "BJ",
            "SJ",
            "FJ",
        ]
    ].copy()

    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["team_points"] = home["home_score"]
    home["opp_points"] = home["away_score"]
    home["team_role"] = "home"

    away = home.copy()
    away["team"] = home["away_team"]
    away["opponent"] = home["home_team"]
    away["team_points"] = home["away_score"]
    away["opp_points"] = home["home_score"]
    away["team_role"] = "away"

    cols_to_drop = ["home_team", "away_team", "home_score", "away_score"]
    home.drop(columns=cols_to_drop, inplace=True)
    away.drop(columns=cols_to_drop, inplace=True)

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["team"] = team_games["team"].replace(TEAM_ALIAS)
    team_games["opponent"] = team_games["opponent"].replace(TEAM_ALIAS)

    team_games = team_games.merge(
        team_penalties[
            [
                "season",
                "game_id",
                "team",
                "team_penalties",
                "team_penalty_yards",
                "team_score_swing_abs",
            ]
        ],
        on=["season", "game_id", "team"],
        how="left",
    )

    opponent_penalties = team_penalties.rename(
        columns={
            "team": "opponent",
            "team_penalties": "opponent_penalties",
            "team_penalty_yards": "opponent_penalty_yards",
            "team_score_swing_abs": "opponent_score_swing_abs",
        }
    )
    team_games = team_games.merge(
        opponent_penalties[["season", "game_id", "opponent", "opponent_penalties"]],
        on=["season", "game_id", "opponent"],
        how="left",
    )

    team_games["penalties_on_team"] = team_games["team_penalties"].fillna(0).astype(int)
    team_games["penalties_on_opponent"] = (
        team_games["opponent_penalties"].fillna(0).astype(int)
    )
    team_games["penalty_diff"] = (
        team_games["penalties_on_team"] - team_games["penalties_on_opponent"]
    )

    team_games["actual_margin"] = team_games["team_points"] - team_games["opp_points"]
    team_games["expected_margin"] = np.where(
        team_games["team_role"] == "away", -team_games["spread_line"], team_games["spread_line"]
    )
    team_games["odds_delta"] = team_games["actual_margin"] - team_games["expected_margin"]
    team_games["odds_shock"] = team_games["odds_delta"].abs()

    team_games["crew_fingerprint"] = team_games[
        ["referee", "U", "DJ", "LJ", "BJ", "SJ", "FJ"]
    ].astype(str).agg(" | ".join, axis=1)

    team_games.sort_values(["gameday", "gametime_hours"], inplace=True)
    team_games.reset_index(drop=True, inplace=True)
    return team_games


def _crew_labels_map() -> Dict[str, List[str]]:
    labels = label_crews()
    mapping: Dict[str, List[str]] = {}
    for row in labels.itertuples(index=False):
        mapping.setdefault(row.referee, []).append(row.label)
    return mapping


def _crew_feature_map() -> Dict[str, Dict[str, float]]:
    features = compute_crew_features()
    return {
        row.referee: row._asdict()
        for row in features.itertuples(index=False)
    }


def _team_name_lookup() -> Dict[str, str]:
    meta = _load_team_metadata()
    if meta.empty:
        return {}
    return dict(zip(meta["team_abbr"], meta["team_name"]))


def _format_labels(labels: List[str]) -> str:
    if not labels:
        return "baseline_control"
    return ", ".join(sorted(labels))


def _find_extreme_games(team_df: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
    insights: Dict[str, List[Dict[str, str]]] = {
        "script_resets": [],
        "penalty_spikes": [],
        "overtime_crashes": [],
    }
    work = team_df.copy()
    work["prev_margin_avg"] = (
        work.groupby("season")["actual_margin"].rolling(window=3, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    )
    work["margin_delta_vs_prev"] = work["actual_margin"] - work["prev_margin_avg"]

    for row in work.itertuples():
        if math.isnan(row.prev_margin_avg):
            continue
        new_face = (
            row.referee not in work.loc[: row.Index - 1, "referee"].tail(3).values
        )
        if new_face and abs(row.margin_delta_vs_prev) >= 10:
            direction = "surge" if row.margin_delta_vs_prev > 0 else "collapse"
            insights["script_resets"].append(
                {
                    "season": row.season,
                    "week": row.week,
                    "opponent": row.opponent,
                    "result": row.actual_margin,
                    "referee": row.referee,
                    "direction": direction,
                    "delta": row.margin_delta_vs_prev,
                }
            )

        if row.penalties_on_team >= team_df["penalties_on_team"].quantile(0.9):
            insights["penalty_spikes"].append(
                {
                    "season": row.season,
                    "week": row.week,
                    "opponent": row.opponent,
                    "penalties": row.penalties_on_team,
                    "referee": row.referee,
                }
            )

        if row.overtime == 1 and row.actual_margin < 0:
            insights["overtime_crashes"].append(
                {
                    "season": row.season,
                    "week": row.week,
                    "opponent": row.opponent,
                    "referee": row.referee,
                    "margin": row.actual_margin,
                }
            )

    return insights


def _season_timeline(team_df: pd.DataFrame, labels_map: Dict[str, List[str]]) -> List[str]:
    lines: List[str] = []
    grouped = (
        team_df.groupby(["season", "referee"])
        .agg(
            weeks=("week", lambda x: ",".join(str(int(w)) for w in sorted(x.unique()))),
            games=("game_id", "count"),
            avg_margin=("actual_margin", "mean"),
        )
        .reset_index()
        .sort_values(["season", "referee"])
    )
    for row in grouped.itertuples():
        label_str = _format_labels(labels_map.get(row.referee, []))
        lines.append(
            f"- {int(row.season)}: {row.referee} ({row.games} games; weeks {row.weeks}; avg margin {row.avg_margin:+.1f}; labels: {label_str})"
        )
    return lines


def _crew_impact(team_df: pd.DataFrame, feature_map: Dict[str, Dict[str, float]], labels_map: Dict[str, List[str]]) -> List[str]:
    lines: List[str] = []
    crew_stats = (
        team_df.groupby("referee")
        .agg(
            games=("game_id", "count"),
            team_margin_avg=("actual_margin", "mean"),
            penalties_on_team_avg=("penalties_on_team", "mean"),
            penalty_diff_avg=("penalty_diff", "mean"),
            odds_delta_avg=("odds_delta", "mean"),
            points_total_avg=("points_total", "mean"),
        )
        .reset_index()
        .sort_values("games", ascending=False)
    )

    for row in crew_stats.itertuples():
        if row.games < MIN_GAMES_FOR_CREW_IMPACT:
            continue
        labels = _format_labels(labels_map.get(row.referee, []))
        feature_row = feature_map.get(row.referee, {})
        overtime_rate = feature_row.get("overtime_rate", 0)
        lines.append(
            f"- {row.referee} ({labels}): {row.games} games, avg margin {row.team_margin_avg:+.1f}, penalties on team {row.penalties_on_team_avg:.1f}, penalty diff {row.penalty_diff_avg:+.1f}, odds delta {row.odds_delta_avg:+.1f}, overtime rate {overtime_rate:.2%}"
        )
    if not lines:
        lines.append("- Sample size too small for sustained crew impact; most assignments were one-offs.")
    return lines


def _broadcast_section(team_df: pd.DataFrame) -> List[str]:
    primetime = team_df.loc[team_df["is_primetime"]]
    low_vis = team_df.loc[team_df["is_low_visibility"]]
    other = team_df.loc[~team_df["is_primetime"]]

    def _fmt(df: pd.DataFrame) -> Tuple[float, float]:
        if df.empty:
            return (0.0, 0.0)
        return (
            df["penalties_on_team"].mean(),
            df["points_total"].mean(),
        )

    primetime_pen, primetime_pts = _fmt(primetime)
    low_pen, low_pts = _fmt(low_vis)
    other_pen, other_pts = _fmt(other)

    return [
        f"- Primetime: {primetime_pen:.1f} penalties on team, {primetime_pts:.1f} total points",
        f"- Low-visibility windows: {low_pen:.1f} penalties on team, {low_pts:.1f} total points",
        f"- Other slots: {other_pen:.1f} penalties on team, {other_pts:.1f} total points",
    ]


def _narrative_section(insights: Dict[str, List[Dict[str, str]]]) -> List[str]:
    lines: List[str] = []
    for event in insights["script_resets"][:3]:
        lines.append(
            f"- Week {int(event['week'])} {int(event['season'])} vs {event['opponent']}: {event['direction']} of {event['delta']:+.1f} points with {event['referee']} stepping in (final margin {event['result']:+.1f})."
        )
    for event in insights["penalty_spikes"][:2]:
        lines.append(
            f"- Week {int(event['week'])} {int(event['season'])} vs {event['opponent']}: {event['penalties']} flags dropped (crew lead {event['referee']})."
        )
    for event in insights["overtime_crashes"][:2]:
        lines.append(
            f"- Week {int(event['week'])} {int(event['season'])} vs {event['opponent']}: overtime gut punch {event['margin']:+.1f} under {event['referee']}."
        )
    if not lines:
        lines.append("- No dramatic referee-linked swings detected beyond baseline variance.")
    return lines


def _load_narrative_notes() -> Dict[str, Dict[str, any]]:
    if not NARRATIVE_JSON_PATH.exists():
        return {}
    data = json.loads(NARRATIVE_JSON_PATH.read_text())
    # Normalize keys for quick lookup
    normalized = {}
    for team, payload in data.items():
        normalized[team.replace(" ", "").upper()] = payload
    return normalized


def _classify_bet_edge_tag(bet_edge: str) -> Optional[str]:
    lower = bet_edge.lower()
    if "over" in lower:
        return "over"
    if "under" in lower:
        return "under"
    if "ot" in lower:
        return "ot"
    if "ml" in lower or "moneyline" in lower:
        return "ml"
    return None


def _calculate_bet_edge_hit_rate(team: str, bet_edge: str, team_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    tag = _classify_bet_edge_tag(bet_edge)
    if not tag:
        return None

    df = team_df.copy()
    df = df[df["total_line"].notna()] if tag in {"over", "under"} else df
    if df.empty:
        return None

    if tag == "over":
        actual = df["points_total"] > df["total_line"]
    elif tag == "under":
        actual = df["points_total"] < df["total_line"]
    elif tag == "ot":
        actual = df["overtime"] == 1
    else:  # ml
        actual = df["actual_margin"] > 0

    if actual.empty:
        return None

    y_true = actual.astype(bool).tolist()
    y_pred = [True] * len(y_true)
    hit_rate = accuracy_score(y_true, y_pred)
    return {
        "team": team,
        "bet_edge": bet_edge,
        "tag": tag,
        "hit_rate": float(hit_rate),
        "sample_size": int(len(y_true)),
    }


def generate_reports() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for existing in OUTPUT_DIR.glob("*.md"):
        existing.unlink()

    team_games = _build_team_games()
    labels_map = _crew_labels_map()
    feature_map = _crew_feature_map()
    team_names = _team_name_lookup()
    narrative_notes = _load_narrative_notes()
    validation_records: Dict[str, Dict[str, float]] = {}

    for team, team_df in team_games.groupby("team"):
        team_df = team_df.sort_values(["gameday", "gametime_hours"]).reset_index(drop=True)
        insights = _find_extreme_games(team_df)
        timeline_lines = _season_timeline(team_df, labels_map)
        impact_lines = _crew_impact(team_df, feature_map, labels_map)
        broadcast_lines = _broadcast_section(team_df)
        narrative_lines = _narrative_section(insights)

        crew_examples = team_df.groupby("referee").head(1)[
            ["referee", "crew_fingerprint"]
        ]
        crew_listing = "\n".join(
            f"- {row.referee}: {row.crew_fingerprint}" for row in crew_examples.itertuples(index=False)
        )

        team_title = team_names.get(team, team)
        header = f"# {team_title} Referee Autopsy (2018-2024)\n"
        body = "\n".join(
            [
                "## Crew Rotation Timeline",
                "\n".join(timeline_lines) or "No repeated crews detected.",
                "\n## Style Impact",
                "\n".join(impact_lines),
                "\n## Narrative Correlations",
                "\n".join(narrative_lines),
                "\n## Broadcast Behavior",
                "\n".join(broadcast_lines),
                "\n## Crew Rosters Seen",
                crew_listing or "- Crew data unavailable.",
            ]
        )

        output_path = OUTPUT_DIR / f"{team}.md"
        note_entry = narrative_notes.get(team)
        if note_entry:
            body += "\n\n## Narrative Edge\n"
            body += f"- Script weight: {note_entry.get('script_weight', 'N/A')}\n"
            if note_entry.get("bet_edge"):
                body += f"- Bet edge: {note_entry['bet_edge']}\n"
                hit_info = _calculate_bet_edge_hit_rate(team, note_entry["bet_edge"], team_df)
                if hit_info:
                    validation_records[team] = hit_info
                    body += f"- Historical hit rate: {hit_info['hit_rate']:.2%} over {hit_info['sample_size']} games\n"
        output_path.write_text(f"{header}\n{body}\n", encoding="utf-8")

    if validation_records:
        NARRATIVE_VALIDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        NARRATIVE_VALIDATION_PATH.write_text(json.dumps(validation_records, indent=2))


if __name__ == "__main__":
    generate_reports()
