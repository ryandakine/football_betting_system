import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/referee_conspiracy")
CREW_PATH = DATA_DIR / "crew_game_log.parquet"
PENALTY_PATH = DATA_DIR / "penalties_2018_2024.parquet"
OUTPUT_PATH = DATA_DIR / "sniper_predictions.json"


def safe_read(path, columns=None):
    if not path.exists():
        return pd.DataFrame(columns=columns or [])
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=columns or [])
    if columns:
        missing = [c for c in columns if c not in df.columns]
        for col in missing:
            df[col] = np.nan
        return df[columns]
    return df


def load_penalty_metrics():
    penalties = safe_read(
        PENALTY_PATH,
        ["game_id", "penalty_team", "defteam", "qtr"],
    )
    if penalties.empty:
        return pd.DataFrame(columns=["game_id", "defensive_rate", "qtr"])
    penalties = penalties.fillna({"penalty_team": "", "defteam": "", "qtr": 0.0})
    penalties["is_defensive"] = penalties["penalty_team"] == penalties["defteam"]
    summary = (
        penalties.groupby("game_id")
        .agg(defensive_rate=("is_defensive", "mean"), qtr=("qtr", "max"))
        .reset_index()
    )
    return summary


def flag_spike_series(row):
    base = row.flag_density * max(1.0, row.total_plays / 60.0)
    return {
        "opening_quarter": float(base * (0.5 + 0.4 * row.defensive_rate)),
        "mid_game": float(base * (0.7 + 0.3 * row.penalties / max(1, row.total_plays))),
        "late_game": float(base * (1.0 + row.late_rate * 0.9)),
        "overtime": float(base * (1.1 + row.overtime_plays / max(1.0, row.total_plays))),
    }


def hrm_sniper(row):
    blueball = row.home_tds <= 1 and row.home_fgs >= 3 and row.week > 10
    blackout = str(row.roof).lower() in {"dome", "closed", "fixed"} or bool(row.is_low_visibility)
    overtime_hell = row.overtime_plays >= 6 and row.flag_density > 0.13 and row.defensive_rate > 0.7
    flag_farm = (row.flag_density >= 0.115) or (row.penalties >= 18 and row.qtr >= 4.0 and row.overtime_plays > 4)
    if row.total_line > 0:
        under_hit = 1.0 if row.points_total <= row.total_line else 0.0
    else:
        under_hit = 0.5
    asshole = bool(row.defensive_rate > 0.68 or row.flag_density > 0.13 or row.penalties >= 18)
    if overtime_hell:
        return {
            "h_module": "FLAG_FUCKING_FARM",
            "l_module": {"under_prob": 0.93, "flag_spike": 2.6},
            "confidence": 0.96,
            "asshole_ref_detected": True,
            "flag_spike_series": flag_spike_series(row),
            "ref_bias_therapy_needed": 100.0,
        }
    if blueball and blackout:
        return {
            "h_module": "SCRIPTED_BLUEBALL",
            "l_module": {"under_prob": 0.91, "flag_spike": 1.8},
            "confidence": 0.97,
            "asshole_ref_detected": asshole,
            "flag_spike_series": flag_spike_series(row),
            "ref_bias_therapy_needed": 88.0 if asshole else 45.0,
        }
    if flag_farm:
        return {
            "h_module": "FLAG_FARM",
            "l_module": {"under_prob": 0.82 + 0.05 * under_hit, "flag_spike": 2.1},
            "confidence": 0.9,
            "asshole_ref_detected": asshole,
            "flag_spike_series": flag_spike_series(row),
            "ref_bias_therapy_needed": 76.0 if asshole else 40.0,
        }
    if row.flag_density > 0.095 and row.week > 8:
        return {
            "h_module": "SCRIPTED_BLACKOUT",
            "l_module": {"under_prob": 0.78 + under_hit * 0.12, "flag_spike": 1.6},
            "confidence": 0.84,
            "asshole_ref_detected": asshole,
            "flag_spike_series": flag_spike_series(row),
            "ref_bias_therapy_needed": 52.0 if asshole else 20.0,
        }
    return {
        "h_module": "NORMAL_FLOW",
        "l_module": {"under_prob": 0.55 + 0.08 * under_hit, "flag_spike": 1.0},
        "confidence": 0.61,
        "asshole_ref_detected": asshole,
        "flag_spike_series": flag_spike_series(row),
        "ref_bias_therapy_needed": 18.0 if asshole else 5.0,
    }


def infer_home_splits(score):
    score = max(0.0, float(score))
    td_guess = int(score // 7)
    fg_guess = int(max(0.0, (score - td_guess * 7) // 3))
    return td_guess, fg_guess


if __name__ == "__main__":
    crew = safe_read(
        CREW_PATH,
        [
            "game_id",
            "home_score",
            "away_score",
            "flag_density",
            "penalties",
            "overtime_plays",
            "total_line",
            "points_total",
            "week",
            "total_plays",
            "is_low_visibility",
            "roof",
            "home_team",
            "season",
            "is_primetime",
        ],
    )
    if crew.empty:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps({}, indent=2))
    else:
        penalties = load_penalty_metrics()
        crew = crew.merge(penalties, on="game_id", how="left")
        defaults = {
            "away_score": 0.0,
            "flag_density": 0.0,
            "penalties": 0,
            "overtime_plays": 0.0,
            "total_line": 0.0,
            "points_total": 0.0,
            "week": 0,
            "total_plays": 60.0,
            "is_low_visibility": False,
            "roof": "outdoors",
            "defensive_rate": 0.5,
            "qtr": 4.0,
            "late_rate": 0.25,
        }
        for key, value in defaults.items():
            if key not in crew.columns:
                crew[key] = value
        crew = crew.fillna(defaults)
        crew["points_total"] = crew["points_total"].where(crew["points_total"] > 0, crew["home_score"] + crew["away_score"])
        crew["total_line"] = crew["total_line"].where(crew["total_line"] > 0, crew["points_total"])
        splits = crew["home_score"].apply(infer_home_splits)
        crew["home_tds"] = [h for h, _ in splits]
        crew["home_fgs"] = [f for _, f in splits]
        preds = crew.apply(hrm_sniper, axis=1)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w") as f:
            json.dump({str(k): v for k, v in preds.to_dict().items()}, f, indent=2)
