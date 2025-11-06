#!/usr/bin/env python3
"""
Data ingestion and analysis pipeline for the NFL Referee Conspiracy Engine.

This module orchestrates fetching officiating assignments, play-by-play penalties,
and schedule context for the 2018-2024 seasons. It stores normalized tables that
downstream analytics (crew clustering, team reports) can build on.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from mistletoe import Document
from mistletoe.ast_renderer import ASTRenderer
from nfl_data_py import import_officials, import_pbp_data, import_schedules
from tqdm import tqdm

DATA_DIR = Path(os.getenv("REF_CONSPIRACY_DATA_DIR", "data/referee_conspiracy"))
DEFAULT_SEASONS = list(range(2018, 2025))
CACHE_DIR = Path(os.getenv("REF_CONSPIRACY_CACHE_DIR", DATA_DIR / "cache"))
CACHE_TTL_SECONDS = int(os.getenv("REF_CONSPIRACY_CACHE_TTL", 900))
STADIUM_CONFIG_PATH = Path(os.getenv("STADIUM_CONFIG_PATH", DATA_DIR / "stadium_config.json"))

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    seasons: List[int]
    officials_path: Path
    penalties_path: Path
    schedules_path: Path

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_scrape(key: str, fetcher, ttl: int = CACHE_TTL_SECONDS, *args, **kwargs):
    cache_file = CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        with cache_file.open("rb") as fh:
            timestamp, payload = pickle.load(fh)
        if time.time() - timestamp < ttl:
            return payload
    result = fetcher(*args, **kwargs)
    with cache_file.open("wb") as fh:
        pickle.dump((time.time(), result), fh)
    return result


def _officials_cache_path(seasons: Iterable[int]) -> Path:
    start, end = min(seasons), max(seasons)
    return DATA_DIR / f"officials_{start}_{end}.parquet"


def _penalties_cache_path(seasons: Iterable[int]) -> Path:
    start, end = min(seasons), max(seasons)
    return DATA_DIR / f"penalties_{start}_{end}.parquet"


def _schedules_cache_path(seasons: Iterable[int]) -> Path:
    start, end = min(seasons), max(seasons)
    return DATA_DIR / f"schedules_{start}_{end}.parquet"


def load_officials(seasons: Iterable[int], force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch officiating assignments (crew chief + positions) for each game.
    Cached to parquet to avoid hammering the upstream source repeatedly.
    """
    ensure_data_dir()
    path = _officials_cache_path(seasons)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    try:
        df = import_officials(list(seasons))
        df.to_parquet(path, index=False)
        validate_df(df, ["game_id", "official_id", "off_pos"], "officials")
        return df
    except Exception as exc:
        if path.exists():
            logger.warning("Official import failed (%s); using cached data from %s.", exc, path)
            return pd.read_parquet(path)
        raise


def _load_penalty_season(season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pbp = import_pbp_data([season])
    pbp["season"] = season

    penalties = (
        pbp[pbp["penalty"] == 1]
        .loc[
            :,
            [
                "game_id",
                "play_id",
                "posteam",
                "defteam",
                "penalty_team",
                "penalty_type",
                "penalty_yards",
                "score_differential",
                "score_differential_post",
                "quarter_end",
                "qtr",
            ],
        ]
        .copy()
    )
    penalties["season"] = season
    penalties["score_swing"] = (
        penalties["score_differential_post"] - penalties["score_differential"]
    )

    plays = (
        pbp.groupby("game_id", group_keys=False)
        .apply(
            lambda df: pd.Series(
                {
                    "total_plays": df["play_id"].count(),
                    "regulation_plays": (df["qtr"] <= 4).sum(),
                    "overtime_plays": (df["qtr"] > 4).sum(),
                }
            )
        )
        .reset_index()
    )
    plays["season"] = season
    return penalties, plays


def load_penalties(seasons: Iterable[int], force_refresh: bool = False) -> pd.DataFrame:
    """
    Pull play-by-play data and retain only penalty snaps plus game-level play counts.
    """
    ensure_data_dir()
    path = _penalties_cache_path(seasons)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)


    seasons_list = list(seasons)
    penalty_frames: List[pd.DataFrame] = []
    play_counts: List[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=min(8, len(seasons_list))) as executor:
        future_map = {executor.submit(_load_penalty_season, season): season for season in seasons_list}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Downloading play-by-play"):
            penalties, plays = future.result()
            penalty_frames.append(penalties)
            play_counts.append(plays)

    penalties = pd.concat(penalty_frames, ignore_index=True)
    play_counts_df = pd.concat(play_counts, ignore_index=True)
    penalties = penalties.merge(play_counts_df, on=["game_id", "season"], how="left")
    penalties.to_parquet(path, index=False)
    validate_df(penalties, ["game_id", "play_id", "penalty"], "penalties")
    return penalties


def load_schedules(seasons: Iterable[int], force_refresh: bool = False) -> pd.DataFrame:
    """
    Pull schedule records with scoring, location, and betting context.
    """
    ensure_data_dir()
    path = _schedules_cache_path(seasons)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    try:
        df = import_schedules(list(seasons))
        df.to_parquet(path, index=False)
        validate_df(df, ["game_id", "away_team", "home_team"], "schedules")
        return df
    except Exception as exc:
        if path.exists():
            logger.warning("Schedule import failed (%s); using cached data from %s.", exc, path)
            return pd.read_parquet(path)
        raise


def fetch_all(seasons: Iterable[int], force_refresh: bool = False) -> DatasetBundle:
    """
    Materialize the three core tables and report where they landed.
    """
    seasons_list = sorted(set(seasons))
    officials = load_officials(seasons_list, force_refresh=force_refresh)
    penalties = load_penalties(seasons_list, force_refresh=force_refresh)
    schedules = load_schedules(seasons_list, force_refresh=force_refresh)

    bundle = DatasetBundle(
        seasons=seasons_list,
        officials_path=_officials_cache_path(seasons_list),
        penalties_path=_penalties_cache_path(seasons_list),
        schedules_path=_schedules_cache_path(seasons_list),
    )

    summary = {
        "officials_rows": len(officials),
        "penalty_rows": len(penalties),
        "schedule_rows": len(schedules),
        "seasons": seasons_list,
    }
    print(json.dumps(summary, indent=2))
    return bundle


def validate_df(df: pd.DataFrame, columns: List[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} dataframe missing columns: {missing}")


def _seasonal_ref_timing(
    base_probability: float,
    adjustment: float,
    script_weight: float,
    context_factor: float = 1.0,
) -> Dict[str, float]:
    weight = max(0.0, min(0.8, script_weight * context_factor))
    blend = 1.0 / (1.0 + pow(2.71828, -(weight * 10 - 5)))
    probability = base_probability * (1 - blend) + (base_probability + adjustment) * blend
    cap = 0.8 if weight > 0.3 else 0.5
    probability = max(0.0, min(cap, probability))
    logger.debug(
        "Seasonal blend | base=%.3f adj=%.3f weight=%.3f context=%.2f -> prob=%.3f",
        base_probability,
        adjustment,
        weight,
        context_factor,
        probability,
    )
    return {"probability": probability, "blend": blend, "cap": cap}


def _flatten_ast_text(node: Any) -> str:
    if isinstance(node, dict):
        if node.get("type") in {"text", "raw_text"}:
            return node.get("content", "")
        return " ".join(_flatten_ast_text(child) for child in node.get("children", []))
    if isinstance(node, list):
        return " ".join(_flatten_ast_text(child) for child in node)
    return str(node)


def _load_narrative_notes(markdown_path: Path = DATA_DIR / "team_autopsy_notes.md") -> Dict[str, Dict[str, Any]]:
    if not markdown_path.exists():
        logger.warning("Narrative markdown not found at %s", markdown_path)
        return {}
    entries: Dict[str, Dict[str, Any]] = {}
    with ASTRenderer() as renderer:
        ast = renderer.render(Document(markdown_path.read_text()))
    children = ast.get("children", [])
    idx = 0
    while idx < len(children):
        node = children[idx]
        if node.get("type") == "heading" and node.get("level") == 3:
            team = _flatten_ast_text(node).strip()
            idx += 1
            fragments: List[str] = []
            while idx < len(children):
                nxt = children[idx]
                if nxt.get("type") == "heading" and nxt.get("level") <= 3:
                    break
                fragments.append(_flatten_ast_text(nxt).strip())
                idx += 1
            notes = [frag for frag in fragments if frag]
            joined = " ".join(notes).lower()
            script_weight = 0.5
            for token, value in [
                ("cash cow", 0.8),
                ("dynasty", 0.8),
                ("protected", 0.78),
                ("hero", 0.72),
                ("rebuild", 0.65),
                ("tank", 0.6),
                ("filler", 0.55),
            ]:
                if token in joined:
                    script_weight = max(script_weight, value)
            explicit = re.search(r"script weight[:\s]+([0-9]+)", joined)
            if explicit:
                script_weight = min(0.8, float(explicit.group(1)) / 100.0)
            bet_edge = None
            for note in notes:
                if "bet edge" in note.lower():
                    bet_edge = note.split(":", 1)[-1].strip()
                    break
            entries[team] = {
                "script_weight": round(script_weight, 3),
                "bet_edge": bet_edge,
                "notes": notes,
            }
        else:
            idx += 1
    json_path = DATA_DIR / "narrative_notes.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(entries, indent=2))
    return entries


def _load_stadium_keywords() -> List[str]:
    if STADIUM_CONFIG_PATH.exists():
        try:
            data = json.loads(STADIUM_CONFIG_PATH.read_text())
            return [kw.lower() for kw in data.get("overseas_keywords", [])]
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse stadium config: %s", exc)
    return [
        "wembley",
        "tottenham",
        "allianz",
        "deutsche",
        "twickenham",
        "estadio",
        "mexico",
        "frankfurt",
    ]


def _fetch_nfl_crews() -> Optional[Dict[str, Any]]:
    url = "https://www.nfl.com/inside-football/operations/assignments/"

    def _fetch():
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return {"source": "nfl.com", "length": len(resp.text)}

    try:
        return cache_scrape("nfl_crews", _fetch)
    except Exception as exc:
        logger.debug("NFL crew scrape failed: %s", exc)
        return None


def _fetch_noaa_weather(lat: float = 39.7392, lon: float = -104.9903) -> Optional[Dict[str, Any]]:
    endpoint = f"https://api.weather.gov/points/{lat},{lon}"

    def _fetch():
        resp = requests.get(endpoint, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        forecast_url = data["properties"]["forecastHourly"]
        forecast = requests.get(forecast_url, timeout=10).json()
        first = forecast["properties"]["periods"][0]
        return {"probabilityOfPrecipitation": first.get("probabilityOfPrecipitation", {}).get("value")}

    try:
        return cache_scrape(f"noaa_{lat}_{lon}", _fetch, ttl=1800)
    except Exception as exc:
        logger.debug("NOAA weather fetch failed: %s", exc)
        return None


def _fetch_blackout_data() -> Optional[Dict[str, Any]]:
    url = "https://www.nflblackout.com/api/current"

    def _fetch():
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        resp.raise_for_status()

    try:
        return cache_scrape("blackout_zones", _fetch, ttl=600)
    except Exception as exc:
        logger.debug("Blackout fetch failed: %s", exc)
        return None


def _fetch_reddit_noise(subreddit: str = "nfl") -> Optional[int]:
    reddit_id = os.getenv("REDDIT_ID", "")
    reddit_secret = os.getenv("REDDIT_SECRET", "")
    reddit_agent = os.getenv("REDDIT_AGENT", "ref-conspiracy-engine")

    def _fetch():
        import praw

        client = praw.Reddit(
            client_id=reddit_id,
            client_secret=reddit_secret,
            user_agent=reddit_agent,
        )
        submissions = client.subreddit(subreddit).hot(limit=25)
        return sum("chant" in (s.title.lower() + s.selftext.lower()) for s in submissions)

    try:
        return cache_scrape(f"reddit_noise_{subreddit}", _fetch, ttl=600)
    except Exception as exc:
        logger.debug("Reddit fetch failed: %s", exc)
        return None


def live_mode(bundle: DatasetBundle, poll_seconds: int = 900) -> None:
    logger.info("Entering live mode with poll interval %s seconds", poll_seconds)
    narrative = _load_narrative_notes()
    cached_crews = cached_weather = cached_blackout = cached_noise = "Unknown"
    while True:
        try:
            crews = _fetch_nfl_crews()
            if crews is not None:
                cached_crews = crews
        except Exception as exc:
            logger.debug("Live crews fetch error: %s", exc)
        try:
            weather = _fetch_noaa_weather()
            if weather is not None:
                cached_weather = weather
        except Exception as exc:
            logger.debug("Live weather fetch error: %s", exc)
        try:
            blackout = _fetch_blackout_data()
            if blackout is not None:
                cached_blackout = blackout
        except Exception as exc:
            logger.debug("Live blackout fetch error: %s", exc)
        try:
            reddit_noise = _fetch_reddit_noise()
            if reddit_noise is not None:
                cached_noise = reddit_noise
        except Exception as exc:
            logger.debug("Live reddit fetch error: %s", exc)

        logger.info(
            "Live poll | crews=%s weather=%s blackout=%s reddit_chants=%s narrative_entries=%d",
            bool(cached_crews),
            cached_weather,
            bool(cached_blackout),
            cached_noise,
            len(narrative),
        )
        time.sleep(60)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download officiating, penalty, and schedule data for NFL seasons."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        help="List of seasons to ingest (e.g., 2018 2019 ...). Defaults to 2018-2024 inclusive.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached parquet outputs and refetch everything.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Dump the stored file locations as JSON for downstream tooling.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live polling mode for mid-game adjustments.",
    )
    parser.add_argument(
        "--json-export",
        type=str,
        help="Write dataset bundle summary to the given JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    bundle = fetch_all(args.seasons, force_refresh=args.force_refresh)
    if args.print_paths:
        print(bundle.to_json())
    if args.json_export:
        Path(args.json_export).write_text(bundle.to_json())
    if args.live:
        live_mode(bundle)


if __name__ == "__main__":
    main()


try:  # pragma: no cover - optional dependency for tests
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


if pytest:  # pragma: no cover - only executed during tests

    @pytest.fixture
    def mock_requests(requests_mock):
        base = "https://api.weather.gov/points/39.7392,-104.9903"
        forecast_url = "https://example.com/forecast"
        requests_mock.get(base, json={"properties": {"forecastHourly": forecast_url}})
        requests_mock.get(
            forecast_url,
            json={"properties": {"periods": [{"probabilityOfPrecipitation": {"value": 42}}]}},
        )
        requests_mock.get("https://www.nflblackout.com/api/current", json={"status": "ok"})
        return requests_mock


    if pytest:

        def test_get_noaa_weather(mock_requests):
            data = _fetch_noaa_weather()
            assert data and data.get("probabilityOfPrecipitation") == 42


        def test_get_fan_noise_from_reddit(monkeypatch):
            class FakeSubmission:
                def __init__(self, title, text):
                    self.title = title
                    self.selftext = text

            class FakeSubreddit:
                def hot(self, limit):
                    return [
                        FakeSubmission("Chant loud", "chant chant"),
                        FakeSubmission("Regular post", "nothing"),
                    ]

            class FakeReddit:
                def __init__(self, *args, **kwargs):
                    pass

                def subreddit(self, _):
                    return FakeSubreddit()

            monkeypatch.setenv("REDDIT_ID", "id")
            monkeypatch.setenv("REDDIT_SECRET", "secret")
            monkeypatch.setenv("REDDIT_AGENT", "agent")
            import praw

            monkeypatch.setattr(praw, "Reddit", FakeReddit)
            result = _fetch_reddit_noise()
            assert result == 1
