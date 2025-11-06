#!/usr/bin/env python3
"""
API Key Finder & Validator
=========================

Scans for likely API keys in:
- Environment variables
- .env and .env-like files (.env, aci.env, .env.local)
- config/api_keys.json (if present)

Outputs a masked report and optional lightweight
validation for known providers.

Usage:
  python3 scripts/find_api_keys.py            # scan & report
  python3 scripts/find_api_keys.py --validate
  # also validate reachable services

Security: Values are masked. Nothing is written back.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

TRY_FILES = [
    ".env",
    "aci.env",
    ".env.local",
    "config/api_keys.json",
]

# Map canonical service ids to env var names we support
SERVICE_ENV_MAP = {
    "odds_api": ["ODDS_API_KEY", "THE_ODDS_API_KEY"],
    "openweather": ["OPENWEATHER_API_KEY"],
    "claude": ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "perplexity": ["PERPLEXITY_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "grok": ["GROK_API_KEY", "XAI_API_KEY"],
    # extras often used in this repo
    "youtube": ["YOUTUBE_API_KEY"],
    "supabase_url": ["SUPABASE_URL"],
    "supabase_anon": ["SUPABASE_ANON_KEY"],
}


def mask(value: str) -> str:
    if not value:
        return ""
    value = value.strip()
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}…{value[-4:]}"


def load_env_file(path: Path) -> Dict[str, str]:
    results: Dict[str, str] = {}
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            # flatten common patterns
            for k, v in data.items():
                if isinstance(v, str):
                    results[k] = v
        else:
            for line in path.read_text().splitlines():
                line = line.strip()
                is_kv = (not line) or line.startswith("#") or ("=" not in line)
                if is_kv:
                    if not line or line.startswith("#"):
                        continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                results[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return results


def gather_candidates() -> Dict[str, Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {
        sid: {} for sid in SERVICE_ENV_MAP.keys()
    }

    # 1) Process environment variables
    for sid, names in SERVICE_ENV_MAP.items():
        for name in names:
            val = os.environ.get(name)
            if val:
                found[sid][f"env:{name}"] = val

    # 2) Process known files
    root = Path.cwd()
    for rel in TRY_FILES:
        p = (root / rel)
        if not p.exists():
            continue
        data = load_env_file(p)
        for sid, names in SERVICE_ENV_MAP.items():
            for name in names:
                if name in data and data[name]:
                    key_id = f"file:{p}:{name}"
                    found[sid][key_id] = data[name]

    return found


async def validate_keys(
    found: Dict[str, Dict[str, str]]
) -> Dict[str, Tuple[bool, str]]:
    import aiohttp

    results: Dict[str, Tuple[bool, str]] = {}

    async with aiohttp.ClientSession() as session:
        # The Odds API
        if found.get("odds_api"):
            key = next(iter(found["odds_api"].values()))
            try:
                url = "https://api.the-odds-api.com/v4/sports"
                params = {"apiKey": key}
                async with session.get(
                    url, params=params, timeout=10
                ) as r:
                    if r.status == 200:
                        results["odds_api"] = (True, "Valid (200)")
                    elif r.status == 401:
                        results["odds_api"] = (False, "Invalid (401)")
                    else:
                        results["odds_api"] = (False, f"HTTP {r.status}")
            except Exception as e:
                results["odds_api"] = (False, f"Error: {e}")

        # OpenWeatherMap
        if found.get("openweather"):
            key = next(iter(found["openweather"].values()))
            try:
                url = "https://api.openweathermap.org/data/2.5/weather"
                params = {"q": "Kansas City,US", "appid": key}
                async with session.get(url, params=params, timeout=10) as r:
                    if r.status == 200:
                        results["openweather"] = (True, "Valid (200)")
                    elif r.status == 401:
                        results["openweather"] = (False, "Invalid (401)")
                    else:
                        results["openweather"] = (False, f"HTTP {r.status}")
            except Exception as e:
                results["openweather"] = (False, f"Error: {e}")

        # AI providers: just presence check here; detailed tests elsewhere
        for sid in ["claude", "openai", "perplexity", "gemini", "grok"]:
            if found.get(sid):
                results[sid] = (True, "Present")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Validate reachability for supported providers"
        ),
    )
    args = parser.parse_args()

    found = gather_candidates()

    # Prepare report
    report = {
        "summary": {},
        "details": {},
    }

    for sid, sources in found.items():
        present = bool(sources)
        report["summary"][sid] = "found" if present else "missing"
        masked = {k: mask(v) for k, v in sources.items()}
        report["details"][sid] = masked

    print("\n========== API KEY SCAN ==========")
    print(json.dumps(report, indent=2))

    if args.validate:
        try:
            import asyncio
            results = asyncio.run(validate_keys(found))
        except RuntimeError:
            # already running event loop (rare when embedded), fallback
            import nest_asyncio  # type: ignore
            import asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(validate_keys(found))

        print("\n========== VALIDATION ==========")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
