#!/usr/bin/env python3
"""
Utility for retrieving NFL referee assignments from Football Zebras.

This module is designed to provide lightweight, cache-aware access to weekly
referee crews so higher level systems (e.g., the cloud GPU ensemble) can enrich
their feature sets without relying on manual data entry.

The fetcher attempts to:
 1. Hit the predictable assignment slug for the given week/year.
 2. Fallback to the site search listing.
 3. Cache successful responses under data/referee_conspiracy/ref_assignments/.

Usage:
    from referee_assignment_fetcher import RefereeAssignmentsFetcher

    fetcher = RefereeAssignmentsFetcher()
    week_data = fetcher.fetch_week(2024, 8)
    if week_data["ok"]:
        print(week_data["games"])
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


DEFAULT_CACHE_DIR = Path("data/referee_conspiracy/ref_assignments")


@dataclass
class AssignmentResult:
    games: Dict[str, str]
    source_url: Optional[str]
    ok: bool

    def to_json(self) -> str:
        return json.dumps(
            {
                "games": self.games,
                "source_url": self.source_url,
                "ok": self.ok,
            },
            indent=2,
        )


class RefereeAssignmentsFetcher:
    """Scrape Football Zebras for weekly referee crew assignments."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR, session: Optional[requests.Session] = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = session or requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5, status=3, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_week(self, year: int, week: int, use_cache: bool = True, delay_seconds: float = 0.0) -> Dict[str, object]:
        """
        Fetch assignments for a single week.

        Args:
            year: Season year (e.g., 2024).
            week: Week number (1-based).
            use_cache: When True (default) consult/write the on-disk cache.
            delay_seconds: Optional sleep after network fetch to respect rate limits.
        """
        cache_path = self.cache_dir / f"{year}_week_{week:02d}.json"
        if use_cache and cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                cache_path.unlink(missing_ok=True)

        result = self._get_ref_assignments(week=week, year=year)
        if delay_seconds > 0:
            time.sleep(delay_seconds)

        if result["ok"] and use_cache:
            cache_path.write_text(json.dumps(result, indent=2))
        return result

    def fetch_range(self, year: int, start_week: int, end_week: int, delay_seconds: float = 5.0) -> Dict[str, Dict[str, object]]:
        """
        Fetch a range of weeks for a season, returning a dictionary keyed by week.
        """
        assignments: Dict[str, Dict[str, object]] = {}
        for week in range(start_week, end_week + 1):
            assignments[f"week_{week:02d}"] = self.fetch_week(
                year=year,
                week=week,
                use_cache=True,
                delay_seconds=delay_seconds if week < end_week else 0.0,
            )
        return assignments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ref_assignments(self, week: int, year: int) -> Dict[str, object]:
        # Try direct slug with zero-padding first.
        for week_fmt in (f"{week:02d}", f"{week}"):
            direct_url = (
                f"https://www.footballzebras.com/{year}/assignments/week-{week_fmt}-nfl-referee-assignments-{year}/"
            )
            try:
                response = self.session.get(direct_url, timeout=10)
                response.raise_for_status()
                return self._parse_assignments(html=response.text, source_url=direct_url)
            except requests.RequestException:
                continue

        # Fall back to search.
        search_url = f"https://www.footballzebras.com/?s=week+{week}+referee+assignments+{year}"
        try:
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException:
            return {"games": {}, "source_url": None, "ok": False}

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article")
        if not articles:
            return {"games": {}, "source_url": None, "ok": False}

        first_article = articles[0]
        link_tag = first_article.find("a", href=True)
        if not link_tag:
            return {"games": {}, "source_url": None, "ok": False}

        article_url = link_tag["href"]
        try:
            article_resp = self.session.get(article_url, timeout=10)
            article_resp.raise_for_status()
            return self._parse_assignments(html=article_resp.text, source_url=article_url)
        except requests.RequestException:
            return {"games": {}, "source_url": None, "ok": False}

    def _parse_assignments(self, html: str, source_url: str) -> Dict[str, object]:
        soup = BeautifulSoup(html, "html.parser")
        content = soup.find("div", class_="entry-content")
        if not content:
            return {"games": {}, "source_url": source_url, "ok": False}

        assignments: Dict[str, str] = {}
        for p_tag in content.find_all("p"):
            text = p_tag.get_text().strip()
            if not text:
                continue
            normalized = text.replace("@", " — ")
            match = re.match(r"^(.+?)\s+at\s+(.+?)\s+—\s+(.+)$", normalized)
            if match:
                away, home, referee = match.groups()
                assignments[f"{away.strip()} at {home.strip()}"] = referee.strip()

        return {"games": assignments, "source_url": source_url, "ok": bool(assignments)}


__all__ = ["RefereeAssignmentsFetcher", "AssignmentResult"]


if __name__ == "__main__":  # pragma: no cover - manual utility
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NFL referee assignments from Football Zebras")
    parser.add_argument("year", type=int, help="Season year (e.g., 2024)")
    parser.add_argument("week", type=int, nargs="?", help="Specific week to fetch")
    parser.add_argument("--start-week", type=int, default=1, help="Start week for range fetch")
    parser.add_argument("--end-week", type=int, default=None, help="End week for range fetch")
    parser.add_argument("--no-cache", action="store_true", help="Bypass cached results")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between requests when fetching a range")

    args = parser.parse_args()
    fetcher = RefereeAssignmentsFetcher()

    if args.week is not None:
        result = fetcher.fetch_week(args.year, args.week, use_cache=not args.no_cache)
        print(json.dumps(result, indent=2))
    else:
        end_week = args.end_week or args.start_week
        result = fetcher.fetch_range(args.year, args.start_week, end_week, delay_seconds=args.delay)
        print(json.dumps(result, indent=2))
