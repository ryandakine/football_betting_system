#!/usr/bin/env python3
"""
odds_monitor.py – watch a JSON odds file and pretty-print fresh updates.

* Uses watchdog for instant updates (falls back to polling).
* Rich-styled logs if 'rich' is installed.
* Debounces duplicate payloads.
* New: retry after 100 ms if JSON is corrupt (writer was still writing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# ── optional rich logger ──────────────────────────────────────────
try:
    from rich.console import Console
    from rich.logging import RichHandler

    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console = None

log = logging.getLogger("odds-monitor")


# ── helper --------------------------------------------------------
def digest(obj: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def pretty_print(payload: dict[str, Any]) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    games = payload.get("total_games", len(payload.get("games", [])))
    tigers = len(payload.get("tigers_games", []))
    header = f"📊  Updated {games} MLB games  ({ts})"
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(f"🐅  Tigers games found: {tigers}")
    extra = games - tigers
    if extra:
        print(f"📊  Plus {extra} other MLB games updated")
    print(f"     Last update: {payload.get('last_update', 'unknown')}")
    print(sep, end="\n\n")


def load_json(path: Path) -> dict[str, Any] | None:
    """Read JSON; retry once after 100 ms if decode fails."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        time.sleep(0.1)  # writer might still be writing
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.error("Corrupt JSON after retry (%s): %s", path.name, exc)
            return None
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to read %s – %s", path.name, exc)
        return None


# ── watchdog path -------------------------------------------------
def run_with_watchdog(target: Path) -> None:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    class Handler(FileSystemEventHandler):
        _last_hash: str | None = None

        def on_modified(self, event):
            if Path(event.src_path).resolve() != target:
                return
            data = load_json(target)
            if not data:
                return
            h = digest(data)
            if h != self._last_hash:
                self._last_hash = h
                pretty_print(data)

    log.info("🎯 Using watchdog for real-time updates.")
    observer = Observer()
    observer.schedule(Handler(), str(target.parent), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ── polling fallback ---------------------------------------------
def run_with_polling(target: Path, interval: float) -> None:
    log.info("⏱  watchdog unavailable → polling every %.1f s", interval)
    last_mtime = 0.0
    last_hash: str | None = None
    try:
        while True:
            if target.exists():
                mtime = target.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    data = load_json(target)
                    if data:
                        h = digest(data)
                        if h != last_hash:
                            last_hash = h
                            pretty_print(data)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n👋  Stopping odds monitor.")


# ── CLI -----------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Watch odds JSON file and print updates.")
    p.add_argument(
        "file",
        nargs="?",
        default="fresh_odds.json",
        help="Path to JSON file (default: fresh_odds.json)",
    )
    p.add_argument(
        "--interval",
        "-i",
        type=float,
        default=2.0,
        help="Polling interval in seconds if watchdog not available (def 2)",
    )
    args = p.parse_args()
    target = Path(args.file).expanduser().resolve()

    print("🚀 Odds monitor started.")
    print("📄 Watching:", target)

    try:
        import watchdog  # noqa: F401

        run_with_watchdog(target)
    except ModuleNotFoundError:
        run_with_polling(target, args.interval)


if __name__ == "__main__":
    main()
