#!/usr/bin/env python3
# gold_standard_mlb_launcher.py
"""
Gold-Standard MLB Betting System – Launcher (real data)
Default slate: 2025-06-18
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

from fixed_gold_standard_mlb_system import FixedGoldStandardMLBSystem

# tuneables
MAX_OPPORTUNITIES, BATCH_SIZE, PARALLEL_REQUESTS = 690, 20, 5
BANKROLL, BASE_UNIT = 500.0, 5.0


# ── helpers ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--date",
        type=lambda s: date.fromisoformat(s),
        default=date(2025, 6, 18),
        help="Slate date (YYYY-MM-DD)",
    )
    return p.parse_args()


def setup_log() -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = Path("logs") / f"launcher_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(fname, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    lg = logging.getLogger("launcher")
    lg.info("Log file → %s", fname)
    return lg


def banner(slate: date) -> None:
    print(
        f"\n╔═══════════════════════════════════════════════════════════════════╗\n"
        f"║  🏆  GOLD-STANDARD MLB BETTING SYSTEM – FULL MODE ({slate})  ║\n"
        f"║  Opportunities {MAX_OPPORTUNITIES:<4} │ Batch {BATCH_SIZE:<2} │"
        f" Parallel {PARALLEL_REQUESTS}                       ║\n"
        f"╚═══════════════════════════════════════════════════════════════════╝\n",
        flush=True,
    )


def load_env() -> None:
    env = Path(".env")
    if env.is_file():
        for ln in env.read_text("utf-8").splitlines():
            if ln.strip() and not ln.lstrip().startswith("#") and "=" in ln:
                k, v = (s.strip() for s in ln.split("=", 1))
                os.environ.setdefault(k, v)


# ── async pipeline ─────────────────────────────────────────────────────
async def run_pipeline(slate: date, lg: logging.Logger) -> int:
    sys_obj = FixedGoldStandardMLBSystem(
        bankroll=BANKROLL,
        base_unit_size=BASE_UNIT,
        slate_date=slate,
        max_opportunities=MAX_OPPORTUNITIES,
        batch_size=BATCH_SIZE,
        parallel_requests=PARALLEL_REQUESTS,
    )
    lg.info("Running pipeline …")
    res = await sys_obj.run_fixed_pipeline()
    recs = res["recommendations"]

    if not recs:
        print("⚠️  No edges found.")
        return 0

    # pretty print top 10
    print("\n🎯  Top Recommendations\n───────────────────────")
    for i, r in enumerate(recs[:10], 1):
        ev, stake = r.get("expected_value", 0), r.get("stake", 0)
        roi = f"{ev / stake * 100:.1f} %" if stake else "n/a"
        print(
            f"{i:2}. {r['selection']} | Stake ${stake:.2f} | EV ${ev:.2f} | ROI {roi}"
        )

    # save results
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"results_{slate}_{datetime.now():%H%M%S}.json"
    try:
        import json

        out.write_text(json.dumps(res, indent=2))
        lg.info("Saved results → %s", out)
    except Exception as exc:  # noqa: BLE001
        lg.error("Save failed: %s", exc)
    return 0


# ── entry point ───────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    banner(args.date)
    lg = setup_log()
    load_env()

    try:
        rc = asyncio.run(run_pipeline(args.date, lg))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        rc = 1
    except Exception as exc:  # noqa: BLE001
        lg.exception("Fatal: %s", exc)
        rc = 1

    print(f"\n[EXIT] code={rc}")
    sys.exit(rc)


if __name__ == "__main__":  # pragma: no cover
    main()
