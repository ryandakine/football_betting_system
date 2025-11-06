#!/usr/bin/env python3
"""
Parlay Auto
===========
Automatically fetches NFL odds, selects a simple 3-leg parlay, and prints
odds/payout/risk. No GUI required.

Usage:
  python3 scripts/parlay_auto.py

Options (env):
  ODDS_API_KEY: API key for The Odds API (recommended)
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
import json


def ensure_project_on_path():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


async def fetch_structured_odds():
    ensure_project_on_path()
    from football_odds_fetcher import FootballOddsFetcher

    api_key = os.getenv("ODDS_API_KEY", "")
    # sport_key defaults to americanfootball_nfl
    async with FootballOddsFetcher(api_key=api_key) as f:
        return await f.get_all_odds_with_props()


def _to_american_from_maybe(val: float) -> float:
    # Heuristic: if absolute value > 20 and < 1000, assume already American
    # If between ~1.01 and 10, assume decimal and convert
    if val >= 1.01 and val <= 10.0:
        # decimal → american
        if val >= 2.0:
            return (val - 1.0) * 100.0
        return -100.0 / (val - 1.0)
    return float(val)


def pick_bets(structured) -> list[str]:
    # Prefer H2H bets; choose three reasonable favorites by closest to -130
    bets = []
    try:
        for b in structured.h2h_bets:
            # home
            ha = _to_american_from_maybe(b.home_odds)
            bets.append((abs(abs(ha) - 130), f"{b.home_team} {ha:+.0f}"))
            # away
            aa = _to_american_from_maybe(b.away_odds)
            bets.append((abs(abs(aa) - 130), f"{b.away_team} {aa:+.0f}"))
    except Exception:
        pass

    if not bets:
        return []

    bets.sort(key=lambda x: x[0])
    top = [b for _, b in bets[:3]]
    return top


def compute_parlay(bets: list[str], stake: float = 10.0) -> dict:
    ensure_project_on_path()
    from football_master_gui import ParlayCalculator

    calc = ParlayCalculator()
    res = calc.calculate_parlay_odds(bets)
    payout = calc.calculate_payout(res, stake=stake)
    risk = calc.get_risk_assessment(res)
    return {"bets": bets, "odds": res, "payout": payout, "risk": risk}


def print_summary(result: dict):
    bets = result["bets"]
    odds = result["odds"]
    payout = result["payout"]
    risk = result["risk"]
    print("\n📊 Auto Parlay Summary")
    print("=" * 42)
    print(f"Bets (x{odds.get('legs', 0)}):")
    for i, b in enumerate(bets, 1):
        print(f"  {i}. {b}")
    print(
        f"Odds: {odds.get('american_odds', 0):+,.0f}  "
        f"(decimal {odds.get('decimal_odds')})  "
        f"Implied {odds.get('implied_probability', 0):.2f}%"
    )
    print(
        f"Stake: ${payout['stake']:.2f}  Payout: ${payout['payout']:.2f}  "
        f"Profit: ${payout['profit']:.2f}  ROI: {payout['roi_percent']:.2f}%"
    )
    print(f"Risk: {risk}")
    warnings = odds.get("correlation_warnings", [])
    risk_factors = odds.get("risk_factors", [])
    if warnings or risk_factors:
        print("\n⚠️ Notes:")
        for w in warnings + risk_factors:
            print(f"  - {w}")


def main():
    try:
        structured = asyncio.run(fetch_structured_odds())
        bets = pick_bets(structured)
        if not bets:
            print("❌ No bets available to build a parlay.")
            sys.exit(2)
        result = compute_parlay(bets, stake=10.0)
        print_summary(result)
        # Also output JSON for automation use
        out = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "bets": result["bets"],
            "odds": {
                "american": result["odds"].get("american_odds"),
                "decimal": result["odds"].get("decimal_odds"),
                "implied_probability_percent": result["odds"].get(
                    "implied_probability"
                ),
                "legs": result["odds"].get("legs"),
            },
            "payout": result["payout"],
            "risk": result["risk"],
            "warnings": result["odds"].get("correlation_warnings", []),
        }
        path = Path("reports/auto")
        path.mkdir(parents=True, exist_ok=True)
        filename = f"auto_parlay_{out['date']}.json"
        out_file = path / filename
        out_file.write_text(json.dumps(out, indent=2))
        print("\n✅ Saved:")
        print(out_file)
    except Exception as e:
        print(f"❌ Auto parlay failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
