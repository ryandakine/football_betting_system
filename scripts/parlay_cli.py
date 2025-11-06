#!/usr/bin/env python3
"""
Parlay CLI
==========
Quick headless checker for Parlay odds/payout/risk without the GUI.

Usage examples:
  python3 scripts/parlay_cli.py \
    -b "Kansas City Chiefs -120" \
    -b "Buffalo Bills +140" \
    -b "Eagles vs Cowboys - ML -150" \
    -s 10

  python3 scripts/parlay_cli.py \
    -f bets.txt \
    -s 25 \
    --json

The script imports ParlayCalculator from football_master_gui.py.
"""

import argparse
import json
import sys
from pathlib import Path


def ensure_project_on_path():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Parlay headless checker")
    parser.add_argument(
        "-b",
        "--bet",
        action="append",
        help="Bet string, e.g. 'Team -120' or 'Team vs Team - ML -150'",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to file with one bet per line",
    )
    parser.add_argument(
        "-s",
        "--stake",
        type=float,
        default=10.0,
        help="Stake amount (default 10.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON only",
    )

    args = parser.parse_args()

    bets = list(args.bet or [])
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        for line in file_path.read_text().splitlines():
            line = line.strip()
            if line:
                bets.append(line)

    if not bets:
        print("❌ No bets provided. Use -b or -f.", file=sys.stderr)
        sys.exit(1)

    ensure_project_on_path()
    from football_master_gui import ParlayCalculator  # noqa: E402

    calc = ParlayCalculator()
    res = calc.calculate_parlay_odds(bets)
    payout = calc.calculate_payout(res, stake=args.stake)
    risk = calc.get_risk_assessment(res)

    out = {
        "bets": bets,
        "odds": {
            "decimal": res.get("decimal_odds"),
            "american": res.get("american_odds"),
            "implied_probability_percent": res.get("implied_probability"),
            "legs": res.get("legs"),
        },
        "payout": payout,
        "risk": risk,
        "warnings": res.get("correlation_warnings", []),
        "risk_factors": res.get("risk_factors", []),
        "parsed_bets": res.get("parsed_bets", []),
    }

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print("\n📊 Parlay Summary")
    print("=" * 40)
    print(f"Bets (x{out['odds']['legs']}):")
    for i, b in enumerate(bets, 1):
        print(f"  {i}. {b}")
    print()
    odds_line = (
        f"Odds: {out['odds']['american']:+.0f}  "
        f"(decimal {out['odds']['decimal']})  "
        f"Implied {out['odds']['implied_probability_percent']:.2f}%"
    )
    print(odds_line)
    payout_line = (
        f"Stake: ${payout['stake']:.2f}  Payout: ${payout['payout']:.2f}  "
        f"Profit: ${payout['profit']:.2f} ROI: {payout['roi_percent']:.2f}%"
    )
    print(payout_line)
    print(f"Risk: {risk}")
    if out["warnings"] or out["risk_factors"]:
        print("\n⚠️ Notes:")
        for w in out["warnings"] + out["risk_factors"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()

