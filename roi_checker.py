import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def read_json_safe(file_path: Path):
    try:
        with file_path.open("r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse {file_path}: {e}")
        return None


def convert_american_odds(price: int) -> float:
    """Convert American odds to decimal odds."""
    if price > 0:
        return (price / 100.0) + 1  # e.g., 3300 → 34.0 (profit + stake)
    else:
        return (100.0 / abs(price)) + 1  # e.g., -10000 → 1.01


def calculate_roi(picks_data: dict, closing_lines: dict) -> float:
    total_bets = 0
    total_return = 0.0
    for pick in picks_data.get("picks", []):
        game_id = pick.get("game_id")
        price = pick.get("price")
        if not game_id or not price:
            continue
        decimal_odds = convert_american_odds(price)
        for line in closing_lines.get("lines", []):
            if game_id == line.get("game_id"):
                total_bets += 1
                outcome = line.get("outcome")
                if outcome == "win":
                    # Return is stake (1 unit) + profit (decimal_odds - 1)
                    total_return += decimal_odds
                break
    if total_bets == 0:
        return 0.0
    return (total_return / total_bets) * 100


def main(date_str: str | None = None):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    picks_file = Path("data/picks") / f"{date_str}_picks.json"
    lines_file = Path("data/closing_lines") / f"{date_str}_closing.json"

    picks_data = read_json_safe(picks_file)
    lines_data = read_json_safe(lines_file)

    if picks_data and lines_data:
        roi = calculate_roi(picks_data, lines_data)
        logging.info("ROI for %s: %.2f%%", date_str, roi)
    else:
        logging.error("Cannot calculate ROI: missing data")


if __name__ == "__main__":
    main()
