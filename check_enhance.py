#!/usr/bin/env python3
"""
Check the current file structure and enhance it safely
"""


def check_current_file(filename):
    """Check what's in the current file."""

    with open(filename, encoding="utf-8") as f:
        content = f.read()

    print("=== CHECKING CURRENT FILE STRUCTURE ===")

    # Look for key functions
    functions = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            func_name = line.strip().split("(")[0].replace("def ", "")
            functions.append((func_name, i + 1))

    print(f"Found {len(functions)} functions:")
    for func_name, line_num in functions:
        print(f"  Line {line_num}: {func_name}")

    # Look for the parsing function specifically
    parsing_function_line = None
    for i, line in enumerate(lines):
        if "_parse_and_select_odds" in line and "def" in line:
            parsing_function_line = i + 1
            break

    if parsing_function_line:
        print(
            f"\nFound _parse_and_select_odds function at line {parsing_function_line}"
        )

        # Show the current implementation
        print("\nCurrent implementation:")
        start_line = parsing_function_line - 1
        for i in range(start_line, min(len(lines), start_line + 20)):
            print(f"  {i+1:3}: {lines[i]}")

        return True, parsing_function_line
    else:
        print("\n❌ Could not find _parse_and_select_odds function")
        return False, None


def enhance_by_line_replacement(filename):
    """Enhance by replacing specific lines in the parsing function."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("\n=== ENHANCING BY LINE REPLACEMENT ===")

    # Find the lines that say "For now, return empty DataFrames"
    target_line = None
    for i, line in enumerate(lines):
        if "For now, return empty DataFrames" in line:
            target_line = i
            break

    if target_line is None:
        print("Could not find target line to replace")
        return False

    print(f"Found target line at {target_line + 1}: {lines[target_line].strip()}")

    # Replace the placeholder implementation with real parsing
    enhanced_parsing = """    # Parse all odds data into records
    records = []

    for game in games_data:
        game_id = str(game.get("id", ""))
        commence_time = game.get("commence_time", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        for bookmaker in game.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key", "")

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if not isinstance(outcomes, list):
                    continue

                for outcome in outcomes:
                    outcome_name = outcome.get("name", "")
                    price = outcome.get("price", 0)
                    point = outcome.get("point")

                    records.append({
                        "game_id": game_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker_key": bookmaker_key,
                        "market_key": market_key,
                        "outcome_name": outcome_name,
                        "price": float(price) if price else 0.0,
                        "point": float(point) if point else None,
                    })

    if not records:
        logger.warning("No odds records parsed from API response")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    # Convert to DataFrame and separate market types
    parsed_odds_df = pl.DataFrame(records)
    logger.info(f"SUCCESS: Parsed {len(records)} odds records from {len(games_data)} games")

    # Separate market types
    moneyline_odds = parsed_odds_df.filter(pl.col("market_key") == "h2h")
    totals_odds = parsed_odds_df.filter(pl.col("market_key") == "totals")
    player_props = parsed_odds_df.filter(pl.col("market_key").str.starts_with("player_"))

    logger.info(f"Market breakdown: Moneyline({len(moneyline_odds)}) Totals({len(totals_odds)}) Props({len(player_props)})")

    # Find best moneyline odds
    if not moneyline_odds.is_empty():
        best_moneyline = moneyline_odds.group_by(["game_id", "outcome_name"]).agg([
            pl.col("price").max().alias("best_price"),
            pl.col("bookmaker_key").first().alias("best_bookmaker"),
            pl.col("home_team").first().alias("home_team"),
            pl.col("away_team").first().alias("away_team"),
            pl.col("commence_time").first().alias("commence_time"),
        ])
        logger.info(f"BEST ODDS: Found {len(best_moneyline)} moneyline outcomes")
    else:
        best_moneyline = pl.DataFrame()

    # Find best totals odds
    if not totals_odds.is_empty():
        best_totals = totals_odds.group_by(["game_id", "outcome_name", "point"]).agg([
            pl.col("price").max().alias("best_price"),
            pl.col("bookmaker_key").first().alias("best_bookmaker"),
            pl.col("home_team").first().alias("home_team"),
            pl.col("away_team").first().alias("away_team"),
            pl.col("commence_time").first().alias("commence_time"),
        ])
        logger.info(f"BEST TOTALS: Found {len(best_totals)} totals outcomes")
    else:
        best_totals = pl.DataFrame()

    return best_moneyline, player_props, best_totals
"""

    # Find where to insert the new code (replace the placeholder return statement)
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # If this is the target line, replace it and the next few lines
        if i == target_line:
            # Skip the comment and return statement
            while i < len(lines) and not lines[i].strip().startswith("return"):
                i += 1

            # Skip the return statement
            if i < len(lines):
                i += 1

            # Insert our enhanced parsing
            for enhanced_line in enhanced_parsing.split("\n"):
                new_lines.append(enhanced_line + "\n")
        else:
            new_lines.append(line)
            i += 1

    # Write the enhanced file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print("SUCCESS: Enhanced the parsing function!")
    return True


def test_compilation(filename):
    """Test if the enhanced file compiles."""
    try:
        with open(filename, encoding="utf-8") as f:
            compile(f.read(), filename, "exec")
        print("SUCCESS: Enhanced file compiles!")
        return True
    except SyntaxError as e:
        print(f"ERROR: Syntax error on line {e.lineno}: {e.msg}")
        return False


if __name__ == "__main__":
    filename = "daily_prediction_and_backtest.py"

    # First check what's in the file
    found_function, line_num = check_current_file(filename)

    if found_function:
        # Try to enhance it
        if enhance_by_line_replacement(filename):
            if test_compilation(filename):
                print("\n=== ENHANCEMENT COMPLETE ===")
                print("Test your enhanced system:")
                print("python daily_prediction_and_backtest.py")
                print("\nYou should now see detailed odds parsing!")
            else:
                print("Enhancement failed - compilation error")
        else:
            print("Enhancement failed - could not modify function")
    else:
        print("Cannot enhance - function not found")
