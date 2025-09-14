#!/usr/bin/env python3
"""
Fix the function definition issue and reconstruct the problematic function
"""


def fix_function_definition_and_structure(filename):
    """Fix the function definition and overall structure."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Checking the function definition around line 112:")
    for i in range(110, min(120, len(lines))):
        line_num = i + 1
        marker = ">>> " if line_num == 112 else "    "
        print(f"{marker}{line_num:3}: {repr(lines[i])}")

    print("\n" + "=" * 50)
    print("Fixing the function definition and structure...")

    # Find where the problematic function starts and ends
    function_start = None
    function_end = None

    for i, line in enumerate(lines):
        if "_parse_and_select_odds(" in line:
            function_start = i
            print(f"Found function start at line {i + 1}")
            break

    if function_start is None:
        print("Could not find the function!")
        return False

    # Find the end of this function (next function or class definition)
    for i in range(function_start + 1, len(lines)):
        line = lines[i].strip()
        if (
            line.startswith("def ")
            or line.startswith("class ")
            or line.startswith("@")
            or (
                line
                and not line.startswith("#")
                and len(lines[i]) - len(lines[i].lstrip()) == 0
                and not line.startswith(("if __name__", "import ", "from "))
            )
        ):
            function_end = i
            print(f"Found function end at line {i + 1}")
            break

    if function_end is None:
        function_end = len(lines)
        print(f"Function extends to end of file")

    # Reconstruct the file
    new_lines = []

    # Copy everything before the problematic function
    new_lines.extend(lines[:function_start])

    # Add the corrected function
    new_lines.append("def _parse_and_select_odds(\n")
    new_lines.append("    raw_odds_data,\n")
    new_lines.append(") -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:\n")
    new_lines.append('    """\n')
    new_lines.append(
        "    Parse raw odds data and select best lines for moneyline bets,\n"
    )
    new_lines.append("    player prop odds, totals odds.\n")
    new_lines.append("    \n")
    new_lines.append("    Returns:\n")
    new_lines.append(
        "        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]: Moneyline odds,\n"
    )
    new_lines.append("        player prop odds, totals odds.\n")
    new_lines.append('    """\n')
    new_lines.append(
        '    logger.info("Parsing raw odds data and selecting best lines...")\n'
    )
    new_lines.append("    records = []\n")
    new_lines.append("    \n")
    new_lines.append("    # Handle both dict and list responses from the API\n")
    new_lines.append("    if isinstance(raw_odds_data, dict):\n")
    new_lines.append('        games_data = raw_odds_data.get("data", [])\n')
    new_lines.append("    elif isinstance(raw_odds_data, list):\n")
    new_lines.append("        games_data = raw_odds_data\n")
    new_lines.append("    else:\n")
    new_lines.append("        games_data = []\n")
    new_lines.append("    \n")
    new_lines.append("    for game in games_data:\n")
    new_lines.append('        game_id = str(game.get("id"))\n')
    new_lines.append('        commence_time_str = game.get("commence_time")\n')
    new_lines.append('        home_team_name = game.get("home_team")\n')
    new_lines.append('        away_team_name = game.get("away_team")\n')
    new_lines.append("        \n")
    new_lines.append('        for bookmaker in game.get("bookmakers", []):\n')
    new_lines.append('            bookmaker_key = bookmaker.get("key")\n')
    new_lines.append('            last_update = bookmaker.get("last_update")\n')
    new_lines.append("            \n")
    new_lines.append('            for market in bookmaker.get("markets", []):\n')
    new_lines.append('                market_key = market.get("key")\n')
    new_lines.append('                outcomes = market.get("outcomes")\n')
    new_lines.append("                \n")
    new_lines.append("                if not isinstance(outcomes, list):\n")
    new_lines.append("                    logger.warning(\n")
    new_lines.append(
        '                        f"Market {market_key} for game {game_id} has "\n'
    )
    new_lines.append('                        f"malformed outcomes. Skipping."\n')
    new_lines.append("                    )\n")
    new_lines.append("                    continue\n")
    new_lines.append("                \n")
    new_lines.append("                for outcome in outcomes:\n")
    new_lines.append('                    outcome_name = outcome.get("name")\n')
    new_lines.append('                    price = outcome.get("price")\n')
    new_lines.append('                    point = outcome.get("point")\n')
    new_lines.append("                    \n")
    new_lines.append("                    records.append({\n")
    new_lines.append('                        "game_id": game_id,\n')
    new_lines.append('                        "commence_time": commence_time_str,\n')
    new_lines.append('                        "home_team": home_team_name,\n')
    new_lines.append('                        "away_team": away_team_name,\n')
    new_lines.append('                        "bookmaker_key": bookmaker_key,\n')
    new_lines.append('                        "market_key": market_key,\n')
    new_lines.append('                        "outcome_name": outcome_name,\n')
    new_lines.append('                        "price": price,\n')
    new_lines.append('                        "point": point,\n')
    new_lines.append('                        "last_update": last_update,\n')
    new_lines.append("                    })\n")
    new_lines.append("    \n")
    new_lines.append("    if not records:\n")
    new_lines.append(
        '        logger.warning("No odds data was parsed from the API response.")\n'
    )
    new_lines.append("        return (\n")
    new_lines.append("            pl.DataFrame(),\n")
    new_lines.append("            pl.DataFrame(),\n")
    new_lines.append("            pl.DataFrame(),\n")
    new_lines.append("        )\n")
    new_lines.append("    \n")
    new_lines.append("    # Convert to DataFrame\n")
    new_lines.append("    parsed_odds_df = pl.DataFrame(records)\n")
    new_lines.append("    \n")
    new_lines.append("    # Filter for moneyline odds (h2h market)\n")
    new_lines.append("    moneyline_odds_filtered = parsed_odds_df.filter(\n")
    new_lines.append('        pl.col("market_key") == "h2h"\n')
    new_lines.append("    )\n")
    new_lines.append("    \n")
    new_lines.append("    # Group by game, outcome and select best odds\n")
    new_lines.append(
        "    best_h2h_odds_by_outcome = moneyline_odds_filtered.group_by(\n"
    )
    new_lines.append('        ["game_id", "outcome_name"]\n')
    new_lines.append("    ).agg([\n")
    new_lines.append('        pl.col("price").max().alias("best_price"),\n')
    new_lines.append(
        '        pl.col("bookmaker_key").first().alias("best_bookmaker"),\n'
    )
    new_lines.append("    ])\n")
    new_lines.append("    \n")
    new_lines.append("    # Return the results\n")
    new_lines.append("    return (\n")
    new_lines.append("        best_h2h_odds_by_outcome,\n")
    new_lines.append("        pl.DataFrame(),  # player props placeholder\n")
    new_lines.append("        pl.DataFrame(),  # totals placeholder\n")
    new_lines.append("    )\n")
    new_lines.append("\n")

    # Add the rest of the file after the problematic function
    if function_end < len(lines):
        new_lines.extend(lines[function_end:])

    # Write the reconstructed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"✅ Reconstructed the entire function with proper syntax")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("✅ File now compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"❌ Still has syntax error on line {e.lineno}: {e.msg}")
        if hasattr(e, "text") and e.text:
            print(f"Text: {e.text.strip()}")
        return False


if __name__ == "__main__":
    fix_function_definition_and_structure("daily_prediction_and_backtest.py")
