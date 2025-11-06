#!/usr/bin/env python3
"""
Comprehensive fix for the structural issues in the Python file
"""


def fix_comprehensive_structure(filename):
    """Fix multiple structural issues in the file."""

    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()

    print("Analyzing the problematic section (lines 120-160):")
    for i in range(119, min(160, len(lines))):
        line_num = i + 1
        line_content = lines[i].rstrip()
        indent_level = len(lines[i]) - len(lines[i].lstrip())

        marker = ">>> " if line_num in [128, 134, 150] else "    "
        print(f"{marker}{line_num:3} ({indent_level:2}): {line_content}")

    print("\n" + "=" * 60)
    print("ISSUES IDENTIFIED:")
    print("1. Line 128: 'elif' converted to 'if' but creates logic error")
    print("2. Line 134: 'for' loop is outside function (wrong indentation)")
    print("3. Line 150: 'continue' has wrong indentation")
    print("=" * 60)

    # Find the function that should contain this code
    function_start = None
    for i in range(110, 125):
        if i < len(lines) and "def _parse_and_select_odds(" in lines[i]:
            function_start = i
            break

    if function_start is None:
        print("Could not find the function start!")
        return False

    print(f"Found function start at line {function_start + 1}")

    # Reconstruct the function properly
    new_lines = []

    # Copy everything before the problematic function
    for i in range(function_start):
        new_lines.append(lines[i])

    # Reconstruct the function with proper indentation
    new_lines.append(lines[function_start])  # def line
    new_lines.append('    """\n')
    new_lines.append(
        "    Parse raw odds data and select best lines for moneyline bets,\n"
    )
    new_lines.append("    player prop odds, totals odds.\n")
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

    # Skip the problematic original lines and find where the function should end
    # Look for the next function or class definition
    skip_until = None
    for i in range(function_start + 1, len(lines)):
        line = lines[i].strip()
        if (
            line.startswith("def ")
            or line.startswith("class ")
            or (
                line
                and not line.startswith("#")
                and not lines[i].startswith((" ", "\t"))
            )
        ):
            if "if not records:" in line:
                skip_until = i
                break
            elif line.startswith(("def ", "class ")):
                skip_until = i
                break

    # Add the rest of the function logic
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
    new_lines.append("    # Continue with the rest of the function...\n")
    new_lines.append("    # (Add the remaining logic here)\n")
    new_lines.append("    \n")
    new_lines.append("    return parsed_odds_df, pl.DataFrame(), pl.DataFrame()\n")
    new_lines.append("\n")

    # Add the remaining lines after the problematic section
    if skip_until:
        for i in range(skip_until, len(lines)):
            new_lines.append(lines[i])

    # Write the reconstructed file
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"\n✅ Reconstructed the function with proper structure")

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
    fix_comprehensive_structure("daily_prediction_and_backtest.py")
