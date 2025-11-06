#!/usr/bin/env python3
"""
Simple, safe enhancement of the working system
"""


def enhance_odds_parsing_only(filename):
    """Just enhance the odds parsing function safely."""

    with open(filename) as f:
        content = f.read()

    # Find and replace just the _parse_and_select_odds function
    old_function_start = content.find("def _parse_and_select_odds(")
    old_function_end = content.find(
        "\nclass DailyPredictionManager:", old_function_start
    )

    if old_function_start == -1 or old_function_end == -1:
        print("❌ Could not find function to replace")
        return False

    # New enhanced parsing function
    new_function = '''def _parse_and_select_odds(raw_odds_data) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Parse raw odds data and return structured DataFrames."""
    logger.info("Parsing raw odds data and selecting best lines...")

    # Handle both dict and list responses from the API
    if isinstance(raw_odds_data, dict):
        games_data = raw_odds_data.get("data", [])
    elif isinstance(raw_odds_data, list):
        games_data = raw_odds_data
    else:
        games_data = []

    if not games_data:
        logger.warning("No games data available")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    logger.info(f"Processing {len(games_data)} games from API...")

    # Parse all odds data
    records = []

    for game in games_data:
        game_id = str(game.get("id", ""))
        commence_time = game.get("commence_time", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        logger.debug(f"Processing game: {away_team} @ {home_team}")

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

    # Convert to DataFrame
    parsed_odds_df = pl.DataFrame(records)
    logger.info(f"SUCCESS Parsed {len(records)} odds records from {len(games_data)} games")

    # Separate different market types and find best odds
    moneyline_odds = parsed_odds_df.filter(pl.col("market_key") == "h2h")
    totals_odds = parsed_odds_df.filter(pl.col("market_key") == "totals")
    player_props = parsed_odds_df.filter(pl.col("market_key").str.starts_with("player_"))

    # Log what we found
    logger.info(f"Market breakdown:")
    logger.info(f"   Moneyline odds: {len(moneyline_odds)} records")
    logger.info(f"   Totals odds: {len(totals_odds)} records")
    logger.info(f"   Player props: {len(player_props)} records")

    # Process moneyline odds to find best prices
    if not moneyline_odds.is_empty():
        try:
            best_moneyline = moneyline_odds.group_by(
                ["game_id", "outcome_name"]
            ).agg([
                pl.col("price").max().alias("best_price"),
                pl.col("bookmaker_key").first().alias("best_bookmaker"),
                pl.col("home_team").first().alias("home_team"),
                pl.col("away_team").first().alias("away_team"),
                pl.col("commence_time").first().alias("commence_time"),
            ])

            logger.info(f"BEST ODDS Found best moneyline odds for {len(best_moneyline)} outcomes")

            # Show sample of best odds
            if len(best_moneyline) > 0:
                sample = best_moneyline.head(3)
                for row in sample.iter_rows(named=True):
                    logger.info(f"   {row['outcome_name']}: {row['best_price']:.2f} ({row['best_bookmaker']})")

        except Exception as e:
            logger.error(f"Error processing moneyline odds: {e}")
            best_moneyline = pl.DataFrame()
    else:
        best_moneyline = pl.DataFrame()

    # Process totals odds
    if not totals_odds.is_empty():
        try:
            best_totals = totals_odds.group_by(
                ["game_id", "outcome_name", "point"]
            ).agg([
                pl.col("price").max().alias("best_price"),
                pl.col("bookmaker_key").first().alias("best_bookmaker"),
                pl.col("home_team").first().alias("home_team"),
                pl.col("away_team").first().alias("away_team"),
                pl.col("commence_time").first().alias("commence_time"),
            ])
            logger.info(f"BEST TOTALS Found best totals odds for {len(best_totals)} outcomes")
        except Exception as e:
            logger.error(f"Error processing totals odds: {e}")
            best_totals = pl.DataFrame()
    else:
        best_totals = pl.DataFrame()

    return best_moneyline, player_props, best_totals

'''

    # Replace the function
    new_content = (
        content[:old_function_start] + new_function + content[old_function_end:]
    )

    # Write the updated file with proper encoding
    with open(filename, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("Enhanced odds parsing function updated!")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("File compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"Syntax error on line {e.lineno}: {e.msg}")
        return False


if __name__ == "__main__":
    if enhance_odds_parsing_only("daily_prediction_and_backtest.py"):
        print("SUCCESS Test the enhanced system:")
        print("python daily_prediction_and_backtest.py")
        print("\nThis enhancement will:")
        print("- Parse all 15 games' odds data")
        print("- Separate moneyline, totals, and player props")
        print("- Find best odds across bookmakers")
        print("- Provide detailed logging of what was found")
    else:
        print("\nEnhancement failed - keeping original working version")
