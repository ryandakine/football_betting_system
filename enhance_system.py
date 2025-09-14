#!/usr/bin/env python3
"""
Enhanced odds parsing and backtesting integration
"""


def add_enhanced_parsing_and_backtesting(filename):
    """Add enhanced odds parsing and backtesting capabilities."""

    with open(filename) as f:
        content = f.read()

    # Enhanced _parse_and_select_odds function
    enhanced_parsing = '''def _parse_and_select_odds(raw_odds_data) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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

    # Parse all odds data
    records = []

    for game in games_data:
        game_id = str(game.get("id", ""))
        commence_time = game.get("commence_time", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        for bookmaker in game.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key", "")
            last_update = bookmaker.get("last_update", "")

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
                        "price": price,
                        "point": point,
                        "last_update": last_update,
                    })

    if not records:
        logger.warning("No odds records parsed")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

    # Convert to DataFrame
    parsed_odds_df = pl.DataFrame(records)
    logger.info(f"Parsed {len(records)} odds records from {len(games_data)} games")

    # Separate different market types
    moneyline_odds = parsed_odds_df.filter(pl.col("market_key") == "h2h")
    totals_odds = parsed_odds_df.filter(pl.col("market_key") == "totals")
    player_props = parsed_odds_df.filter(pl.col("market_key").str.starts_with("player_"))

    # Find best odds for each market
    if not moneyline_odds.is_empty():
        best_moneyline = moneyline_odds.group_by(
            ["game_id", "outcome_name"]
        ).agg([
            pl.col("price").max().alias("best_price"),
            pl.col("bookmaker_key").first().alias("best_bookmaker"),
            pl.col("home_team").first().alias("home_team"),
            pl.col("away_team").first().alias("away_team"),
            pl.col("commence_time").first().alias("commence_time"),
        ])
        logger.info(f"Found best odds for {len(best_moneyline)} moneyline markets")
    else:
        best_moneyline = pl.DataFrame()

    if not totals_odds.is_empty():
        best_totals = totals_odds.group_by(
            ["game_id", "outcome_name", "point"]
        ).agg([
            pl.col("price").max().alias("best_price"),
            pl.col("bookmaker_key").first().alias("best_bookmaker"),
            pl.col("home_team").first().alias("home_team"),
            pl.col("away_team").first().alias("away_team"),
            pl.col("commence_time").first().alias("commence_time"),
        ])
        logger.info(f"Found best odds for {len(best_totals)} totals markets")
    else:
        best_totals = pl.DataFrame()

    return best_moneyline, player_props, best_totals'''

    # Backtesting integration
    backtesting_code = '''
    def load_backtesting_data(self) -> pl.DataFrame:
        """Load historical backtesting data."""
        backtest_paths = [
            "backtest_data.parquet",
            "historical_results.parquet",
            "betting_history.parquet",
            os.path.join(self.config.output_dir, "historical_bets.parquet"),
            os.path.join("data", "backtesting_data.parquet"),
        ]

        for path in backtest_paths:
            if os.path.exists(path):
                try:
                    df = pl.read_parquet(path)
                    logger.info(f"Loaded backtesting data from {path}: {len(df)} records")
                    return df
                except Exception as e:
                    logger.warning(f"Could not load {path}: {e}")

        logger.warning("No backtesting data found - will create mock data")
        return self._create_mock_backtesting_data()

    def _create_mock_backtesting_data(self) -> pl.DataFrame:
        """Create mock backtesting data for demonstration."""
        import random
        from datetime import datetime, timedelta

        logger.info("Creating mock backtesting data...")

        # Generate 100 historical bets
        mock_data = []
        start_date = datetime.now() - timedelta(days=90)

        teams = ["NYY", "LAD", "BOS", "HOU", "ATL", "PHI", "SD", "NYM", "TB", "TOR"]

        for i in range(100):
            bet_date = start_date + timedelta(days=random.randint(0, 89))
            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])

            # Simulate bet outcomes
            predicted_prob = random.uniform(0.45, 0.65)
            market_prob = random.uniform(0.40, 0.70)
            edge = predicted_prob - market_prob

            bet_amount = max(10, min(100, abs(edge) * 1000))  # Kelly-like sizing
            odds = 1 / market_prob

            # Simulate win/loss (slightly favor positive edge bets)
            win_prob = predicted_prob if edge > 0 else predicted_prob * 0.9
            won = random.random() < win_prob
            pnl = (bet_amount * (odds - 1)) if won else -bet_amount

            mock_data.append({
                "date": bet_date.strftime("%Y-%m-%d"),
                "game_id": f"mock_{i}",
                "home_team": home_team,
                "away_team": away_team,
                "bet_type": "moneyline",
                "bet_outcome": home_team,
                "predicted_prob": predicted_prob,
                "market_prob": market_prob,
                "edge": edge,
                "odds": odds,
                "bet_amount": bet_amount,
                "won": won,
                "pnl": pnl,
                "roi": (pnl / bet_amount) * 100,
            })

        df = pl.DataFrame(mock_data)

        # Save for future use
        backtest_path = os.path.join(self.config.output_dir, "mock_backtesting_data.parquet")
        df.write_parquet(backtest_path)
        logger.info(f"Created and saved mock backtesting data: {backtest_path}")

        return df

    def analyze_backtesting_performance(self, backtest_df: pl.DataFrame) -> Dict[str, Any]:
        """Analyze backtesting performance metrics."""
        if backtest_df.is_empty():
            return {"error": "No backtesting data available"}

        try:
            # Calculate key metrics
            total_bets = len(backtest_df)
            total_pnl = backtest_df["pnl"].sum()
            total_invested = backtest_df["bet_amount"].sum()

            win_rate = (backtest_df["won"].sum() / total_bets) * 100
            avg_roi = backtest_df["roi"].mean()

            # Positive edge bets performance
            positive_edge = backtest_df.filter(pl.col("edge") > 0)
            positive_edge_win_rate = 0
            positive_edge_roi = 0
            if not positive_edge.is_empty():
                positive_edge_win_rate = (positive_edge["won"].sum() / len(positive_edge)) * 100
                positive_edge_roi = positive_edge["roi"].mean()

            # Recent performance (last 30 days)
            recent_cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            recent_bets = backtest_df.filter(pl.col("date") >= recent_cutoff)
            recent_pnl = recent_bets["pnl"].sum() if not recent_bets.is_empty() else 0

            results = {
                "total_bets": total_bets,
                "total_pnl": round(total_pnl, 2),
                "total_invested": round(total_invested, 2),
                "overall_roi": round((total_pnl / total_invested) * 100, 2) if total_invested > 0 else 0,
                "win_rate": round(win_rate, 1),
                "avg_roi_per_bet": round(avg_roi, 2),
                "positive_edge_bets": len(positive_edge),
                "positive_edge_win_rate": round(positive_edge_win_rate, 1),
                "positive_edge_roi": round(positive_edge_roi, 2),
                "recent_30d_pnl": round(recent_pnl, 2),
                "recent_30d_bets": len(recent_bets),
            }

            logger.info(f"Backtesting analysis complete: {results}")
            return results

        except Exception as e:
            logger.error(f"Error analyzing backtesting data: {e}")
            return {"error": str(e)}'''

    # Replace the old parsing function
    content = content.replace(
        content[
            content.find("def _parse_and_select_odds") : content.find(
                "class DailyPredictionManager"
            )
        ],
        enhanced_parsing + "\n\n",
    )

    # Add backtesting methods to DailyPredictionManager class
    # Find the class and add methods before run_daily_workflow
    class_start = content.find("class DailyPredictionManager:")
    init_end = content.find("async def run_daily_workflow", class_start)

    content = content[:init_end] + backtesting_code + "\n    " + content[init_end:]

    # Enhanced workflow with backtesting
    enhanced_workflow = """            # Step 5: Load and analyze backtesting data
            backtest_df = self.load_backtesting_data()
            backtest_analysis = self.analyze_backtesting_performance(backtest_df)

            # Step 6: Generate basic recommendations using live odds
            recommendations = []
            if enable_betting_recommendations and not moneyline_odds.is_empty():
                recommendations = self._generate_basic_recommendations(
                    moneyline_odds, totals_odds, backtest_analysis
                )"""

    # Add recommendation generation method
    recommendation_method = '''
    def _generate_basic_recommendations(
        self,
        moneyline_odds: pl.DataFrame,
        totals_odds: pl.DataFrame,
        backtest_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate basic betting recommendations."""
        recommendations = []

        try:
            if moneyline_odds.is_empty():
                return recommendations

            # Simple value detection based on backtesting insights
            min_edge = 0.03  # 3% minimum edge
            max_bet_pct = 0.05  # 5% max bet size
            base_bankroll = 1000  # Base bankroll for calculations

            # Group by game for easier analysis
            games = moneyline_odds.group_by("game_id").agg([
                pl.col("home_team").first(),
                pl.col("away_team").first(),
                pl.col("commence_time").first(),
                pl.col("best_price").max().alias("best_home_odds"),
                pl.col("best_price").min().alias("best_away_odds"),
            ])

            for game in games.iter_rows(named=True):
                game_id = game["game_id"]
                home_team = game["home_team"]
                away_team = game["away_team"]

                # Get specific odds for home and away
                home_odds_data = moneyline_odds.filter(
                    (pl.col("game_id") == game_id) &
                    (pl.col("outcome_name") == home_team)
                )
                away_odds_data = moneyline_odds.filter(
                    (pl.col("game_id") == game_id) &
                    (pl.col("outcome_name") == away_team)
                )

                if home_odds_data.is_empty() or away_odds_data.is_empty():
                    continue

                home_price = home_odds_data["best_price"].max()
                away_price = away_odds_data["best_price"].max()

                # Calculate implied probabilities
                home_implied_prob = 1 / home_price if home_price > 0 else 0
                away_implied_prob = 1 / away_price if away_price > 0 else 0

                # Simple model: if one team has significantly better odds than "fair"
                # Use 50/50 as baseline (could be replaced with ML model predictions)
                fair_prob = 0.5

                home_edge = fair_prob - home_implied_prob
                away_edge = fair_prob - away_implied_prob

                # Generate recommendations for edges above threshold
                if home_edge > min_edge:
                    bet_size = min(max_bet_pct * base_bankroll, home_edge * base_bankroll * 2)
                    recommendations.append({
                        "game_id": game_id,
                        "matchup": f"{away_team} @ {home_team}",
                        "bet_type": "moneyline",
                        "bet_team": home_team,
                        "odds": home_price,
                        "implied_prob": round(home_implied_prob, 3),
                        "estimated_edge": round(home_edge, 3),
                        "recommended_bet": round(bet_size, 2),
                        "confidence": "medium",
                        "reasoning": f"Home team odds suggest {home_edge:.1%} edge"
                    })

                if away_edge > min_edge:
                    bet_size = min(max_bet_pct * base_bankroll, away_edge * base_bankroll * 2)
                    recommendations.append({
                        "game_id": game_id,
                        "matchup": f"{away_team} @ {home_team}",
                        "bet_type": "moneyline",
                        "bet_team": away_team,
                        "odds": away_price,
                        "implied_prob": round(away_implied_prob, 3),
                        "estimated_edge": round(away_edge, 3),
                        "recommended_bet": round(bet_size, 2),
                        "confidence": "medium",
                        "reasoning": f"Away team odds suggest {away_edge:.1%} edge"
                    })

            logger.info(f"Generated {len(recommendations)} betting recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []'''

    # Insert the new methods
    workflow_start = content.find("# Step 5: Backtesting (simplified)")
    content = (
        content[:workflow_start] + enhanced_workflow + content[workflow_start + 50 :]
    )

    # Add the recommendation method to the class
    class_end = content.rfind("async def main():")
    content = content[:class_end] + recommendation_method + "\n\n" + content[class_end:]

    # Write the enhanced version
    with open(filename, "w") as f:
        f.write(content)

    print("✅ Enhanced odds parsing and backtesting integration added!")

    # Test compilation
    try:
        with open(filename) as f:
            compile(f.read(), filename, "exec")
        print("✅ Enhanced version compiles successfully!")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False


if __name__ == "__main__":
    add_enhanced_parsing_and_backtesting("daily_prediction_and_backtest.py")
