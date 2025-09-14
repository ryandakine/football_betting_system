#!/usr/bin/env python3
"""
FanDuel Analysis Results Monitor
This script monitors and displays the results from your MLB betting analysis workflow.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path


def load_env_vars():
    """Load environment variables"""
    try:
        from dotenv import load_dotenv

        load_dotenv("aci.env")
    except ImportError:
        pass


def get_supabase_client():
    """Get Supabase client"""
    try:
        from supabase import Client, create_client

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            print("❌ Supabase credentials not configured")
            return None

        return create_client(supabase_url, supabase_key)
    except ImportError:
        print("❌ Supabase client not available - install with: pip install supabase")
        return None
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        return None


def get_latest_analysis(supabase):
    """Get the latest analysis results"""
    try:
        result = (
            supabase.table("fanduel_betting_analysis")
            .select("*")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if result.data:
            return result.data[0]
        else:
            return None
    except Exception as e:
        print(f"❌ Error fetching latest analysis: {e}")
        return None


def get_daily_summary(supabase, date=None):
    """Get daily summary statistics"""
    if not date:
        date = datetime.now().date()

    try:
        result = (
            supabase.table("daily_summary_stats")
            .select("*")
            .eq("analysis_date", date.isoformat())
            .execute()
        )

        if result.data:
            return result.data[0]
        else:
            return None
    except Exception as e:
        print(f"❌ Error fetching daily summary: {e}")
        return None


def get_recent_tracking_data(supabase, days=7):
    """Get recent tracking data"""
    try:
        start_date = (datetime.now() - timedelta(days=days)).date()

        result = (
            supabase.table("fanduel_tracking_history")
            .select("*")
            .gte("analysis_date", start_date.isoformat())
            .order("created_at", desc=True)
            .execute()
        )

        return result.data
    except Exception as e:
        print(f"❌ Error fetching tracking data: {e}")
        return []


def display_latest_analysis(analysis):
    """Display the latest analysis results"""
    print("\n📊 Latest Analysis Results")
    print("=" * 50)

    if not analysis:
        print("❌ No analysis data found")
        return

    # Parse the JSON data
    games_data = analysis.get("games_data", [])
    summary_data = analysis.get("summary_data", {})
    ai_analysis = analysis.get("ai_analysis", "No AI analysis available")

    print(f"📅 Analysis Date: {analysis.get('created_at', 'Unknown')}")
    print(f"🎮 Total Games: {summary_data.get('totalGames', 0)}")
    print(
        f"💰 Total Money Left on Table: ${summary_data.get('totalMoneyLeftOnTable', 0):.2f}"
    )
    print(
        f"📈 Average Money Left on Table: ${summary_data.get('averageMoneyLeftOnTable', 0):.2f}"
    )

    print("\n🎯 Top Value Opportunities:")
    if games_data:
        # Sort games by money left on table
        sorted_games = sorted(
            games_data,
            key=lambda x: x.get("moneyLeftOnTable", {}).get("homeTeam", 0)
            + x.get("moneyLeftOnTable", {}).get("awayTeam", 0),
            reverse=True,
        )

        for i, game in enumerate(sorted_games[:5], 1):
            home_team = game.get("homeTeam", "Unknown")
            away_team = game.get("awayTeam", "Unknown")
            money_left = game.get("moneyLeftOnTable", {}).get("homeTeam", 0) + game.get(
                "moneyLeftOnTable", {}
            ).get("awayTeam", 0)

            if money_left > 0:
                print(
                    f"  {i}. {away_team} @ {home_team}: ${money_left:.2f} left on table"
                )

    print(f"\n🤖 AI Analysis:")
    print(
        f"   {ai_analysis[:200]}..." if len(ai_analysis) > 200 else f"   {ai_analysis}"
    )


def display_daily_summary(summary):
    """Display daily summary statistics"""
    print("\n📈 Daily Summary Statistics")
    print("=" * 50)

    if not summary:
        print("❌ No daily summary data found")
        return

    print(f"📅 Date: {summary.get('analysis_date', 'Unknown')}")
    print(f"🎮 Total Games: {summary.get('totalGames', 0)}")
    print(
        f"💰 Total Money Left on Table: ${summary.get('totalMoneyLeftOnTable', 0):.2f}"
    )
    print(
        f"📊 Average Money Left on Table: ${summary.get('averageMoneyLeftOnTable', 0):.2f}"
    )
    print(
        f"🎯 Games with Value Opportunities: {summary.get('games_with_value_opportunities', 0)}"
    )
    print(
        f"📊 Percentage with Opportunities: {summary.get('percentage_games_with_opportunities', 0):.1f}%"
    )
    print(f"🏆 Best Performing Book: {summary.get('best_performing_book', 'Unknown')}")
    print(
        f"⚠️  Worst Performing Book: {summary.get('worst_performing_book', 'Unknown')}"
    )


def display_trends(tracking_data):
    """Display trends from tracking data"""
    print("\n📈 Recent Trends (Last 7 Days)")
    print("=" * 50)

    if not tracking_data:
        print("❌ No tracking data found")
        return

    # Calculate trends
    total_money_left = []
    total_games = []

    for entry in tracking_data:
        tracking_info = entry.get("tracking_data", {})
        summary = tracking_info.get("summary", {})

        total_money_left.append(summary.get("totalMoneyLeftOnTable", 0))
        total_games.append(summary.get("totalGames", 0))

    if total_money_left:
        avg_money_left = sum(total_money_left) / len(total_money_left)
        avg_games = sum(total_games) / len(total_games)

        print(f"📊 Average Daily Money Left on Table: ${avg_money_left:.2f}")
        print(f"🎮 Average Daily Games Analyzed: {avg_games:.1f}")

        if len(total_money_left) > 1:
            trend = (
                "📈 Increasing"
                if total_money_left[0] > total_money_left[-1]
                else "📉 Decreasing"
            )
            print(f"📈 Trend: {trend}")

    print(f"📅 Data Points: {len(tracking_data)} days")


def display_recommendations(analysis, summary):
    """Display recommendations based on the data"""
    print("\n💡 Recommendations")
    print("=" * 50)

    if not analysis and not summary:
        print("❌ No data available for recommendations")
        return

    recommendations = []

    # Get key metrics
    total_money_left = 0
    if analysis:
        summary_data = analysis.get("summary_data", {})
        total_money_left = summary_data.get("totalMoneyLeftOnTable", 0)

    if summary:
        total_money_left = summary.get("totalMoneyLeftOnTable", total_money_left)
        percentage_opportunities = summary.get("percentage_games_with_opportunities", 0)

        if percentage_opportunities > 50:
            recommendations.append(
                "🔍 Consider multi-book approach - high percentage of games have better odds elsewhere"
            )
        elif percentage_opportunities < 20:
            recommendations.append(
                "✅ FanDuel provides good value - stick with single book approach"
            )

    if total_money_left > 50:
        recommendations.append(
            "💰 Significant money left on table - consider shopping around for better odds"
        )
    elif total_money_left < 10:
        recommendations.append(
            "✅ Minimal value lost - FanDuel offers competitive odds"
        )

    if not recommendations:
        recommendations.append(
            "📊 Continue monitoring - not enough data for specific recommendations"
        )

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")


def main():
    """Main function"""
    print("⚾ MLB FanDuel Analysis - Results Monitor")
    print("=" * 50)

    # Load environment variables
    load_env_vars()

    # Get Supabase client
    supabase = get_supabase_client()
    if not supabase:
        print(
            "❌ Cannot connect to database. Please check your Supabase configuration."
        )
        return

    # Get data
    latest_analysis = get_latest_analysis(supabase)
    daily_summary = get_daily_summary(supabase)
    tracking_data = get_recent_tracking_data(supabase)

    # Display results
    display_latest_analysis(latest_analysis)
    display_daily_summary(daily_summary)
    display_trends(tracking_data)
    display_recommendations(latest_analysis, daily_summary)

    print("\n🎯 Next Steps:")
    print("1. Run your n8n workflow to get fresh data")
    print("2. Check the recommendations above")
    print("3. Monitor trends over time")
    print("4. Adjust your betting strategy based on the insights")


if __name__ == "__main__":
    main()
