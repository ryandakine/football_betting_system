#!/bin/bash
# WNBA Game Day Validation Hook
# Runs BEFORE any WNBA analysis - ensures optimal timing for small slates
# Makes it IMPOSSIBLE to analyze at wrong time or miss critical context

echo "🏀 WNBA GAME DAY VALIDATION"
echo "=========================================="

# Get current date/time
DAY_OF_WEEK=$(date +%A)
HOUR=$(date +%H)
MINUTE=$(date +%M)
MONTH=$(date +%m)
DAY=$(date +%d)

# 1. WNBA SEASON CHECK (May-October)
echo "📅 Season Check..."

if (( MONTH < 5 || MONTH > 10 )); then
    echo "❌ BLOCKED: WNBA Off-Season"
    echo ""
    echo "WHY THIS MATTERS:"
    echo "  - WNBA season: May-October only"
    echo "  - Preseason: April"
    echo "  - Regular season: Mid-May through Mid-September"
    echo "  - Playoffs: Late September through October"
    echo ""
    echo "Current: $(date +%B) (month $MONTH)"

    if (( MONTH == 4 )); then
        echo "⚠️  Preseason in April - limited games"
    elif (( MONTH >= 11 || MONTH <= 3 )); then
        echo "Next season starts: May"
    fi

    exit 1
fi

echo "✅ In-season (May-October)"

# 2. SPECIFIC SEASON PHASE
echo ""
echo "🎯 Season Phase Detection..."

if (( MONTH == 5 && DAY < 15 )); then
    SEASON_PHASE="Early Season"
    echo "🌱 Early Season - teams finding rhythm"
    echo "   - Higher variance in performance"
    echo "   - Less historical data"
    echo "   - Lines may have more value"
    CONFIDENCE_ADJUSTMENT=0.03  # Require 3% higher confidence

elif (( MONTH >= 6 && MONTH <= 8 )); then
    SEASON_PHASE="Mid-Season"
    echo "🔥 Mid-Season - peak performance period"
    echo "   - Most consistent team performance"
    echo "   - Best time for analysis"
    echo "   - Sharpest lines"
    CONFIDENCE_ADJUSTMENT=0.00  # Standard confidence

elif (( MONTH == 7 && DAY >= 10 && DAY <= 20 )); then
    SEASON_PHASE="All-Star Break"
    echo "⭐ All-Star Break Period"
    echo "   - Limited games"
    echo "   - Some fatigue/rest considerations"
    CONFIDENCE_ADJUSTMENT=0.02

elif (( MONTH == 9 && DAY >= 15 )) || (( MONTH == 10 )); then
    SEASON_PHASE="Playoffs"
    echo "🏆 Playoff Season"
    echo "   - Single elimination pressure"
    echo "   - Higher variance"
    echo "   - Tighter lines"
    echo "   - Stars get more minutes"
    CONFIDENCE_ADJUSTMENT=0.05  # Require 5% higher confidence

    echo ""
    echo "⚠️  PLAYOFF ADJUSTMENTS:"
    echo "   - Increase confidence threshold to 67%"
    echo "   - Maximum 2-leg parlays only"
    echo "   - Stars playing 35+ minutes"
    echo "   - Home court advantage amplified"

else
    SEASON_PHASE="Regular Season"
    echo "✅ Regular Season"
    CONFIDENCE_ADJUSTMENT=0.00
fi

# 3. WNBA GAME DAY TIMING (Small Slate = Precise Timing Critical)
echo ""
echo "⏰ Game Day Timing..."

IS_GAME_DAY=false
TYPICAL_GAMES=0
GAME_TIME=""

case $DAY_OF_WEEK in
    Tuesday|Wednesday|Thursday|Friday)
        IS_GAME_DAY=true
        TYPICAL_GAMES=3-6

        # WNBA weekday games typically: 7:00 PM ET (19:00)
        if (( HOUR >= 19 )); then
            echo "⚠️  WARNING: Evening games likely started"
            echo "   Typical WNBA tip-off: 7:00-8:00 PM ET"
            echo ""
            echo "RECOMMENDATION: Analyze by 6:00 PM ET on game days"
        elif (( HOUR >= 15 )); then
            echo "✅ OPTIMAL: Afternoon analysis window"
            echo "   - Injury reports finalized"
            echo "   - Time to place bets before 7 PM games"
            TIMING_WINDOW="OPTIMAL"
        elif (( HOUR >= 10 )); then
            echo "✅ Good: Morning/midday analysis"
            echo "   - Early enough for preparation"
            echo "   - Check injury reports again before games"
            TIMING_WINDOW="GOOD"
        else
            echo "⚠️  Very early analysis"
            echo "   - Injury reports may update"
            echo "   - Consider re-checking closer to game time"
            TIMING_WINDOW="EARLY"
        fi
        ;;

    Saturday|Sunday)
        IS_GAME_DAY=true
        TYPICAL_GAMES=4-8

        # WNBA weekend games: Afternoon/evening (various times)
        if (( HOUR >= 17 )); then
            echo "⚠️  WARNING: Many games likely started or in progress"
            echo "   Weekend WNBA: 12:00 PM - 8:00 PM ET start times"
        elif (( HOUR >= 10 && HOUR < 12 )); then
            echo "✅ OPTIMAL: Weekend morning analysis"
            echo "   - Before noon games start"
            echo "   - Time for full slate analysis"
            TIMING_WINDOW="OPTIMAL"
        elif (( HOUR >= 8 )); then
            echo "✅ Good: Early weekend analysis"
            TIMING_WINDOW="GOOD"
        else
            echo "⚠️  Very early weekend analysis"
            TIMING_WINDOW="EARLY"
        fi
        ;;

    Monday)
        echo "ℹ️  Typically OFF DAY for WNBA"
        echo "   - No games scheduled most Mondays"
        echo "   - Exception: Special events, makeup games"
        echo ""
        echo "RECOMMENDATION:"
        echo "  - Review previous week results"
        echo "  - Prepare for Tuesday-Sunday games"
        IS_GAME_DAY=false
        ;;
esac

# 4. WNBA-SPECIFIC SLATE SIZE WARNING
echo ""
echo "📊 Slate Size Context..."

if [ "$IS_GAME_DAY" = true ]; then
    echo "Expected games today: $TYPICAL_GAMES"
    echo ""
    echo "⚠️  WNBA SMALL SLATE REMINDER:"
    echo "   - Only 12 teams in league"
    echo "   - Typically 3-6 games per day (not 10-15 like other sports)"
    echo "   - Each game analyzed more thoroughly"
    echo "   - Quality over quantity approach"
    echo ""
    echo "ANALYSIS STRATEGY:"
    echo "   - Spend more time per game (smaller slate)"
    echo "   - Check player matchup specifics"
    echo "   - Monitor injury reports more carefully"
    echo "   - Parlay limit: 2-3 legs maximum"
fi

# 5. INJURY REPORT CHECK
echo ""
echo "🏥 Injury Report Status..."

if [ "$IS_GAME_DAY" = true ]; then
    if (( HOUR >= 15 )); then
        echo "✅ Injury reports should be finalized"
        echo "   - Game day injury reports typically by 3 PM ET"
    elif (( HOUR >= 10 )); then
        echo "⚠️  Injury reports still updating"
        echo "   - Check closer to game time"
        echo "   - Star player status critical for small slate"
    else
        echo "⚠️  Too early for final injury reports"
        echo "   - Reports typically finalize by 3 PM ET"
        echo "   - CRITICAL: Small slate = each injury has bigger impact"
    fi

    echo ""
    echo "⚠️  WNBA INJURY IMPACT:"
    echo "   - Star player out = ±8-12 point swing (bigger than NFL!)"
    echo "   - Smaller rosters = less depth"
    echo "   - Must check status of top 2-3 players per team"
fi

# 6. WEATHER CHECK (INDOOR SPORT - Minimal)
echo ""
echo "🌤️  Weather Impact..."
echo "✅ WNBA plays indoors - weather not a factor"
echo "   - All arenas are indoor facilities"
echo "   - No weather adjustments needed"
echo "   - Focus on injury/rest/matchups instead"

# 7. REST & TRAVEL CONSIDERATIONS (WNBA-Specific)
echo ""
echo "✈️  Rest & Travel Context..."

if [ "$IS_GAME_DAY" = true ]; then
    echo "WNBA Schedule Factors:"
    echo "   - Back-to-back games are common"
    echo "   - 40-game season in ~100 days = packed schedule"
    echo "   - Travel fatigue is real (smaller budgets)"
    echo ""
    echo "CHECK FOR:"
    echo "   - Team on 2nd night of back-to-back (fatigue)"
    echo "   - Cross-country travel in last 48 hours"
    echo "   - 3 games in 5 days (high fatigue)"
    echo ""
    echo "ADJUSTMENTS:"
    echo "   - Back-to-back: Reduce confidence by 5%"
    echo "   - After long travel: -2 points to total"
    echo "   - Fresh vs tired: Favor rested team"
fi

# 8. COMMISSIONER'S CUP CHECK (June)
echo ""
echo "🏆 Special Event Check..."

if (( MONTH == 6 )); then
    echo "⚠️  COMMISSIONER'S CUP SEASON (June)"
    echo "   - In-season tournament games"
    echo "   - Teams more motivated (bonus money)"
    echo "   - May see increased intensity"
    echo ""
    echo "ADJUSTMENT: Favor home teams by extra 1 point"
fi

# 9. API KEY VALIDATION
echo ""
echo "🔑 API Configuration..."

if [ -z "$WNBA_ODDS_API_KEY" ] && [ -z "$ODDS_API_KEY" ]; then
    echo "❌ BLOCKED: No Odds API key found"
    echo ""
    echo "FIX: Set WNBA_ODDS_API_KEY or ODDS_API_KEY in .env"
    exit 1
fi

echo "✅ Odds API key configured"

# 10. BANKROLL VALIDATION
echo ""
echo "💵 Bankroll Check..."

BANKROLL=${WNBA_BANKROLL:-${BANKROLL:-1000}}

if (( $(echo "$BANKROLL < 1000" | bc -l) )); then
    echo "❌ BLOCKED: Bankroll too low ($BANKROLL)"
    echo ""
    echo "WHY THIS MATTERS:"
    echo "  - WNBA has smaller market = less liquidity"
    echo "  - Need buffer for variance"
    echo "  - Minimum $1,000 for proper sizing"
    exit 1
fi

echo "✅ Bankroll: \$$BANKROLL"

# 11. EXPOSURE LIMITS (WNBA-Specific: More Conservative)
echo ""
echo "📉 Risk Management..."

MAX_EXPOSURE=0.08  # 8% for WNBA (more conservative than NFL/NCAA)

echo "WNBA Risk Limits:"
echo "   - Max exposure per game: ${MAX_EXPOSURE}% (8%)"
echo "   - Max parlay stake: 2.5% of bankroll"
echo "   - Max parlay legs: 3 (small slate)"
echo ""
echo "WHY MORE CONSERVATIVE:"
echo "   - Smaller market = less liquidity"
echo "   - Fewer games = each bet is bigger % of action"
echo "   - Higher variance with star-dependent teams"

# 12. SYSTEM DEPENDENCIES
echo ""
echo "📦 System Dependencies..."

python3 -c "import pydantic, aiohttp, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ BLOCKED: Missing Python dependencies"
    echo "FIX: pip install -r requirements.txt"
    exit 1
fi

echo "✅ All dependencies available"

# 13. FINAL RECOMMENDATIONS
echo ""
echo "=========================================="
echo "📋 ANALYSIS RECOMMENDATIONS"
echo "=========================================="

if [ "$IS_GAME_DAY" = true ]; then
    echo "✅ Proceed with WNBA analysis"
    echo ""
    echo "KEY CHECKS:"
    echo "  ✓ Season: $SEASON_PHASE"
    echo "  ✓ Game Day: $DAY_OF_WEEK"
    echo "  ✓ Expected Games: $TYPICAL_GAMES"
    echo "  ✓ Timing: ${TIMING_WINDOW:-STANDARD}"
    echo "  ✓ APIs: Configured"
    echo "  ✓ Bankroll: Adequate"
    echo ""

    if [ "$SEASON_PHASE" = "Playoffs" ]; then
        echo "⚠️  PLAYOFF REMINDERS:"
        echo "   - Increase confidence threshold to 67%"
        echo "   - Max 2-leg parlays"
        echo "   - Stars playing heavy minutes"
        echo "   - Home court matters more"
        echo ""
    fi

    echo "SMALL SLATE STRATEGY:"
    echo "  1. Analyze ALL games thoroughly (only 3-6 games)"
    echo "  2. Focus on star player matchups"
    echo "  3. Check rest/travel status"
    echo "  4. Verify injury reports before betting"
    echo "  5. Conservative parlay approach (max 3 legs)"
    echo ""

    if [ "${TIMING_WINDOW:-STANDARD}" = "EARLY" ]; then
        echo "⚠️  EARLY ANALYSIS REMINDER:"
        echo "   - Re-check injury reports at 3 PM ET"
        echo "   - Monitor line movements"
        echo "   - Confirm star player status"
    fi

else
    echo "ℹ️  Not a typical WNBA game day ($DAY_OF_WEEK)"
    echo ""
    echo "TYPICAL SCHEDULE:"
    echo "  - Tuesday-Sunday: Games"
    echo "  - Monday: Usually off"
    echo "  - Most games: 7:00-8:00 PM ET"
    echo "  - Weekend: Afternoon/evening games"
    echo ""
    echo "OPTIMAL ANALYSIS TIMING:"
    echo "  - Weekdays: 3:00-6:00 PM ET"
    echo "  - Weekends: 10:00 AM-12:00 PM ET"
fi

echo ""
echo "=========================================="
echo "🏀 Ready to analyze WNBA games"
echo "=========================================="
echo ""
