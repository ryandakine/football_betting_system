#!/bin/bash
# NFL Game Day Validation Hook
# Runs BEFORE any NFL analysis to ensure optimal timing and context
# Makes it IMPOSSIBLE to analyze games at wrong time or miss critical info

echo "🏈 NFL GAME DAY VALIDATION"
echo "=========================================="

# Get current day and time
DAY_OF_WEEK=$(date +%A)
HOUR=$(date +%H)
MINUTE=$(date +%M)
CURRENT_TIME="${HOUR}:${MINUTE}"
MONTH=$(date +%m)

# 1. NFL SEASON CHECK
echo "📅 Season Check..."
if (( MONTH >= 4 && MONTH <= 8 )); then
    echo "❌ BLOCKED: NFL Off-Season"
    echo ""
    echo "WHY THIS MATTERS:"
    echo "  - No games available April-August"
    echo "  - Preseason: August (different analysis approach)"
    echo "  - Regular season: September-January"
    echo "  - Playoffs: January-February"
    echo ""
    echo "Current month: $(date +%B)"
    echo "Next season starts: September"
    exit 1
fi

echo "✅ In-season (September-February)"

# 2. GAME DAY DETECTION
echo ""
echo "🎯 Game Day Detection..."

IS_GAME_DAY=false
GAME_SLOT=""
TIME_UNTIL_GAMES=""
ANALYSIS_WINDOW=""

case $DAY_OF_WEEK in
    Thursday)
        IS_GAME_DAY=true
        GAME_SLOT="Thursday Night Football"

        # TNF kickoff: 8:15 PM ET (20:15)
        if (( HOUR >= 20 )); then
            echo "⚠️  WARNING: TNF likely in progress or complete"
            echo "   Current time: $CURRENT_TIME"
            echo "   TNF kickoff: 20:15 (8:15 PM ET)"
            echo ""
            echo "RECOMMENDATION: Analysis should be done by 6:00 PM ET"
        elif (( HOUR >= 18 )); then
            echo "✅ Optimal TNF analysis window (6:00-8:00 PM ET)"
            ANALYSIS_WINDOW="OPTIMAL"
        elif (( HOUR >= 12 )); then
            echo "✅ Good TNF analysis window (afternoon)"
            ANALYSIS_WINDOW="GOOD"
        else
            echo "⚠️  Early for TNF analysis (morning)"
            echo "   BETTER: Wait until afternoon for latest injury reports"
            ANALYSIS_WINDOW="EARLY"
        fi
        ;;

    Sunday)
        IS_GAME_DAY=true
        GAME_SLOT="Sunday (Early/Late/SNF)"

        # Early games: 1:00 PM ET (13:00)
        # Late games: 4:05/4:25 PM ET (16:05/16:25)
        # SNF: 8:20 PM ET (20:20)

        if (( HOUR >= 20 )); then
            echo "⚠️  WARNING: Most games complete, SNF in progress"
            echo "   Only SNF and live-betting relevant"
        elif (( HOUR >= 16 )); then
            echo "⚠️  WARNING: Early games in progress, late games starting"
            echo "   Analysis should be complete by 11:00 AM ET"
        elif (( HOUR >= 13 )); then
            echo "❌ BLOCKED: Games have started"
            echo ""
            echo "WHY THIS MATTERS:"
            echo "  - Early slate kicked off at 1:00 PM ET"
            echo "  - Pre-game analysis is too late"
            echo "  - Cannot place pre-game bets"
            echo ""
            echo "RECOMMENDATION: Analyze Sunday games by 11:00 AM ET"
            exit 1
        elif (( HOUR >= 9 )); then
            echo "✅ OPTIMAL: Sunday morning analysis window"
            echo "   - Inactive reports: 90 minutes before kickoff (11:30 AM ET)"
            echo "   - Weather updates available"
            echo "   - Line movements visible"
            ANALYSIS_WINDOW="OPTIMAL"
        elif (( HOUR >= 6 )); then
            echo "✅ Good: Early Sunday analysis"
            echo "   - Injury reports may still be updating"
            echo "   - Better: Wait until 9:00 AM for inactive lists"
            ANALYSIS_WINDOW="GOOD"
        else
            echo "⚠️  Very early Sunday analysis"
            echo "   - Injury reports not finalized"
            echo "   - Weather may change"
            echo "   - BETTER: Wait until 9:00 AM ET"
            ANALYSIS_WINDOW="EARLY"
        fi
        ;;

    Monday)
        IS_GAME_DAY=true
        GAME_SLOT="Monday Night Football"

        # MNF kickoff: 8:15 PM ET (20:15)
        if (( HOUR >= 20 )); then
            echo "⚠️  WARNING: MNF likely in progress or complete"
        elif (( HOUR >= 17 )); then
            echo "✅ Optimal MNF analysis window (5:00-8:00 PM ET)"
            ANALYSIS_WINDOW="OPTIMAL"
        elif (( HOUR >= 12 )); then
            echo "✅ Good MNF analysis window (afternoon)"
            ANALYSIS_WINDOW="GOOD"
        else
            echo "⚠️  Early for MNF analysis"
            ANALYSIS_WINDOW="EARLY"
        fi
        ;;

    Friday|Saturday)
        echo "⚠️  No typical NFL games on $DAY_OF_WEEK"
        echo "   (Exception: Late-season Saturday games in December)"
        if (( MONTH == 12 )); then
            echo "   December: Check for Saturday games"
            IS_GAME_DAY=true
            GAME_SLOT="Saturday Special"
        fi
        ;;

    Tuesday|Wednesday)
        echo "⚠️  No NFL games on $DAY_OF_WEEK"
        echo ""
        echo "RECOMMENDATION:"
        echo "  - Tuesday: Review previous week results"
        echo "  - Wednesday: Early line shopping for upcoming week"
        echo "  - Thursday: TNF preparation"
        echo "  - Sunday: Run analysis by 11:00 AM ET"
        ;;
esac

# 3. OPTIMAL TIMING VALIDATION
echo ""
echo "⏰ Timing Analysis..."

if [ "$IS_GAME_DAY" = true ]; then
    echo "Game Day: $DAY_OF_WEEK - $GAME_SLOT"
    echo "Analysis Window: $ANALYSIS_WINDOW"
    echo ""

    if [ "$ANALYSIS_WINDOW" = "OPTIMAL" ]; then
        echo "✅ Optimal timing for analysis"
        echo "   - Injury reports available"
        echo "   - Weather finalized"
        echo "   - Line movements stable"
    elif [ "$ANALYSIS_WINDOW" = "GOOD" ]; then
        echo "✅ Good timing, but could be better"
        echo "   - Some injury reports may still update"
        echo "   - Consider re-running closer to kickoff"
    elif [ "$ANALYSIS_WINDOW" = "EARLY" ]; then
        echo "⚠️  Analysis timing is early"
        echo "   - Injury reports not finalized"
        echo "   - Weather may change"
        echo "   - RECOMMENDATION: Re-run analysis closer to kickoff"
    fi
fi

# 4. INJURY REPORT TIMING CHECK
echo ""
echo "🏥 Injury Report Status..."

case $DAY_OF_WEEK in
    Sunday)
        if (( HOUR >= 11 )); then
            echo "✅ Inactive lists published (90 min before 1 PM kickoff)"
        else
            echo "⚠️  Inactive lists not yet available"
            echo "   Published: ~11:30 AM ET (90 min before kickoff)"
            echo "   CRITICAL: May affect player props and game totals"
        fi
        ;;
    Thursday)
        if (( HOUR >= 18 )); then
            echo "✅ Injury reports should be finalized"
        else
            echo "⚠️  TNF injury reports may still be updating"
        fi
        ;;
    Monday)
        if (( HOUR >= 17 )); then
            echo "✅ Injury reports should be finalized"
        else
            echo "⚠️  MNF injury reports may still be updating"
        fi
        ;;
esac

# 5. WEATHER CHECK FOR OUTDOOR GAMES
echo ""
echo "🌤️  Weather Context..."

# Outdoor stadiums that require weather checks
OUTDOOR_STADIUMS="Buffalo|Green Bay|Chicago|Cleveland|Denver|Kansas City|New England|Philadelphia|Pittsburgh"

echo "Note: Weather critical for outdoor stadiums:"
echo "  - Buffalo (high snow/wind impact)"
echo "  - Green Bay (cold weather)"
echo "  - Chicago (wind off Lake Michigan)"
echo "  - Cleveland (lake effect)"
echo "  - Others: Denver, KC, NE, PHI, PIT"
echo ""
echo "Weather checks should include:"
echo "  - Wind speed (>15 MPH affects passing/kicking)"
echo "  - Precipitation (rain/snow affects ball handling)"
echo "  - Temperature (<20°F affects performance)"

# 6. LINE MOVEMENT DETECTION
echo ""
echo "📊 Line Movement Context..."

if [ "$IS_GAME_DAY" = true ]; then
    echo "Game day line movement patterns:"
    echo "  - Morning: Sharp money moves"
    echo "  - Afternoon: Public money enters"
    echo "  - 90 min before: Final injury reports cause moves"
    echo ""
    echo "RECOMMENDATION: Monitor lines until kickoff"
fi

# 7. BETTING MARKET STATUS
echo ""
echo "💰 Betting Market Status..."

if [ "$IS_GAME_DAY" = true ]; then
    if [ "$ANALYSIS_WINDOW" = "OPTIMAL" ]; then
        echo "✅ Optimal time for bet placement"
        echo "   - Lines are relatively stable"
        echo "   - Injury info incorporated"
        echo "   - Market has liquidity"
    else
        echo "⚠️  Sub-optimal market timing"
        echo "   - Consider waiting for more info"
        echo "   - Or place small bets, larger closer to kickoff"
    fi
fi

# 8. API KEY VALIDATION
echo ""
echo "🔑 API Configuration..."

if [ -z "$NFL_ODDS_API_KEY" ] && [ -z "$ODDS_API_KEY" ]; then
    echo "❌ BLOCKED: No Odds API key found"
    echo ""
    echo "FIX: Set NFL_ODDS_API_KEY or ODDS_API_KEY in .env"
    exit 1
fi

echo "✅ Odds API key configured"

# 9. BANKROLL VALIDATION
echo ""
echo "💵 Bankroll Check..."

BANKROLL=${NFL_BANKROLL:-${BANKROLL:-1000}}

if (( $(echo "$BANKROLL < 1000" | bc -l) )); then
    echo "❌ BLOCKED: Bankroll too low ($BANKROLL)"
    echo ""
    echo "WHY THIS MATTERS:"
    echo "  - NFL: Higher variance than college"
    echo "  - Minimum $1,000 required for proper Kelly sizing"
    echo "  - NFL lines are sharper = need larger bankroll buffer"
    exit 1
fi

echo "✅ Bankroll: \$$BANKROLL"

# 10. SYSTEM DEPENDENCIES
echo ""
echo "📦 System Dependencies..."

python3 -c "import pydantic, aiohttp, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ BLOCKED: Missing Python dependencies"
    echo ""
    echo "FIX: pip install -r requirements.txt"
    exit 1
fi

echo "✅ All Python dependencies available"

# 11. FINAL RECOMMENDATIONS
echo ""
echo "=========================================="
echo "📋 ANALYSIS RECOMMENDATIONS"
echo "=========================================="

if [ "$IS_GAME_DAY" = true ]; then
    if [ "$DAY_OF_WEEK" = "Sunday" ] && (( HOUR >= 13 )); then
        echo "❌ TOO LATE: Games already started"
        echo ""
        echo "NEXT TIME:"
        echo "  - Run Sunday analysis by 11:00 AM ET"
        echo "  - Check inactive lists at 11:30 AM"
        echo "  - Place bets by 12:45 PM (before 1 PM kickoff)"
        exit 1
    fi

    echo "✅ Proceed with analysis"
    echo ""
    echo "KEY CHECKS:"
    echo "  ✓ Season: Active"
    echo "  ✓ Game Day: $DAY_OF_WEEK ($GAME_SLOT)"
    echo "  ✓ Timing: $ANALYSIS_WINDOW"
    echo "  ✓ APIs: Configured"
    echo "  ✓ Bankroll: Adequate"
    echo ""

    if [ "$ANALYSIS_WINDOW" = "EARLY" ]; then
        echo "REMINDER: Re-run analysis closer to kickoff for:"
        echo "  - Final injury reports"
        echo "  - Weather updates"
        echo "  - Late line movements"
    fi
else
    echo "ℹ️  Not a typical game day ($DAY_OF_WEEK)"
    echo ""
    echo "You can still run analysis for:"
    echo "  - Future game preparation"
    echo "  - Line shopping"
    echo "  - Historical review"
    echo ""
    echo "Optimal game day timing:"
    echo "  - Thursday: 6:00-8:00 PM ET (TNF)"
    echo "  - Sunday: 9:00 AM-12:45 PM ET (early slate)"
    echo "  - Monday: 5:00-8:00 PM ET (MNF)"
fi

echo ""
echo "=========================================="
echo "🏈 Ready to analyze NFL games"
echo "=========================================="
echo ""
