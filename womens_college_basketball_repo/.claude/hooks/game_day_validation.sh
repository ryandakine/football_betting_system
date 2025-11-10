#!/bin/bash
# Women's College Basketball Game Day Validation Hook
# Runs BEFORE any WCBB analysis - handles large slates and tournament timing
# Makes it IMPOSSIBLE to miss critical timing windows

echo "🏀 WOMEN'S COLLEGE BASKETBALL VALIDATION"
echo "=========================================="

# Get current date/time
DAY_OF_WEEK=$(date +%A)
HOUR=$(date +%H)
MONTH=$(date +%m)
DAY=$(date +%d)

# 1. WCBB SEASON CHECK (November-March)
echo "📅 Season Check..."

if (( MONTH >= 4 && MONTH <= 10 )); then
    echo "❌ BLOCKED: Women's College Basketball Off-Season"
    echo ""
    echo "WHY THIS MATTERS:"
    echo "  - WCBB season: November-March"
    echo "  - Preseason: October (exhibitions)"
    echo "  - Regular season: November-February"
    echo "  - Conference tournaments: Early March"
    echo "  - NCAA Tournament: Mid-late March"
    echo ""
    echo "Current: $(date +%B)"
    echo "Next season starts: November"
    exit 1
fi

echo "✅ In-season (November-March)"

# 2. SEASON PHASE DETECTION
echo ""
echo "🎯 Season Phase..."

if (( MONTH == 11 )) || (( MONTH == 12 && DAY <= 22 )); then
    SEASON_PHASE="Early Season"
    echo "🌱 Early Season (Non-Conference)"
    echo "   - Teams establishing identity"
    echo "   - More variance in performance"
    echo "   - Some cupcake games (avoid)"
    echo "   - Mid-majors can surprise power teams"
    CONFIDENCE_ADJUSTMENT=0.02  # +2% confidence required

elif (( MONTH == 12 && DAY > 22 )) || (( MONTH == 1 )) || (( MONTH == 2 )); then
    SEASON_PHASE="Conference Play"
    echo "🔥 Conference Play"
    echo "   - Most predictable period"
    echo "   - Best data availability"
    echo "   - Rivalry games (emotional factor)"
    echo "   - Optimal betting conditions"
    CONFIDENCE_ADJUSTMENT=0.00  # Standard confidence

elif (( MONTH == 3 && DAY <= 10 )); then
    SEASON_PHASE="Conference Tournaments"
    echo "🏆 Conference Tournament Week"
    echo "   - Single elimination"
    echo "   - NCAA berths on the line"
    echo "   - Upsets more common"
    echo "   - Teams with nothing to lose dangerous"
    CONFIDENCE_ADJUSTMENT=0.03  # +3% confidence

    echo ""
    echo "⚠️  TOURNAMENT ADJUSTMENTS:"
    echo "   - Cinderella alert: Underdogs motivated"
    echo "   - Fatigue: Teams may play 3-4 games in 4 days"
    echo "   - Desperation: Bubble teams playing for NCAA bid"
    echo "   - Max 2-leg parlays (higher correlation)"

elif (( MONTH == 3 && DAY > 10 )); then
    SEASON_PHASE="NCAA Tournament"
    echo "🏆🏆🏆 NCAA TOURNAMENT (March Madness)"
    echo "   - HIGHEST VARIANCE of entire year"
    echo "   - Bracket psychology matters"
    echo "   - Upsets are expected (12-5, 11-6 seeds)"
    echo "   - Single elimination = extreme pressure"
    CONFIDENCE_ADJUSTMENT=0.05  # +5% confidence

    echo ""
    echo "⚠️  MARCH MADNESS RULES:"
    echo "   - Increase confidence to 63%"
    echo "   - Max 2-leg parlays ONLY"
    echo "   - Avoid 1-16, 2-15 games (trap)"
    echo "   - Focus on 5-12, 6-11, 7-10 matchups"
    echo "   - Check tournament experience of coaches"

else
    SEASON_PHASE="Regular Season"
    echo "✅ Regular Season"
    CONFIDENCE_ADJUSTMENT=0.00
fi

# 3. GAME DAY DETECTION (WCBB heavy on weekends)
echo ""
echo "⏰ Game Day Timing..."

IS_GAME_DAY=false
EXPECTED_GAMES=0

case $DAY_OF_WEEK in
    Saturday|Sunday)
        IS_GAME_DAY=true
        EXPECTED_GAMES="50-100+ games"

        # Weekend games: Noon-8 PM ET (staggered)
        if (( HOUR >= 18 )); then
            echo "⚠️  WARNING: Most games in progress or complete"
            echo "   Weekend WCBB: 12:00 PM - 8:00 PM ET starts"
            echo ""
            echo "NEXT TIME: Analyze by 11:00 AM ET on weekends"
        elif (( HOUR >= 9 && HOUR < 12 )); then
            echo "✅ OPTIMAL: Weekend morning analysis window"
            echo "   - Before noon games start"
            echo "   - Time for full slate analysis"
            echo "   - Can prioritize best opportunities"
            TIMING_WINDOW="OPTIMAL"
        elif (( HOUR >= 7 )); then
            echo "✅ Good: Early weekend morning"
            echo "   - Injury reports should be available"
            TIMING_WINDOW="GOOD"
        else
            echo "⚠️  Very early weekend analysis"
            echo "   - Some injury reports may update"
            TIMING_WINDOW="EARLY"
        fi
        ;;

    Thursday|Friday)
        IS_GAME_DAY=true
        EXPECTED_GAMES="20-40 games"

        echo "✅ Weeknight games"
        echo "   - Smaller slate than weekend"
        echo "   - Typical start: 6:00-9:00 PM ET"

        if (( HOUR >= 18 )); then
            echo "⚠️  Evening - some games may have started"
        elif (( HOUR >= 15 )); then
            echo "✅ OPTIMAL: Afternoon analysis window"
            TIMING_WINDOW="OPTIMAL"
        elif (( HOUR >= 10 )); then
            echo "✅ Good: Mid-day analysis"
            TIMING_WINDOW="GOOD"
        else
            echo "⚠️  Early analysis"
            TIMING_WINDOW="EARLY"
        fi
        ;;

    Monday|Tuesday|Wednesday)
        echo "ℹ️  Fewer games on $DAY_OF_WEEK"
        echo "   - Some conference games"
        echo "   - Typically 5-15 games"
        EXPECTED_GAMES="5-15 games"
        IS_GAME_DAY=true
        ;;
esac

# 4. SLATE SIZE CONTEXT
echo ""
echo "📊 Expected Slate Size..."

if [ "$IS_GAME_DAY" = true ]; then
    echo "Expected games today: $EXPECTED_GAMES"
    echo ""

    if [[ "$DAY_OF_WEEK" == "Saturday" ]] || [[ "$DAY_OF_WEEK" == "Sunday" ]]; then
        echo "⚠️  LARGE SLATE STRATEGY:"
        echo "   - 50-100+ games available"
        echo "   - CANNOT analyze all games"
        echo "   - MUST prioritize by edge potential"
        echo "   - Focus on top 20-30 games"
        echo "   - Use conference weighting"
        echo ""
        echo "PRIORITIZATION ORDER:"
        echo "   1. Power conferences (Big Ten, SEC, ACC, Pac-12)"
        echo "   2. High edge potential games"
        echo "   3. Conference rivalries"
        echo "   4. Top 25 matchups"
        echo "   5. Mid-major value opportunities"
    else
        echo "✅ Manageable slate size"
        echo "   - Can analyze most/all games"
        echo "   - Still prioritize by edge potential"
    fi
fi

# 5. CONFERENCE CONTEXT
echo ""
echo "🏫 Conference Landscape..."

echo "Power Conferences (highest priority):"
echo "   - Big Ten (weight: 0.95)"
echo "   - SEC (weight: 0.92)"
echo "   - ACC (weight: 0.88)"
echo "   - Pac-12 (weight: 0.85)"
echo "   - Big 12 (weight: 0.82)"
echo "   - Big East (weight: 0.80)"
echo ""
echo "Mid-Major Opportunities:"
echo "   - WCC, Atlantic 10, American (weight: 0.55-0.60)"
echo "   - Hidden value in less-watched games"
echo "   - Market inefficiencies"

# 6. INJURY/AVAILABILITY CHECK
echo ""
echo "🏥 Injury Report Considerations..."

if [ "$SEASON_PHASE" = "NCAA Tournament" ]; then
    echo "⚠️  TOURNAMENT: Injury reports critical"
    echo "   - Single elimination = stars play through pain"
    echo "   - Check warm-up reports 30 min before tip"
    echo "   - Bench depth matters more"
elif [ "$IS_GAME_DAY" = true ]; then
    echo "Standard injury report process:"
    echo "   - Check team injury reports (usually day-before)"
    echo "   - Monitor social media for updates"
    echo "   - Verify starting lineups 1 hour before tip"
fi

# 7. CUPCAKE GAME WARNING (Early Season)
echo ""
echo "⚠️  Cupcake Game Detection..."

if [ "$SEASON_PHASE" = "Early Season" ]; then
    echo "WATCH FOR:"
    echo "   - Power conference vs low-major (avoid)"
    echo "   - Spreads > 30 points (no value)"
    echo "   - 'Buy games' (smaller schools paid to lose)"
    echo ""
    echo "RULE: Skip games with spread > 25 points"
    echo "      No value, high variance, potential backdoor"
fi

# 8. API KEY VALIDATION
echo ""
echo "🔑 API Configuration..."

if [ -z "$WCBB_ODDS_API_KEY" ] && [ -z "$ODDS_API_KEY" ]; then
    echo "❌ BLOCKED: No Odds API key found"
    echo "FIX: Set WCBB_ODDS_API_KEY or ODDS_API_KEY in .env"
    exit 1
fi

echo "✅ Odds API key configured"
echo "   Sport key: basketball_ncaaw"

# 9. BANKROLL VALIDATION
echo ""
echo "💵 Bankroll Check..."

BANKROLL=${WCBB_BANKROLL:-${BANKROLL:-1000}}

if (( $(echo "$BANKROLL < 1000" | bc -l) )); then
    echo "❌ BLOCKED: Bankroll too low ($BANKROLL)"
    echo ""
    echo "WHY THIS MATTERS:"
    echo "  - WCBB can have large slates (50-100 games)"
    echo "  - Need proper bankroll for Kelly sizing"
    echo "  - Minimum $1,000 for diversification"
    exit 1
fi

echo "✅ Bankroll: \$$BANKROLL"

# 10. RISK MANAGEMENT (WCBB-Specific)
echo ""
echo "📉 Risk Management..."

MAX_EXPOSURE=0.10  # 10% for WCBB

echo "WCBB Risk Limits:"
echo "   - Max exposure per game: 10%"
echo "   - Max parlay stake: 3% of bankroll"
echo "   - Max parlay legs: 4 (2 in tournaments)"
echo "   - Min edge: 5%"
echo "   - Min confidence: 58% (63% in March)"

if [ "$SEASON_PHASE" = "NCAA Tournament" ]; then
    echo ""
    echo "⚠️  MARCH MADNESS RISK ADJUSTMENT:"
    echo "   - Reduce max exposure to 8%"
    echo "   - Max 2-leg parlays only"
    echo "   - Increase edge threshold to 6%"
fi

# 11. SYSTEM DEPENDENCIES
echo ""
echo "📦 System Dependencies..."

python3 -c "import pydantic, aiohttp, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ BLOCKED: Missing Python dependencies"
    echo "FIX: pip install -r requirements.txt"
    exit 1
fi

echo "✅ All dependencies available"

# 12. GAME PRIORITIZATION CHECK
echo ""
echo "🎯 Game Prioritization..."

if [[ "$DAY_OF_WEEK" == "Saturday" ]] || [[ "$DAY_OF_WEEK" == "Sunday" ]]; then
    echo "⚠️  LARGE SLATE DETECTED"
    echo ""
    echo "MANDATORY: Use game prioritization"
    echo "   python3 -c 'from game_prioritization import GamePrioritizer; print(\"Prioritizer available\")'"

    python3 -c "from game_prioritization import GamePrioritizer" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "❌ WARNING: Prioritizer not available"
        echo "   Will analyze games in random order (suboptimal)"
    else
        echo "✅ Prioritizer available"
    fi
fi

# 13. FINAL RECOMMENDATIONS
echo ""
echo "=========================================="
echo "📋 ANALYSIS RECOMMENDATIONS"
echo "=========================================="

if [ "$IS_GAME_DAY" = true ]; then
    echo "✅ Proceed with WCBB analysis"
    echo ""
    echo "KEY CHECKS:"
    echo "  ✓ Season: $SEASON_PHASE"
    echo "  ✓ Game Day: $DAY_OF_WEEK"
    echo "  ✓ Expected Games: $EXPECTED_GAMES"
    echo "  ✓ Timing: ${TIMING_WINDOW:-STANDARD}"
    echo "  ✓ APIs: Configured"
    echo "  ✓ Bankroll: Adequate"
    echo ""

    if [[ "$DAY_OF_WEEK" == "Saturday" ]] || [[ "$DAY_OF_WEEK" == "Sunday" ]]; then
        echo "WEEKEND LARGE SLATE STRATEGY:"
        echo "  1. Run game prioritization FIRST"
        echo "  2. Analyze top 20-30 games only"
        echo "  3. Focus on power conferences"
        echo "  4. Check for mid-major value"
        echo "  5. Build 2-4 leg parlays from qualified games"
        echo ""
    fi

    if [ "$SEASON_PHASE" = "NCAA Tournament" ]; then
        echo "⚠️  MARCH MADNESS REMINDERS:"
        echo "   - Confidence threshold: 58% → 63%"
        echo "   - Max parlay legs: 4 → 2"
        echo "   - Avoid 1-16, 2-15 matchups"
        echo "   - Focus on 5-12 through 8-9 seeds"
        echo "   - Check tournament experience"
        echo ""
    fi

    if [ "${TIMING_WINDOW:-STANDARD}" = "EARLY" ]; then
        echo "⚠️  EARLY ANALYSIS:"
        echo "   - Injury reports may update"
        echo "   - Line movements may occur"
        echo "   - Consider re-running closer to game time"
    fi

    if [ "$SEASON_PHASE" = "Early Season" ]; then
        echo "⚠️  EARLY SEASON WARNINGS:"
        echo "   - Avoid cupcake games (spread > 25)"
        echo "   - Power vs low-major = no value"
        echo "   - Focus on competitive matchups"
    fi

else
    echo "ℹ️  Not a typical WCBB game day"
    echo ""
    echo "WCBB SCHEDULE PATTERN:"
    echo "  - Saturday/Sunday: 50-100+ games (HEAVY)"
    echo "  - Thursday/Friday: 20-40 games"
    echo "  - Monday-Wednesday: 5-15 games"
    echo ""
    echo "OPTIMAL ANALYSIS TIMING:"
    echo "  - Weekends: 9:00 AM - 11:00 AM ET"
    echo "  - Weekdays: 3:00 PM - 6:00 PM ET"
fi

echo ""
echo "=========================================="
echo "🏀 Ready to analyze Women's College Basketball"
echo "=========================================="
echo ""
