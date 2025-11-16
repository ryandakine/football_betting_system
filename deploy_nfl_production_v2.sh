#!/bin/bash
################################################################################
# NFL Production Deployment Pipeline (Refactored)
# Uses new unified core modules for cleaner deployment
#
# Usage: ./deploy_nfl_production_v2.sh [week_number]
# Example: ./deploy_nfl_production_v2.sh 11
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
WEEK=${1:-11}
OUTPUT_DIR="reports/nfl_week_${WEEK}"
PREDICTION_FILE="data/predictions/nfl_prediction_log.json"
BET_SELECTION_FILE="${OUTPUT_DIR}/nfl_top_bets_week_${WEEK}.txt"
REPORT_FILE="${OUTPUT_DIR}/NFL_WEEK_${WEEK}_ACTION_REPORT.md"

# Logging function
log_step() {
    echo -e "${YELLOW}[$1/6] $2${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
}

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Error handler
error_exit() {
    log_error "$1"
    echo ""
    echo "Deployment failed. Check errors above."
    exit 1
}

# Header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         NFL PRODUCTION DEPLOYMENT - WEEK ${WEEK}                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

################################################################################
# Step 1: Environment Checks
################################################################################
log_step 1 "Checking environment..."

# Check API key
if [ -z "${ODDS_API_KEY:-}" ] && [ -z "${THE_ODDS_API_KEY:-}" ]; then
    error_exit "No Odds API key found

    Set environment variable:
      export ODDS_API_KEY='your_key_here'

    Get key from: https://the-odds-api.com/"
fi
log_success "API key configured"

# Check Python
if ! command -v python3 &> /dev/null; then
    error_exit "Python 3 not found"
fi
log_success "Python 3 available"

# Check dependencies
python3 -c "import requests" 2>/dev/null || error_exit "requests library not installed. Run: pip install requests"
log_success "Python dependencies OK"

# Check core modules exist
for module in config/betting_config.py core/game_fetcher.py core/model_ensemble.py; do
    if [ ! -f "$module" ]; then
        error_exit "Core module missing: $module"
    fi
done
log_success "Core modules present"

echo ""

################################################################################
# Step 2: Fetch Live NFL Games
################################################################################
log_step 2 "Fetching live NFL games..."

if python3 nfl_live_tomorrow_plus_v2.py; then
    log_success "Games fetched successfully"
else
    error_exit "Failed to fetch NFL games

    Common issues:
      - Invalid API key (403 Forbidden)
      - Rate limit exceeded
      - Network connection problem

    Try:
      1. Check API key: echo \$ODDS_API_KEY
      2. Get new key: https://the-odds-api.com/"
fi

# Verify games file exists
if [ ! -f "data/nfl_live_games.json" ]; then
    error_exit "Games file not created"
fi

GAME_COUNT=$(python3 -c "import json; print(len(json.load(open('data/nfl_live_games.json'))))" 2>/dev/null || echo "0")
log_success "Fetched ${GAME_COUNT} NFL games"

if [ "$GAME_COUNT" -eq 0 ]; then
    error_exit "No games found for this week"
fi

echo ""

################################################################################
# Step 3: Run 12-Model Ensemble Predictions
################################################################################
log_step 3 "Running 12-model ensemble predictions..."

if python3 run_nfl_12model_deepseek_v2.py --week "$WEEK"; then
    log_success "Predictions generated"
else
    error_exit "Prediction generation failed"
fi

# Verify predictions file
if [ ! -f "$PREDICTION_FILE" ]; then
    error_exit "Predictions file not created"
fi

PRED_COUNT=$(python3 -c "import json; print(len(json.load(open('$PREDICTION_FILE'))))" 2>/dev/null || echo "0")
log_success "Generated ${PRED_COUNT} predictions"

echo ""

################################################################################
# Step 4: Run Unified Bet Selector
################################################################################
log_step 4 "Running unified bet selector (5-tier system)..."

mkdir -p "$OUTPUT_DIR"

if python3 bet_selector_unified.py nfl > "${BET_SELECTION_FILE}" 2>&1; then
    log_success "Top bets identified"
else
    log_error "Bet selection had issues (may still have output)"
fi

echo ""

################################################################################
# Step 5: Display Top Bets
################################################################################
log_step 5 "TOP BETS FOR NFL WEEK ${WEEK}:"
echo ""
cat "${BET_SELECTION_FILE}"
echo ""

################################################################################
# Step 6: Generate Action Report
################################################################################
log_step 6 "Generating action report..."

cat > "$REPORT_FILE" <<EOF
# NFL Week ${WEEK} - Action Report
**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Deployment:** Refactored v2 (unified core modules)

## Summary
- **Games Analyzed:** ${GAME_COUNT}
- **Predictions Generated:** ${PRED_COUNT}
- **Report Location:** ${OUTPUT_DIR}

## Top Bets

\`\`\`
$(cat "${BET_SELECTION_FILE}")
\`\`\`

## Next Steps

### 1. Video Scouting (CRITICAL - 75 minutes)
📋 **Checklist:** \`NFL_VIDEO_SCOUTING_CHECKLIST.md\`

For each Top 5 bet:
- [ ] Watch last 3 games (YouTube highlights - 10 min per team)
- [ ] Check injury reports (NFL.com, Twitter - 3 min)
- [ ] Verify weather forecast (Weather.com - 2 min)
- [ ] Check referee assignment (Google - 2 min)
- [ ] Review line movement (OddsPortal - 2 min)

**Bet Adjustment:**
- 5/5 checks ✅ → Bet 100% of tier amount
- 4/5 checks → Bet 70%
- 3/5 checks → Bet 50% or skip
- 2/5 or fewer ❌ → Skip bet

### 2. Saturday Evening (6:00 PM)
- [ ] Final line movement check
- [ ] Lock in bet amounts
- [ ] Set Sunday alarms

### 3. Sunday Betting Windows
- [ ] **11:00 AM - 12:45 PM:** Place early game bets (1 PM kickoffs)
- [ ] **3:30 PM - 4:20 PM:** Place late game bets (4 PM kickoffs)
- [ ] **7:00 PM - 8:15 PM:** SNF bet (only if in Top 5)

### 4. Bankroll Management
- **Current Bankroll:** \$101
- **TIER 1 MEGA EDGE:** 2.5% (\$2.53) [Rare Week ${WEEK}]
- **TIER 2 SUPER EDGE:** 2.0% (\$2.02) [Rare]
- **TIER 3 STRONG EDGE:** 1.5% (\$1.52) [Target 1-2]
- **TIER 4 MODERATE:** 1.2% (\$1.21) [Most bets]
- **TIER 5 SELECTIVE:** 1.0% (\$1.01) [Skip unless confident]

### 5. Betting Rules (DO NOT VIOLATE)
1. ✅ Video scouting is mandatory
2. ✅ Never chase losses
3. ✅ Stick to tier sizing
4. ✅ Maximum 5 bets per week
5. ✅ Circuit breaker at 25% drawdown

## Expected Results

**Conservative (55% win rate):**
- Bets: 2-3 games
- Risk: \$3-4
- Expected profit: \$0.50-1.50

**Moderate (58% win rate):**
- Bets: 3-4 games
- Risk: \$4-6
- Expected profit: \$1.50-2.50

## Files Generated
- **Games:** \`data/nfl_live_games.json\`
- **Predictions:** \`${PREDICTION_FILE}\`
- **Top Bets:** \`${BET_SELECTION_FILE}\`
- **This Report:** \`${REPORT_FILE}\`

## System Architecture (Refactored v2)
- ✅ Unified configuration: \`config/betting_config.py\`
- ✅ Unified game fetcher: \`core/game_fetcher.py\`
- ✅ Model ensemble base class: \`core/model_ensemble.py\`
- ✅ NFL 12-model implementation: \`run_nfl_12model_deepseek_v2.py\`
- ✅ Cleaner, maintainable, DRY code

## Deployment Status
✅ NFL Week ${WEEK} predictions deployed successfully!

**Next:** Video scouting → Saturday line checks → Sunday betting
EOF

log_success "Action report saved: ${REPORT_FILE}"
echo ""

################################################################################
# Summary
################################################################################
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     DEPLOYMENT COMPLETE                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}📊 NFL Week ${WEEK} System Deployed (Refactored v2)${NC}"
echo ""
echo -e "${YELLOW}📁 Files Created:${NC}"
echo "  • Games: data/nfl_live_games.json (${GAME_COUNT} games)"
echo "  • Predictions: ${PREDICTION_FILE} (${PRED_COUNT} predictions)"
echo "  • Top Bets: ${BET_SELECTION_FILE}"
echo "  • Action Report: ${REPORT_FILE}"
echo ""
echo -e "${YELLOW}🎯 Next Steps:${NC}"
echo "  1. Video scouting (75 min): NFL_VIDEO_SCOUTING_CHECKLIST.md"
echo "  2. Action plan: BETTING_ACTION_PLAN_NFL.md"
echo "  3. Saturday 6 PM: Final line checks"
echo "  4. Sunday 11 AM: Place early game bets"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
