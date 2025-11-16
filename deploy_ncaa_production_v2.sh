#!/bin/bash
################################################################################
# NCAA Production Deployment Pipeline (Refactored)
# Uses new unified core modules for cleaner deployment
#
# Usage: ./deploy_ncaa_production_v2.sh [week_number]
# Example: ./deploy_ncaa_production_v2.sh 12
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
WEEK=${1:-12}
OUTPUT_DIR="reports/ncaa_week_${WEEK}"
PREDICTION_FILE="data/predictions/prediction_log.json"
BET_SELECTION_FILE="${OUTPUT_DIR}/ncaa_top_bets_week_${WEEK}.txt"
REPORT_FILE="${OUTPUT_DIR}/NCAA_WEEK_${WEEK}_ACTION_REPORT.md"

# Logging functions
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
echo -e "${BLUE}║        NCAA PRODUCTION DEPLOYMENT - WEEK ${WEEK}                   ║${NC}"
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

# Check core modules
for module in config/betting_config.py core/game_fetcher.py core/model_ensemble.py; do
    if [ ! -f "$module" ]; then
        error_exit "Core module missing: $module"
    fi
done
log_success "Core modules present"

echo ""

################################################################################
# Step 2: Fetch Live NCAA Games
################################################################################
log_step 2 "Fetching live NCAA games..."

if python3 ncaa_live_week_11_plus_v2.py; then
    log_success "Games fetched successfully"
else
    error_exit "Failed to fetch NCAA games

    Common issues:
      - Invalid API key (403 Forbidden)
      - Rate limit exceeded
      - Network connection problem
      - No games scheduled this week

    Try:
      1. Check API key: echo \$ODDS_API_KEY
      2. Get new key: https://the-odds-api.com/"
fi

# Verify games file exists
if [ ! -f "data/ncaa_live_games.json" ]; then
    error_exit "Games file not created"
fi

GAME_COUNT=$(python3 -c "import json; print(len(json.load(open('data/ncaa_live_games.json'))))" 2>/dev/null || echo "0")
log_success "Fetched ${GAME_COUNT} NCAA games"

if [ "$GAME_COUNT" -eq 0 ]; then
    log_error "No games found for this week"
    log_info "This may be normal if:"
    log_info "  - Week ${WEEK} games haven't been posted yet"
    log_info "  - Season is over (NCAA has 15 regular season weeks)"
    log_info "  - It's a bye week for most teams"
    exit 1
fi

echo ""

################################################################################
# Step 3: Run 12-Model Ensemble Predictions
################################################################################
log_step 3 "Running 12-model ensemble predictions..."

if python3 run_ncaa_12model_deepseek_v2.py --week "$WEEK"; then
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

if python3 bet_selector_unified.py ncaa > "${BET_SELECTION_FILE}" 2>&1; then
    log_success "Top bets identified"
else
    log_error "Bet selection had issues (may still have output)"
fi

echo ""

################################################################################
# Step 5: Display Top Bets
################################################################################
log_step 5 "TOP BETS FOR NCAA WEEK ${WEEK}:"
echo ""
cat "${BET_SELECTION_FILE}"
echo ""

################################################################################
# Step 6: Generate Action Report
################################################################################
log_step 6 "Generating action report..."

# Determine game day based on week
GAME_DAY="Saturday"
if [ "$WEEK" -lt 2 ]; then
    GAME_DAY="Thursday/Friday/Saturday (Week ${WEEK})"
fi

cat > "$REPORT_FILE" <<EOF
# NCAA Week ${WEEK} - Action Report
**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Deployment:** Refactored v2 (unified core modules)

## Summary
- **Games Analyzed:** ${GAME_COUNT}
- **Predictions Generated:** ${PRED_COUNT}
- **Report Location:** ${OUTPUT_DIR}
- **Primary Game Day:** ${GAME_DAY}

## Top Bets

\`\`\`
$(cat "${BET_SELECTION_FILE}")
\`\`\`

## NCAA Week ${WEEK} Characteristics

**Early Season (Weeks 1-4):**
- ✅ MEGA EDGE opportunities (P5 vs cupcakes)
- ✅ High predictability (talent mismatches)
- ✅ Non-conference games (81-88% home win rate)
- 📈 Target: 70-85% win rate

**Mid Season (Weeks 5-8):**
- ⚠️  Moderate edges (conference play starts)
- ⚠️  More competitive games
- 📈 Target: 60-70% win rate

**Late Season (Weeks 9-12):**
- ⚠️  Conference games (closer matchups)
- ⚠️  Rivalry games (unpredictable)
- ⚠️  Bowl positioning (motivation varies)
- 📈 Target: 55-60% win rate

**Your Week:** Week ${WEEK} expectations above

## Next Steps

### 1. Video Scouting (CRITICAL - 75 minutes)
📋 **Use NFL checklist** (same process applies to NCAA)

For each Top 5 bet:
- [ ] Watch last 3 games (YouTube highlights - 10 min per team)
- [ ] Check injury reports (ESPN, team sites - 3 min)
- [ ] Verify weather forecast (only for outdoor games - 2 min)
- [ ] Check referee assignment (if available - 2 min)
- [ ] Review line movement (OddsPortal - 2 min)

**Bet Adjustment:**
- 5/5 checks ✅ → Bet 100% of tier amount
- 4/5 checks → Bet 70%
- 3/5 checks → Bet 50% or skip
- 2/5 or fewer ❌ → Skip bet

### 2. Friday Evening (if Friday games)
- [ ] Final line movement check (6:00 PM)
- [ ] Place Friday night bets (6:00-7:00 PM window)
- [ ] Track MACtion results (if applicable)

### 3. Saturday Betting Windows
- [ ] **11:30 AM:** Place early game bets (12:00 PM kickoffs)
- [ ] **3:00 PM:** Place afternoon bets (3:30 PM kickoffs)
- [ ] **6:30 PM:** Place evening bets (7:00/7:30 PM kickoffs)

### 4. Bankroll Management
- **Current Bankroll:** \$101
- **TIER 1 MEGA EDGE:** 2.5% (\$2.53) [Only Weeks 1-3 typically]
- **TIER 2 SUPER EDGE:** 2.0% (\$2.02) [Early season non-conf]
- **TIER 3 STRONG EDGE:** 1.5% (\$1.52) [Conference home games]
- **TIER 4 MODERATE:** 1.2% (\$1.21) [Most Week ${WEEK} bets]
- **TIER 5 SELECTIVE:** 1.0% (\$1.01) [Late season only]

### 5. NCAA-Specific Betting Rules

**Auto-Bet Situations (Week 1-3):**
- Power 5 home vs FCS opponent with 20+ point spread
- Top 10 team home vs unranked non-conference
- Conference champ home vs Group of 5 in Week 1-2

**Auto-Skip Situations:**
- Backup QB making first start in rivalry game
- 3+ defensive starters out
- Severe weather (lightning delays common in fall)
- Line moved 7+ points against you
- Neutral site games (bowl games, conference championships)

## Expected Results

**Early Season (Weeks 1-4):**
\`\`\`
Expected: 4-6 quality bets
Win rate: 70-85%
Risk: \$6-10
Profit: \$3-6
\`\`\`

**Mid Season (Weeks 5-8):**
\`\`\`
Expected: 3-5 quality bets
Win rate: 60-70%
Risk: \$4-7
Profit: \$1.50-3.50
\`\`\`

**Late Season (Weeks 9-12):**
\`\`\`
Expected: 2-4 quality bets
Win rate: 55-60%
Risk: \$3-5
Profit: \$0.50-2.00
\`\`\`

## Files Generated
- **Games:** \`data/ncaa_live_games.json\`
- **Predictions:** \`${PREDICTION_FILE}\`
- **Top Bets:** \`${BET_SELECTION_FILE}\`
- **This Report:** \`${REPORT_FILE}\`

## System Architecture (Refactored v2)
- ✅ Unified configuration: \`config/betting_config.py\`
- ✅ Unified game fetcher: \`core/game_fetcher.py\`
- ✅ Model ensemble base class: \`core/model_ensemble.py\`
- ✅ NCAA 12-model implementation: \`run_ncaa_12model_deepseek_v2.py\`
- ✅ Consistent with NFL system
- ✅ Cleaner, maintainable, DRY code

## Deployment Status
✅ NCAA Week ${WEEK} predictions deployed successfully!

**Next:** Video scouting → Friday/Saturday betting
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
echo -e "${GREEN}📊 NCAA Week ${WEEK} System Deployed (Refactored v2)${NC}"
echo ""
echo -e "${YELLOW}📁 Files Created:${NC}"
echo "  • Games: data/ncaa_live_games.json (${GAME_COUNT} games)"
echo "  • Predictions: ${PREDICTION_FILE} (${PRED_COUNT} predictions)"
echo "  • Top Bets: ${BET_SELECTION_FILE}"
echo "  • Action Report: ${REPORT_FILE}"
echo ""
echo -e "${YELLOW}🎯 Next Steps:${NC}"
echo "  1. Video scouting (75 min): NFL_VIDEO_SCOUTING_CHECKLIST.md (same process)"
echo "  2. Friday 6 PM: Place Friday night bets (if any)"
echo "  3. Saturday 11:30 AM: Place early game bets"
echo "  4. Saturday 3 PM: Place afternoon game bets"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
