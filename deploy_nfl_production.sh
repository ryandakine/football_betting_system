#!/bin/bash
################################################################################
# NFL Production Deployment Pipeline
# Replicates NCAA system for NFL Sunday betting
#
# Usage: ./deploy_nfl_production.sh [week_number]
# Example: ./deploy_nfl_production.sh 11
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WEEK=${1:-11}
OUTPUT_DIR="reports/nfl_week_${WEEK}"
PREDICTION_FILE="data/predictions/nfl_prediction_log.json"
BET_SELECTION_FILE="${OUTPUT_DIR}/nfl_top_bets_week_${WEEK}.txt"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         NFL PRODUCTION DEPLOYMENT - WEEK ${WEEK}                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check environment
echo -e "${YELLOW}[1/6] Checking environment...${NC}"

if [ -z "$ODDS_API_KEY" ] && [ -z "$THE_ODDS_API_KEY" ]; then
    echo -e "${RED}❌ ERROR: No Odds API key found${NC}"
    echo "   Set environment variable:"
    echo "   export ODDS_API_KEY='your_key_here'"
    exit 1
fi

echo -e "${GREEN}✅ API key configured${NC}"

# Check Python dependencies
python3 -c "import requests" 2>/dev/null || {
    echo -e "${RED}❌ ERROR: requests library not installed${NC}"
    echo "   Run: pip install requests"
    exit 1
}

echo -e "${GREEN}✅ Python dependencies OK${NC}"
echo ""

# Step 2: Fetch live NFL games
echo -e "${YELLOW}[2/6] Fetching live NFL games...${NC}"

if ! python3 nfl_live_tomorrow_plus.py; then
    echo -e "${RED}❌ ERROR: Failed to fetch NFL games${NC}"
    exit 1
fi

# Check if games were fetched
if [ ! -f "data/nfl_live_games.json" ]; then
    echo -e "${RED}❌ ERROR: No games file created${NC}"
    exit 1
fi

GAME_COUNT=$(python3 -c "import json; print(len(json.load(open('data/nfl_live_games.json'))))")
echo -e "${GREEN}✅ Fetched ${GAME_COUNT} NFL games${NC}"
echo ""

# Step 3: Run 12-model ensemble predictions
echo -e "${YELLOW}[3/6] Running 12-model ensemble predictions...${NC}"

if ! python3 run_nfl_12model_deepseek.py; then
    echo -e "${RED}❌ ERROR: Prediction generation failed${NC}"
    exit 1
fi

# Verify predictions were created
if [ ! -f "$PREDICTION_FILE" ]; then
    echo -e "${RED}❌ ERROR: Predictions file not created${NC}"
    exit 1
fi

PRED_COUNT=$(python3 -c "import json; print(len(json.load(open('$PREDICTION_FILE'))))")
echo -e "${GREEN}✅ Generated ${PRED_COUNT} predictions${NC}"
echo ""

# Step 4: Run unified bet selector
echo -e "${YELLOW}[4/6] Running unified bet selector (5-tier system)...${NC}"

mkdir -p "$OUTPUT_DIR"

if ! python3 bet_selector_unified.py nfl > "${BET_SELECTION_FILE}"; then
    echo -e "${RED}❌ ERROR: Bet selection failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Top bets identified${NC}"
echo ""

# Step 5: Display top bets
echo -e "${YELLOW}[5/6] TOP BETS FOR NFL WEEK ${WEEK}:${NC}"
echo ""
cat "${BET_SELECTION_FILE}"
echo ""

# Step 6: Generate action report
echo -e "${YELLOW}[6/6] Generating action report...${NC}"

REPORT_FILE="${OUTPUT_DIR}/NFL_WEEK_${WEEK}_ACTION_REPORT.md"

cat > "$REPORT_FILE" <<EOF
# NFL Week ${WEEK} - Action Report
**Generated:** $(date '+%Y-%m-%d %H:%M:%S')

## Summary
- **Games Analyzed:** ${GAME_COUNT}
- **Predictions Generated:** ${PRED_COUNT}
- **Report Location:** ${OUTPUT_DIR}

## Top Bets

\`\`\`
$(cat "${BET_SELECTION_FILE}")
\`\`\`

## Next Steps

### 1. Video Scouting (CRITICAL - Do NOT skip!)
See: \`NFL_VIDEO_SCOUTING_CHECKLIST.md\`

For each game in Top 5:
- [ ] Watch last 3 games of each team (YouTube highlights)
- [ ] Verify injuries match system assumptions
- [ ] Check weather forecast for game day
- [ ] Review referee assignments
- [ ] Confirm line hasn't moved significantly

### 2. Final Bet Selection
After video scouting, identify:
- **MUST BETS:** Video confirms system edge
- **SKIP BETS:** Video contradicts system (injuries, weather, etc.)
- **REDUCED BETS:** Lower confidence after review

### 3. Bet Placement Timeline
- **Saturday 6:00 PM:** Final line checks
- **Sunday 11:00 AM:** Place early game bets (1:00 PM kickoffs)
- **Sunday 3:30 PM:** Place late game bets (4:05/4:25 PM kickoffs)
- **Sunday 7:00 PM:** Place SNF bet (8:20 PM kickoff)

### 4. Bankroll Management
- **Current Bankroll:** \$101
- **Total Risk This Week:** Follow tier sizing
- **Max Single Bet:** 2.5% (\$2.53) for TIER 1 MEGA EDGE
- **Typical Bet:** 1.0-1.5% (\$1.01-\$1.52) for TIER 4-5

## Files Generated
- Predictions: \`${PREDICTION_FILE}\`
- Top Bets: \`${BET_SELECTION_FILE}\`
- This Report: \`${REPORT_FILE}\`

## Deployment Status
✅ NFL Week ${WEEK} predictions deployed successfully!
EOF

echo -e "${GREEN}✅ Action report saved to: ${REPORT_FILE}${NC}"
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     DEPLOYMENT COMPLETE                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}📊 NFL Week ${WEEK} System Deployed${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review video scouting checklist: NFL_VIDEO_SCOUTING_CHECKLIST.md"
echo "  2. Watch game film for top 5 bets (YouTube)"
echo "  3. Finalize bets: BETTING_ACTION_PLAN_NFL.md"
echo "  4. Place bets Sunday morning/afternoon"
echo ""
echo -e "${YELLOW}Files:${NC}"
echo "  • Predictions: ${PREDICTION_FILE}"
echo "  • Top Bets: ${BET_SELECTION_FILE}"
echo "  • Action Report: ${REPORT_FILE}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
