#!/bin/bash
# Hook: Pre-Analysis Validation
# Runs BEFORE any betting analysis starts
# Makes it IMPOSSIBLE to run with invalid configuration

echo "🔍 PRE-ANALYSIS VALIDATION HOOK"
echo "================================"

# 1. Validate API Keys Exist
if [ -z "$ODDS_API_KEY" ]; then
    echo "❌ BLOCKED: ODDS_API_KEY not set"
    echo ""
    echo "WHY THIS MATTERS:"
    echo "  - System will fail mid-analysis without API access"
    echo "  - Wastes computational resources"
    echo "  - Could miss time-sensitive betting opportunities"
    echo ""
    echo "FIX: Add ODDS_API_KEY to .env file"
    exit 1
fi

# 2. Validate Bankroll Settings
if [ -z "$WCBB_BANKROLL" ]; then
    echo "⚠️  WARNING: WCBB_BANKROLL not set, using default $50,000"
fi

BANKROLL=${WCBB_BANKROLL:-50000}
if (( $(echo "$BANKROLL < 1000" | bc -l) )); then
    echo "❌ BLOCKED: Bankroll too low ($BANKROLL)"
    echo ""
    echo "WHY THIS MATTERS:"
    echo "  - Minimum $1,000 required for proper Kelly sizing"
    echo "  - Prevents over-betting on small bankrolls"
    echo "  - Risk management requires minimum bet units"
    echo ""
    exit 1
fi

# 3. Check Season Timing (WCBB: November - March)
MONTH=$(date +%m)
if (( MONTH >= 4 && MONTH <= 10 )); then
    echo "⚠️  WARNING: Outside WCBB season (November-March)"
    echo "   Current month: $(date +%B)"
    echo "   Fewer games available, reduced edge opportunities"
    echo ""
fi

# 4. Validate Configuration File Exists
if [ ! -f "wcbb_config.py" ]; then
    echo "❌ BLOCKED: wcbb_config.py not found"
    echo ""
    echo "WHY THIS MATTERS:"
    echo "  - Configuration defines betting strategy"
    echo "  - Missing config = undefined behavior"
    echo "  - Risk management rules not enforced"
    echo ""
    exit 1
fi

# 5. Check Python Dependencies
python3 -c "import pydantic, aiohttp, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ BLOCKED: Missing required Python packages"
    echo ""
    echo "FIX: Run 'pip install -r requirements.txt'"
    exit 1
fi

# 6. Network Connectivity Check
curl -s --max-time 5 https://api.the-odds-api.com > /dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  WARNING: Cannot reach Odds API"
    echo "   Check internet connection"
    echo ""
fi

echo "✅ All validations passed"
echo "📊 Bankroll: \$$BANKROLL"
echo "🏀 Sport: Women's College Basketball"
echo "🎯 Ready to analyze"
echo ""
