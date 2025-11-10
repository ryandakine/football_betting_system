#!/bin/bash
#
# NCAA Betting Context Hook
# =========================
# Runs BEFORE every user prompt
# Auto-injects betting system state - agent never has to ask
#
# PRINCIPLE: Context Embedded (Not External)
# Investment → System: Agent forgets, hook remembers

# Check if we're in NCAA betting context
if [[ ! "$USER_MESSAGE" =~ (ncaa|college|football|bet|maction|tuesday|saturday) ]]; then
    exit 0  # Not NCAA-related, skip
fi

# Output context that will be injected into conversation
cat <<'CONTEXT'

---
**NCAA BETTING SYSTEM STATUS** (auto-injected)

💰 **Current Bankroll**: $10,000 (starting)
📊 **Strategy**: 12-Model Super Intelligence
🎯 **Target Win Rate**: 58-60% (elite level)
🏈 **Betting Schedule**: Tuesday-Saturday (college football)

**12-MODEL SYSTEM**:
- Ensemble of 12 specialized models
- Super intelligence orchestration
- Fractional Kelly (25%) bet sizing
- Confidence calibration: 0.90x multiplier

**BETTING THRESHOLDS**:
- Minimum confidence: 70%
- Minimum edge: 3%
- Maximum correlation penalty: 15%
- Risk tolerance: Medium

**CURRENT PRIORITY**: Tuesday MACtion
- Softest lines of the week
- Usually 1 game (easy deep analysis)
- Low-risk validation opportunity

**DATA STATUS**:
- Historical games: 7,331 (2015-2024)
- Market spreads: Pending scraper results
- Models trained: 3/12 loaded

**MARKET SPREAD REQUIREMENT**:
⚠️  Need 80%+ coverage for backtest validation
⚠️  Run scrapers to get historical betting lines

**ODDS API KEY**: 0c405bc90c59a6a83d77bf1907da0299

---

CONTEXT
