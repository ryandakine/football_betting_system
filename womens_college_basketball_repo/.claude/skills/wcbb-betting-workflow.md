---
name: wcbb-betting-workflow
description: |
  Complete workflow for analyzing Women's College Basketball games and generating betting recommendations.
  This Skill encodes the ENTIRE betting process so the agent cannot skip steps or make mistakes.
triggers:
  - analyze wcbb
  - women's basketball betting
  - ncaa women's basketball analysis
---

# Women's College Basketball Betting Workflow

**WHY THIS WORKFLOW EXISTS:**
- Ensures consistent analysis process
- Prevents skipping critical validation steps
- Encodes betting strategy that persists across sessions
- Makes it impossible to analyze without proper setup

## Phase 1: Pre-Flight Validation

**CRITICAL: DO NOT SKIP - These checks prevent mid-analysis failures**

1. Verify environment setup:
   ```bash
   # Check API keys
   python3 -c "from api_config import get_api_keys; assert get_api_keys()['odds_api'], 'Missing Odds API key'"
   ```

2. Validate configuration:
   ```bash
   # Test config loads without errors
   python3 wcbb_config.py
   ```

3. Check season timing:
   - WCBB Season: November - March
   - If outside season: Warn user of limited games
   - Peak performance: January - March

## Phase 2: Data Acquisition

**OBJECTIVE: Fetch games with proper error handling**

1. Initialize Odds API connection:
   ```python
   from main_analyzer import UnifiedWomensCollegeBasketballAnalyzer
   analyzer = UnifiedWomensCollegeBasketballAnalyzer(bankroll=BANKROLL)
   ```

2. Fetch games (with retry logic):
   - Sport key: `basketball_ncaaw`
   - Markets: h2h, spreads, totals
   - Timeout: 25 seconds
   - Retry: 3 attempts with exponential backoff

3. Validate data quality:
   - Check for missing odds
   - Verify game times are future
   - Ensure home/away teams present

## Phase 3: Game Prioritization

**OBJECTIVE: Focus on highest-edge opportunities first**

1. Run prioritizer:
   ```python
   from game_prioritization import GamePrioritizer
   prioritizer = GamePrioritizer()
   sorted_games = prioritizer.optimize_processing_order(games)
   ```

2. Conference priority order:
   - Big Ten (0.95)
   - SEC (0.92)
   - ACC (0.88)
   - Pac-12 (0.85)
   - Big 12 (0.82)
   - Mid-majors (bonus for hidden value)

3. Display prioritization reasoning:
   ```python
   log = prioritizer.get_prioritization_log()
   for game in log[:10]:
       print(f"{game['matchup']}: {game['priority']:.4f}")
       print(f"  Reasons: {game['reasons'][:2]}")
   ```

## Phase 4: Comprehensive Analysis

**OBJECTIVE: Generate AI-powered recommendations with multiple validation layers**

For each prioritized game:

1. Base analysis (unified system)
2. Social sentiment check
3. Crew/referee context
4. AI Council consensus (5 agents)
5. Meta-learner calibration

**RISK MANAGEMENT ENFORCEMENT:**
- Minimum edge: 5% (WCBB_MIN_EDGE_THRESHOLD)
- Minimum confidence: 58% (WCBB_CONFIDENCE_THRESHOLD)
- Maximum exposure: 10% (WCBB_MAX_EXPOSURE)
- **These are CONSTRAINTS, not suggestions**

## Phase 5: Parlay Optimization

**OBJECTIVE: Construct mathematically optimal parlays**

1. Filter qualified games:
   - Edge ≥ 5%
   - Confidence ≥ 58%

2. Generate parlays:
   - 2-leg combinations
   - 3-leg combinations (if ≥3 qualified games)
   - 4-leg combinations (if ≥4 qualified games with edge ≥8%)

3. Apply correlation penalty: 15%
   - WHY: Women's basketball games within same conference/day are correlated
   - Failure to apply = overestimated parlay value

4. Kelly criterion sizing:
   - Quarter Kelly for safety
   - Cap at 3% of bankroll
   - **NEVER exceed these limits**

## Phase 6: Output & Recommendations

**OBJECTIVE: Present actionable, prioritized recommendations**

1. High-edge single games (edge ≥ 8%)
2. Top 5 parlays (by expected value)
3. Risk metrics:
   - Total exposure
   - Worst-case loss
   - Expected ROI

4. Alerts for:
   - Holy Grail games (edge ≥ 15%, confidence ≥ 80%)
   - Conference tourneys (adjust confidence thresholds)

## Phase 7: Performance Tracking

**OBJECTIVE: Log recommendations for post-analysis**

1. Track each bet recommendation:
   ```python
   bet_tracker.track_bet({
       'game_id': game_id,
       'sport_type': 'wcbb',
       'edge': edge,
       'confidence': confidence,
       'stake': stake
   })
   ```

2. Generate report:
   - Recommendations made
   - Edge distribution
   - Confidence distribution

## Error Handling Protocol

**IF API call fails:**
1. Retry with exponential backoff (3 attempts)
2. If all retries fail: Log error, skip game, continue
3. DO NOT crash entire analysis

**IF edge calculation fails:**
1. Mark game as PASS (do not bet)
2. Log reason for failure
3. Continue to next game

**IF bankroll validation fails:**
1. STOP immediately
2. Do not proceed with analysis
3. Alert user of configuration issue

## Success Criteria

✅ Analysis complete with recommendations
✅ All recommendations within risk limits
✅ Performance tracking logged
✅ No unhandled errors

## Notes for Agent

- **This workflow is COMPLETE** - do not add or skip steps
- **Risk limits are CONSTRAINTS** - cannot be overridden
- **Error handling is MANDATORY** - do not let failures cascade
- **Season timing matters** - warn if outside peak season
