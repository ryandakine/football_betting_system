# System Flow Diagram - What Actually Runs

## CLAIMED SYSTEM (What You Think Happens)

```
┌─────────────────────────────────────────────────────────────┐
│         python autonomous_betting_agent.py --week 10        │
│              "12-Model Super Intelligence"                   │
└───────────────────────────────┬─────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                 │
        ┌───────▼────────┐              ┌────────▼────────┐
        │   GAME EDGES   │              │  PLAYER PROPS   │
        │   (Models 1-11)│              │   (Model 12)    │
        └───────┬────────┘              └────────┬────────┘
                │                                 │
    ┌───────────▼──────────────┐                 │
    │  Models 1-10: AI Council │                 │
    │  ────────────────────────│                 │
    │  1. Spread Ensemble     │                 │
    │  2. Total Ensemble      │                 │
    │  3. Moneyline Ensemble  │                 │
    │  4. First Half Spread   │                 │
    │  5. Home Team Total     │                 │
    │  6. Away Team Total     │                 │
    │  7. XGBoost            │                 │
    │  8. Neural Network     │                 │
    │  9. Stacking Meta      │                 │
    │  10. Situational       │                 │
    └──────────┬───────────────┘                 │
               │                                  │
    ┌──────────▼────────────┐          ┌─────────▼────────────┐
    │  Model 11: Referee    │          │  Model 12: Props     │
    │  Intelligence         │          │  Intelligence        │
    │  ───────────────────  │          │  ──────────────────  │
    │  640+ Team-Ref Pairs │          │  Player Predictions  │
    │  22 Referee Profiles │          │  Yards, TDs, Catches │
    └──────────┬────────────┘          └─────────┬────────────┘
               │                                  │
               └──────────────┬───────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  MASTER REPORT     │
                    │  12 Models Combined│
                    │  Top 10 Plays      │
                    └────────────────────┘
```

---

## ACTUAL SYSTEM (What Really Happens)

```
┌─────────────────────────────────────────────────────────────┐
│         python autonomous_betting_agent.py --week 10        │
│              "Claims 12 models, uses 1.5"                   │
└───────────────────────────────┬─────────────────────────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                 │
        ┌───────▼────────┐              ┌────────▼────────┐
        │   GAME EDGES   │              │  PLAYER PROPS   │
        │  (Referee Only)│              │    ❌ SKIPPED   │
        └───────┬────────┘              └─────────────────┘
                │                             (Returns empty)
    ┌───────────▼──────────────┐
    │  Models 1-10: AI Council │
    │  ❌ NOT LOADED           │
    │  ────────────────────────│
    │  1. ❌ Not trained      │
    │  2. ❌ Not trained      │
    │  3. ❌ Not trained      │
    │  4. ❌ Not trained      │
    │  5. ❌ Not trained      │
    │  6. ❌ Not trained      │
    │  7. ❌ Not trained      │
    │  8. ❌ Not trained      │
    │  9. ❌ Not trained      │
    │  10. ❌ Not used        │
    └───────────────────────────┘
                │
                │ (Skipped entirely)
                │
    ┌──────────▼────────────┐
    │  Model 11: Referee    │  ← ONLY MODEL RUNNING
    │  Intelligence ✅      │
    │  ───────────────────  │
    │  640+ Team-Ref Pairs │
    │  22 Referee Profiles │
    │  Multi-market edges  │
    └──────────┬────────────┘
               │
    ┌──────────▼────────────┐
    │  Sentiment Analysis   │  ← NEWLY ADDED
    │  (Half a model) ✅    │
    │  ───────────────────  │
    │  Contrarian signals  │
    │  Public vs Sharp     │
    │  Reddit sentiment    │
    └──────────┬────────────┘
               │
    ┌──────────▼────────────┐
    │  MASTER REPORT        │
    │  Only Referee Edges   │
    │  No Prop Edges        │
    │  No AI Council Edges  │
    └───────────────────────┘
```

---

## DETAILED EXECUTION FLOW

### Entry: autonomous_betting_agent.py

```
START: python autonomous_betting_agent.py --week 10
│
├─ STEP 1: Run Game Analyzer
│  └─ subprocess.run(['python', 'auto_weekly_analyzer.py', '--week', '10', '--json'])
│     │
│     └─ auto_weekly_analyzer.py
│        │
│        ├─ __init__():
│        │  ├─ ✅ Load RefereeIntelligenceModel()
│        │  ├─ ✅ Load TeamRefereeParser()
│        │  ├─ ✅ Load SentimentFeatureExtractor()
│        │  ├─ ✅ Load IntelligentModelSelector()
│        │  └─ ❌ NO AI Council (Models 1-10)
│        │
│        ├─ fetch_real_games(week=10):
│        │  ├─ ✅ Call The Odds API for betting lines
│        │  ├─ ✅ Call Football Zebras for referee assignments
│        │  └─ ✅ Return list of games with odds + referees
│        │
│        └─ For each game:
│           ├─ analyze_single_game(game):
│           │  │
│           │  ├─ Get referee profile
│           │  ├─ Get team-referee bias
│           │  ├─ Detect referee edges (ALL bet types):
│           │  │  ├─ Full game spread
│           │  │  ├─ Full game total
│           │  │  ├─ Moneyline
│           │  │  ├─ First half spread
│           │  │  ├─ Home team total
│           │  │  └─ Away team total
│           │  │
│           │  ├─ Extract sentiment:
│           │  │  ├─ Contrarian opportunity
│           │  │  ├─ Public sentiment
│           │  │  └─ Sharp/public splits
│           │  │
│           │  ├─ Run intelligent model selector:
│           │  │  └─ Recommend: Referee Intelligence (highest confidence)
│           │  │              or Contrarian Sentiment (if strong divergence)
│           │  │
│           │  └─ Return edges found
│           │
│           └─ Output JSON: {
│                 games_analyzed: 14,
│                 edges_found: 7,
│                 top_edges: [{game, edge_type, confidence, reasoning}, ...]
│              }
│
├─ STEP 2: Run Props Analyzer
│  └─ ❌ SKIPPED
│     └─ Return: {props_analyzed: 0, edges_found: 0, top_edges: []}
│
├─ STEP 3: Generate Master Report
│  └─ Combine:
│     ├─ Game edges (from referee intelligence only)
│     ├─ Prop edges (empty)
│     └─ Write to reports/week_10_master_report.txt
│
END: Master report saved
```

---

## MODEL LOADING COMPARISON

### What You Think Loads

```
EnhancedAICouncil.__init__():
│
├─ Load Models 1-3 (Base Ensembles)
│  ├─ models/spread_ensemble.pkl
│  ├─ models/total_ensemble.pkl
│  └─ models/moneyline_ensemble.pkl
│
├─ Load Models 4-6 (New Targets)
│  ├─ models/first_half_spread.pkl
│  ├─ models/home_team_total.pkl
│  └─ models/away_team_total.pkl
│
├─ Load Models 7-8 (Algo Variants)
│  ├─ models/xgboost_ensemble.pkl
│  └─ models/neural_net_deep.pkl
│
├─ Load Model 9 (Stacking)
│  └─ models/stacking_meta.pkl
│
├─ Initialize Model 10 (Situational)
│  └─ SituationalSpecialist() ✅
│
└─ Load Model 11 (Referee)
   └─ RefereeIntelligenceModel() ✅
```

### What Actually Loads

```
EnhancedAICouncil.__init__():
│
├─ Create EnhancedModelRegistry()
│  └─ self.models = {}  ← EMPTY DICT!
│
├─ Models 1-9:
│  └─ ❌ NOTHING LOADED
│     (No .pkl files exist, registry is empty)
│
├─ Model 10 (Situational):
│  └─ ✅ Initialized but NOT USED
│     (Never called in auto_weekly_analyzer)
│
└─ Model 11 (Referee):
   └─ ✅ RefereeIntelligenceModel()
      ├─ Loads referee_stats.json
      ├─ Loads team-referee pairings
      └─ ONLY MODEL ACTUALLY WORKING
```

---

## THE WONKY PARTS

### Wonky #1: Empty Model Registry

```python
# enhanced_model_architectures.py
class EnhancedModelRegistry:
    def __init__(self):
        self.models = {}  # ← EMPTY!

# enhanced_ai_council.py
class EnhancedAICouncil:
    def __init__(self):
        self.enhanced_registry = EnhancedModelRegistry()
        # ^^^ Creates empty registry
        # Never calls: self.enhanced_registry.register_model(...)
```

**Result**: Claims 10 models, has 0 models

### Wonky #2: Hardcoded Predictions

```python
# analyze_single_game.py lines 60-64
game_data["spread_model_home_pct"] = 0.52  # FAKE!
game_data["total_model_over_pct"] = 0.48   # FAKE!
game_data["home_advantage_pct"] = 0.51     # FAKE!

council = EnhancedAICouncil()
prediction = council.make_unified_prediction(game_data)
# ^^^ Thinks it has real model predictions, but they're 50/50 neutral
```

**Result**: EnhancedAICouncil uses meaningless values

### Wonky #3: Props Skipped

```python
# autonomous_betting_agent.py line 226
def _run_prop_analyzer(self, week: int):
    logger.info("⚠️  Skipping prop analyzer (requires sportsbook data scraping)")
    return {
        'props_analyzed': 0,
        'edges_found': 0,
        'top_edges': [],
        'raw_output': 'Props analyzer skipped'
    }
```

**Result**: Claims Model 12 (Props), never runs it

### Wonky #4: False Documentation

```python
# autonomous_betting_agent.py line 246
report.append("System: 12-Model Super Intelligence")

# autonomous_betting_agent.py line 379
report.append("System: 12-Model NFL Super Intelligence")
```

**Reality**: Only 1 model (referee intelligence) + sentiment (0.5 model)

**Total**: 1.5 models, not 12

---

## FILES COMPARISON

### Files That Actually Run

```
✅ autonomous_betting_agent.py       ← Entry point
✅ auto_weekly_analyzer.py           ← Calls Model 11
✅ referee_intelligence_model.py     ← Model 11
✅ parse_team_referee_pairings.py    ← Team-ref data
✅ nfl_odds_integration.py           ← The Odds API
✅ referee_assignments_fetcher.py    ← Football Zebras
✅ ai_council_with_sentiment.py      ← Sentiment analysis
✅ intelligent_model_selector.py     ← Model selection (NEW)
```

### Files That DON'T Run (Dead Code)

```
❌ train_all_10_models.py           ← Never executed
❌ enhanced_ai_council.py            ← Imported but models not loaded
❌ analyze_single_game.py            ← Hardcoded values
❌ analyze_game_simple.py            ← Unused
❌ analyze_props_weekly.py           ← Skipped
❌ ultimate_main_orchestrator.py     ← Unused
❌ enhanced_nfl_with_social.py       ← Unused
❌ main.py                           ← Unused
❌ agent_coordinator.py              ← Unused
❌ + ~40 more unused files
```

---

## SUMMARY VISUAL

```
┌─────────────────────────────────────────┐
│  WHAT YOU THINK YOU HAVE: 12 MODELS    │
├─────────────────────────────────────────┤
│  1. ❌ Spread Ensemble                 │
│  2. ❌ Total Ensemble                  │
│  3. ❌ Moneyline Ensemble              │
│  4. ❌ First Half Spread               │
│  5. ❌ Home Team Total                 │
│  6. ❌ Away Team Total                 │
│  7. ❌ XGBoost                         │
│  8. ❌ Neural Network                  │
│  9. ❌ Stacking Meta                   │
│  10. ❌ Situational                    │
│  11. ✅ Referee Intelligence ← WORKS  │
│  12. ❌ Prop Intelligence              │
│  + 0.5 ✅ Sentiment Analysis ← WORKS  │
└─────────────────────────────────────────┘

ACTUAL: 1.5 models working
CLAIMED: 12 models working
DISCREPANCY: 10.5 models missing!
```

---

## RECOMMENDED FIX

### Option A: Honest Branding
```
Change "12-Model Super Intelligence"
To: "Referee Intelligence + Sentiment System"

Focus on what works:
✅ 640+ team-referee pairings
✅ Contrarian sentiment signals
✅ Multi-market edge detection
✅ Real data integration
```

### Option B: Train the 10 Models
```
1. Collect 15 years of training data
2. Run: python train_advanced_ml_models.py
3. Wire models into auto_weekly_analyzer.py
4. Load .pkl files in EnhancedModelRegistry
5. Actually use all 12 models

Estimated time: 40+ hours
```

**Recommendation**: Option A

Your referee intelligence + sentiment system is solid. Don't overcomplicate with untrained models you're not using.

---

## THE BOTTOM LINE

```
┌────────────────────────────────────────────┐
│  YOU HAVE A GOOD SYSTEM                    │
│  (Referee Intelligence + Sentiment)        │
│                                            │
│  BUT IT'S WONKY BECAUSE:                   │
│  1. Claims 12 models, uses 1.5            │
│  2. Dead code everywhere                   │
│  3. Hardcoded fake values                  │
│  4. Empty model registries                 │
│  5. False documentation                    │
│                                            │
│  FIX: Be honest about what you have        │
│       Clean up the dead code               │
│       Focus on referee + sentiment         │
└────────────────────────────────────────────┘
```
