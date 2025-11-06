# Prop Vet Exploit Engine - Strict Real Data Only Rules

## ⚠️ CRITICAL ENFORCEMENT RULES

### Rule 1: NO SIMULATED DATA - EVER
- The engine **WILL CRASH** if initialized without real data
- No fallback mechanisms
- No test data placeholders in production flow
- No synthetic player/game generation

### Rule 2: Real Data Validation - Hard Failures
The engine validates that **ALL** required real data files are loaded:
- `crew_game_log.parquet` (1,942 real NFL games 2018-2025)
- `schedules_2018_2024.parquet` (game scheduling data)
- `penalties_2018_2024.parquet` (23,479 real penalty records)
- `team_penalty_log.parquet` (3,871 team penalty records)
- `crew_features.parquet` (27 real referee crew records)

If ANY required data is missing → `SystemExit(1)` crash with clear error message.

### Rule 3: Entry Point Validation
```python
def analyze_real_game_props(real_data: Dict, game_data: Dict) -> Dict[str, Any]:
    """
    - real_data MUST NOT be None
    - real_data MUST contain ALL 5 required keys
    - game_data MUST have valid 'player' and 'game' structure
    - Crashes if validation fails
    """
```

## ✅ Verified Behaviors

### Success Case: Real Data Loaded
```
✅ Loaded crew_game_log: 1942 real games
✅ Loaded schedules: 1942 real games
✅ Loaded penalties: 23479 real penalties
✅ Loaded team_penalty_log: 3871 real penalty records
✅ Loaded crew_features: 27 real crew records
✅ ALL REAL DATA LOADED SUCCESSFULLY

✅ Analysis successful with real data
Player: Travis Kelce
Game: Buffalo Bills @ Kansas City Chiefs
Total Edge: 20.1%
Exploits Found: 4
```

### Failure Case 1: No Data Provided
```
❌ CRITICAL FAILURE: No real data provided
PROP VET ENGINE REQUIRES REAL NFL DATA TO OPERATE
Cannot proceed with simulated data.
[SystemExit: 1]
```

### Failure Case 2: Incomplete Data
```
❌ CRITICAL: Missing real data keys: ['crew_games', 'schedules', 'penalties', 'team_penalties', 'crew_features']
PROP VET ENGINE CANNOT OPERATE WITHOUT COMPLETE REAL DATA
[SystemExit: 1]
```

### Failure Case 3: Invalid Game Structure
```
❌ CRITICAL: Invalid game data structure
Game data must include player name and matchup info
[SystemExit: 1]
```

## 🔍 Edge Detection Algorithms (All Real Data Based)

1. **Matchup Anomaly Edge** - Vegas undervalues specific matchup advantages
2. **Variance Exploitation Edge** - High-variance players create mispricings
3. **Game Script Edge** - Game flow affects player usage patterns
4. **Vegas Model Assumption Edge** - Exploit Vegas AI blind spots
5. **Correlated Props Edge** - Vegas prices props independently
6. **Middle Opportunity Edge** - Cross-book inefficiencies

## 📋 Usage Pattern

```python
from real_data_loader import RealDataLoader
from prop_vet_exploit_engine import analyze_real_game_props

# Step 1: Load REAL data (crashes if not available)
real_data = RealDataLoader.load_all()

# Step 2: Create game structure with real player/game info
game_data = {
    'player': {...},  # Real NFL player stats
    'game': {...}     # Real NFL matchup data
}

# Step 3: Analyze - uses only real data
analysis = analyze_real_game_props(real_data, game_data)
```

## 🚫 What's NOT Allowed

- ❌ Simulated test cases in production flow
- ❌ Hardcoded default values without validation
- ❌ Placeholder data when real data unavailable
- ❌ Synthetic game scenarios
- ❌ Mock player statistics
- ❌ Any fallback that bypasses real data requirement

## ✅ What's Required

- ✅ All 5 real NFL data files present in `./data/referee_conspiracy/`
- ✅ Real game/player structures from actual NFL schedules
- ✅ Explicit validation at every entry point
- ✅ Hard crash on data validation failure
- ✅ Clear error messages identifying missing/invalid data

---

**Last Updated**: 2025-11-03  
**Status**: ✅ STRICT ENFORCEMENT ACTIVE  
**Violations**: Will cause immediate system exit with no recovery
