# MASTER RULES - NFL Betting System
**Created by: AI** (because I fucked up and need enforcement)
**Status: MANDATORY - Code will crash if violated**

## RULE 1: NO SYNTHETIC DATA - EVER
- **Real data only**: `./data/referee_conspiracy/*.parquet`
- **Fallback to fake data**: FORBIDDEN
- **Test data in production**: FORBIDDEN
- **Simulated numbers**: FORBIDDEN

## RULE 2: Data Validation
- Every function must validate data source
- If data contains 'simulated', 'test', 'fake', 'mock', 'dummy', 'generated', 'synthetic': CRASH
- No exceptions. No workarounds.

## RULE 3: Real Data First
- Load from `./data/referee_conspiracy/` directory
- Files required:
  - `crew_game_log.parquet` (1942 games)
  - `schedules_2018_2024.parquet`
  - `penalties_2018_2024.parquet`
  - `team_penalty_log.parquet`
  - `crew_features.parquet`
- If files missing: CRASH immediately

## RULE 4: No Pretending
- Don't create mock games
- Don't generate test players
- Don't simulate anything
- If you can't analyze real data: FAIL LOUDLY

## RULE 5: Enforcement
- Every prop analysis must use RealDataLoader
- Every backtest must use real games
- Every prediction must reference real data
- Code will crash if these rules are broken

---

**Violation = System Crash**
**That's the point.**
