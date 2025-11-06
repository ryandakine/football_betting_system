# Thursday Night Football Predictions - Quick Start

## Setup Complete ✓

Your integration workflow is ready for tonight's TNF game!

## Running Predictions

### Quick Run (Use Existing Model)
```bash
./run_tnf_predictions.py --archive
```

This will:
1. Verify data files exist
2. Run HRM Beast (inference only, int8 quantized)
3. Run HRM Sniper 
4. Run Conspiracy Bot (optional)
5. Fuse predictions
6. Display results
7. Archive predictions with timestamp

### With Training (If You Want Fresh Model)
```bash
./run_tnf_predictions.py --train --epochs 5 --archive
```

### Options
- `--train` - Train the Beast model before inference
- `--epochs N` - Number of training epochs (default: 5)
- `--no-int8` - Disable int8 quantization (use if you have GPU)
- `--archive` - Archive predictions with timestamp

## Output Files

Predictions saved to `data/referee_conspiracy/`:
- `hrm_predictions.json` - Beast model output
- `sniper_predictions.json` - Sniper model output
- `conspiracy_predictions.json` - Conspiracy bot output
- `fused_hr_predictions.json` - **Final predictions** (use this)

Logs saved to:
- `data/referee_conspiracy/tnf_predictions.log`

Archived predictions:
- `data/referee_conspiracy/predictions/TIMESTAMP_*.json`

## Test Suite

Run tests before predictions:
```bash
pytest tests/test_referee_style_analysis.py -v
```

## What Each Model Does

### 🦁 Beast (HRM Neural Net)
- Learns patterns from historical data
- 64-dim features, Bayesian scoring inference
- Outputs: script labels, under probability, flag spike

### 🎯 Sniper (Rule Engine)
- Hard-coded conspiracy heuristics
- Fast, deterministic, no ML overhead
- Detects blueballs, blackouts, flag farms

### 👁️ Conspiracy Bot
- Monitors social signals (Twitter spikes)
- Tracks betting odds drift
- Can override predictions with conspiracy signals

### ⚖️ Fusion (Vote Layer)
- Weighted vote between Beast & Sniper
- Weather umpire overrides when both agree strongly
- Final output includes provenance and confidence

## Interpreting Results

Key fields in `fused_hr_predictions.json`:

```json
{
  "2024_12_DET_LAC": {
    "h_module": "FLAG_FUCKING_FARM",      // Script type detected
    "fused_under": 0.823,                  // Under probability (0-1)
    "source": "SNIPER",                    // Which model won
    "raw_beast_conf": 0.72,                // Beast confidence
    "raw_sniper_conf": 0.96                // Sniper confidence
  }
}
```

### Script Types
- `NORMAL_FLOW` - Clean game, no conspiracy detected
- `SCRIPTED_BLUEBALL` - Red zone stalls, under tendencies
- `SCRIPTED_BLACKOUT` - Low visibility, flag surges
- `FLAG_FARM` - High penalty rate, close game manipulation
- `FLAG_FUCKING_FARM` - Overtime carnage, defensive ref bias

### Under Probability
- < 0.45: Over likely
- 0.45-0.55: Neutral
- 0.55-0.75: Under lean
- > 0.75: **Strong under**

### Source
- `BEAST`: Neural net won the vote
- `SNIPER`: Rule engine won
- `WEATHER_UMPIRE`: Weather override kicked in

## Prerequisites

Make sure you have:
1. Run `nfl_referee_conspiracy_engine.py` to generate base data
2. Installed dependencies: `pip install -r requirements.txt`
3. PyTorch + CUDA (optional, CPU/int8 works fine)

## Troubleshooting

### "Missing required data files"
Run the data engine first:
```bash
python nfl_referee_conspiracy_engine.py
```

### "Checkpoint missing"
Either:
- Run with `--train` to create new checkpoint
- Check `data/referee_conspiracy/hrm_checkpoint.pt` exists

### Conspiracy bot fails
This is OK, workflow continues without it. Check:
- `data/referee_conspiracy/conspiracy_signals.json` exists
- Twitter/odds feed is populated

## Tonight's Workflow

```bash
# 1. Verify environment
pytest tests/test_referee_style_analysis.py

# 2. Run predictions
./run_tnf_predictions.py --archive

# 3. Check results
cat data/referee_conspiracy/fused_hr_predictions.json | jq '.'

# 4. Review logs
tail -f data/referee_conspiracy/tnf_predictions.log
```

Good luck! 🏈🎲
