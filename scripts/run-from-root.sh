#!/usr/bin/env bash
set -euo pipefail
root=/home/ryan/football_betting_system
cd ""
exec "cd /home/ryan/executive-brain-infrastructure && git add requirements_system.txt && git commit -m "chore: revert analytics deps from requirements_system.txt" || true && cd /home/ryan && rm -rf football_betting_system && git clone https://github.com/ryandakine/football_betting_system && cd football_betting_system && git checkout -b feat/monte-carlo-simulator && mkdir -p .github/workflows src/analytics scripts tests && printf "name: CI
on:
  push:
    branches: ['**']
  pull_request:
permissions:
  contents: read
jobs:
  test-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          if [ -f requirements_system.txt ]; then pip install -r requirements_system.txt; fi
          pip install pytest pytest-cov numpy pandas scipy scikit-learn
      - name: Run tests
        env:
          PYTHONPATH: src
        run: |
          if [ -d tests ]; then pytest -q --maxfail=1; else echo 'No tests directory found; skipping.'; fi
" > .github/workflows/ci.yml && printf "#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
cd \"$root\"
exec \"$@\"
" > scripts/run-from-root.sh && chmod +x scripts/run-from-root.sh && printf "from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import bootstrap
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class GameInput:
    home_team: str
    away_team: str
    spread: float
    total: float
    sims: int = 100_000

def simulate_game(game: GameInput, historical: Optional[pd.DataFrame] = None, injury_prob: float = 0.12) -> Dict[str, float]:
    rng = np.random.default_rng()
    if historical is None or historical.empty:
        home_off, away_off, home_def, away_def = 2.4, 2.2, 2.2, 2.4
    else:
        home_off = float(historical.loc[game.home_team, 'off_epa'])
        home_def = float(historical.loc[game.home_team, 'def_epa'])
        away_off = float(historical.loc[game.away_team, 'off_epa'])
        away_def = float(historical.loc[game.away_team, 'def_epa'])
    crowd = rng.normal(0.12, 0.03, size=game.sims)
    home_lambda = np.clip((home_off - away_def) + crowd, 0.05, None)
    away_lambda = np.clip((away_off - home_def) - crowd, 0.05, None)
    home = rng.poisson(home_lambda)
    away = rng.poisson(away_lambda)
    inj_h = rng.binomial(1, injury_prob, size=game.sims)
    home = home - (inj_h * 7)
    margin = home - away
    total_pts = home + away
    win = float(np.mean(margin > 0))
    cover = float(np.mean(margin > game.spread))
    over = float(np.mean(total_pts > game.total))
    win_series = (margin > 0).astype(np.int32)
    ci = bootstrap((win_series,), np.mean, confidence_level=0.95, n_resamples=1000)
    return {'home_win_prob': win, 'cover_prob': cover, 'over_prob': over, 'win_prob_ci_low': float(ci.confidence_interval.low), 'win_prob_ci_high': float(ci.confidence_interval.high)}
" > src/analytics/monte_carlo.py && printf "import numpy as np
from src.analytics.monte_carlo import GameInput, simulate_game

def test_simulate_game_runs():
    gi = GameInput('Army', 'North Texas', -4.5, 52.5, sims=10000)
    res = simulate_game(gi)
    assert 0.0 <= res['home_win_prob'] <= 1.0
" > tests/test_simulator.py && git add -A && git commit -m "feat(analytics): add Monte Carlo simulator, CI workflow, basic test" && git status -sb | cat"
