from __future__ import annotations
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
