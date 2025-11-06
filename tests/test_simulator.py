import numpy as np
from src.analytics.monte_carlo import GameInput, simulate_game

def test_simulate_game_runs():
    gi = GameInput('Army', 'North Texas', -4.5, 52.5, sims=10000)
    res = simulate_game(gi)
    assert 0.0 <= res['home_win_prob'] <= 1.0
