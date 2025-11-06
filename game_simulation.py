"""
Game simulation utilities.
"""

import numpy as np


def simulate_game(home_advantage=0.5):
    return np.random.rand(+home_advantage > 0.5)
