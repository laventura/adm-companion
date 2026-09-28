"""The park: a gridworld for imitation learning (this site's problem).

A walker crosses an 8 x 8 park to the gate in the top-right corner. Each
cell is path, grass or mud; walking is pleasant on paths, slower on
grass and miserable in mud. Each step goes the intended way w.p. 0.8 and
slips to one of the other three directions otherwise (walls: stay put).
Rewards depend only on the terrain of the current cell, through features
beta(s, a) = [path, grass, mud, gate] (one-hot, all zero at the end):
R(s, a) = phi_true . beta(s, a), with phi_true = (-1, -2, -6, 10). From the
gate, any action ends the walk (an absorbing end state). Discount 0.95.

Two layouts share the terrain types, so a reward learned on one can be
tested on the other. Mirrors julia/src/problems/park.jl.
"""

import numpy as np

from adm.problems import DiscreteMDP

LAYOUTS = {
    "A": ["GGGGPPPX",
          "GMMGPGGG",
          "GMMGPGMG",
          "PPPPPGMG",
          "PGGGGGMG",
          "PGMMMGGG",
          "PGMMMGPP",
          "PPPPPPPG"],
    "B": ["PPPGGGGX",
          "GGPGMMGP",
          "GMPGMMGP",
          "GMPPPPPP",
          "GMMGGGMG",
          "GGGGMGMG",
          "PPPPMGGG",
          "PGGGMMMG"],
}
TERRAIN = "PGMX"                       # feature order: path, grass, mud, gate
PHI_TRUE = np.array([-1.0, -2.0, -6.0, 10.0])
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))   # N, E, S, W
P_INTENDED = 0.8
GAMMA = 0.95


def park(layout: str = "A", p_intended: float = P_INTENDED, gamma: float = GAMMA):
    """Returns (mdp, features): the DiscreteMDP (cells in row-major order,
    then the end state) and the feature array beta[s, a, k]. The MDP's
    reward is features @ PHI_TRUE."""
    grid = LAYOUTS[layout]
    rows, cols = len(grid), len(grid[0])
    n = rows * cols + 1
    end = n - 1
    T = np.zeros((n, 4, n))
    beta = np.zeros((n, 4, len(TERRAIN)))
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            kind = TERRAIN.index(grid[r][c])
            beta[s, :, kind] = 1.0                  # the same for every action
            if grid[r][c] == "X":
                T[s, :, end] = 1.0                   # the gate ends the walk
                continue
            for a in range(4):
                for d, (dr, dc) in enumerate(MOVES):
                    p = p_intended if d == a else (1 - p_intended) / 3
                    r2, c2 = r + dr, c + dc
                    if not (0 <= r2 < rows and 0 <= c2 < cols):
                        r2, c2 = r, c                # a wall: stay put
                    T[s, a, r2 * cols + c2] += p
    T[end, :, end] = 1.0
    R = beta @ PHI_TRUE
    states = [(r, c) for r in range(rows) for c in range(cols)] + ["end"]
    return DiscreteMDP(T, R, gamma, states, ["N", "E", "S", "W"]), beta


def left_edge_starts(layout: str = "A") -> np.ndarray:
    """Start distribution: uniform over the left column's cells."""
    grid = LAYOUTS[layout]
    rows, cols = len(grid), len(grid[0])
    b = np.zeros(rows * cols + 1)
    b[[r * cols for r in range(rows)]] = 1 / rows
    return b
