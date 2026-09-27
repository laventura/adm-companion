"""Hex World (ADM Appendix F.1): cross a map of hexagonal tiles to reach a reward.

Tiles are (row, col) in "odd-r" offset coordinates: row 0 is at the top, and odd
rows sit half a tile to the right. Each action picks one of six directions. The
move goes that way with probability 0.7, or veers to one of the two neighboring
directions with probability 0.15 each. Moving off the map leaves you in place at
a cost of 1. Acting on a reward tile collects its reward and ends the episode by
moving to an absorbing terminal state, which is the last state index.
"""

import numpy as np

from adm.problems import DiscreteMDP

# Counterclockwise from east, so neighbors in this tuple are neighboring directions.
DIRECTIONS = ("E", "NE", "NW", "W", "SW", "SE")
_STEPS = {  # (row step, col step) for even rows, then for odd rows
    "E": ((0, 1), (0, 1)), "W": ((0, -1), (0, -1)),
    "NE": ((-1, 0), (-1, 1)), "NW": ((-1, -1), (-1, 0)),
    "SE": ((1, 0), (1, 1)), "SW": ((1, -1), (1, 0)),
}

STANDARD_TILES = (
    [(0, c) for c in range(10)]
    + [(1, c) for c in (0, 1, 2, 4, 7, 8)]
    + [(2, c) for c in (0, 1, 2, 3, 5, 6, 7, 9)]
)
STANDARD_REWARDS = {(1, 0): 5.0, (2, 2): -10.0, (2, 9): 10.0}


def neighbor(tile: tuple[int, int], direction: str) -> tuple[int, int]:
    """The tile one step from `tile` in `direction` (it may not exist)."""
    r, c = tile
    dr, dc = _STEPS[direction][r % 2]
    return (r + dr, c + dc)


def hex_mdp(tiles, rewards, gamma=0.9, p_intended=0.7, bump_cost=1.0) -> DiscreteMDP:
    """Build the MDP for any map of tiles and reward tiles."""
    tiles = list(tiles)
    index = {tile: i for i, tile in enumerate(tiles)}
    n, terminal = len(tiles) + 1, len(tiles)
    T = np.zeros((n, len(DIRECTIONS), n))
    R = np.zeros((n, len(DIRECTIONS)))
    p_veer = (1 - p_intended) / 2
    for i, tile in enumerate(tiles):
        if tile in rewards:
            R[i, :] = rewards[tile]
            T[i, :, terminal] = 1.0
            continue
        for a in range(len(DIRECTIONS)):
            # da = 0 is the chosen direction; da = -1 and +1 are the two
            # neighboring directions (60 degrees either way, wrapping around).
            for da, p in ((0, p_intended), (-1, p_veer), (1, p_veer)):
                target = neighbor(tile, DIRECTIONS[(a + da) % len(DIRECTIONS)])
                if target in index:
                    T[i, a, index[target]] += p
                else:
                    # Off the map: stay put, and charge the bump cost in
                    # expectation (weighted by how likely this outcome is).
                    T[i, a, i] += p
                    R[i, a] -= p * bump_cost
    T[terminal, :, terminal] = 1.0
    return DiscreteMDP(T, R, gamma, tiles + ["terminal"], list(DIRECTIONS))


def hex_world() -> DiscreteMDP:
    """The standard 24-tile Hex World with rewards +5, -10 and +10."""
    return hex_mdp(STANDARD_TILES, STANDARD_REWARDS)


def straight_line_hex_world() -> DiscreteMDP:
    """Seven tiles in a row, with +10 on the rightmost tile."""
    return hex_mdp([(0, c) for c in range(7)], {(0, 6): 10.0})
