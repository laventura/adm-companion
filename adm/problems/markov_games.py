"""Markov games for Part V: the predator-prey hex world (ADM Appendix F.13)
and this site's lake. Mirrors julia/src/problems/markov_games.jl.

A two-agent Markov game stores T[s, a1, a2, s2] = T(s2 | s, a1, a2) and
R[i, s, a1, a2], agent i's reward, with 0-based indices.
"""

from dataclasses import dataclass

import numpy as np

from adm.problems.hex_world import DIRECTIONS, neighbor


@dataclass(frozen=True)
class MarkovGame:
    T: np.ndarray            # T[s, a1, a2, s2]
    R: np.ndarray            # R[i, s, a1, a2]
    gamma: float
    states: tuple
    actions: tuple           # actions[i]: labels of agent i's actions
    name: str = ""


# The map of the book's figure F.16, in hex_world's odd-r coordinates
# (odd rows sit half a tile to the right).
PREDATOR_PREY_TILES = ((1, 0), (1, 1), (1, 2),
                       (2, 0), (2, 2), (2, 4), (2, 5),
                       (3, 0), (3, 1), (3, 2), (3, 3), (3, 4))
PREDATOR_START, PREY_START = (1, 2), (3, 2)


def hex_moves(tiles, p_intended=0.7):
    """M[x, a, y]: probability that an agent on tile x choosing direction a
    ends on tile y. As in Hex World: the chosen direction w.p. 0.7, each
    neighboring direction w.p. 0.15; a move off the map stays put."""
    index = {t: k for k, t in enumerate(tiles)}
    n, nA = len(tiles), len(DIRECTIONS)
    M = np.zeros((n, nA, n))
    for x, tile in enumerate(tiles):
        for a in range(nA):
            for da, p in ((0, p_intended), (-1, (1 - p_intended) / 2), (1, (1 - p_intended) / 2)):
                target = neighbor(tile, DIRECTIONS[(a + da) % nA])
                M[x, a, index.get(target, x)] += p
    return M


def predator_prey(gamma=0.9, move_cost=1.0, catch=1.0, eaten=-100.0) -> MarkovGame:
    """F.13 with one predator (agent 0) and one prey (agent 1).

    State (x, y) = (predator's tile, prey's tile), index x * n + y. Each step
    both move as in Hex World. When they share a tile, the predator is paid
    `catch` and the prey `eaten`, and the prey reappears on a uniformly random
    tile while the predator moves on. The predator pays `move_cost` every step.
    """
    tiles = PREDATOR_PREY_TILES
    n, nA = len(tiles), len(DIRECTIONS)
    M = hex_moves(tiles)
    T = np.zeros((n * n, nA, nA, n * n))
    R = np.zeros((2, n * n, nA, nA))
    uniform = np.full(n, 1 / n)
    for x in range(n):
        for y in range(n):
            s = x * n + y
            caught = x == y
            R[0, s] = -move_cost + (catch if caught else 0.0)
            R[1, s] = eaten if caught else 0.0
            for a1 in range(nA):
                for a2 in range(nA):
                    prey_next = uniform if caught else M[y, a2]
                    # Independent moves: outer product of the two agents' next tiles.
                    T[s, a1, a2] = np.outer(M[x, a1], prey_next).ravel()
    states = tuple((tx, ty) for tx in tiles for ty in tiles)
    return MarkovGame(T, R, gamma, states, (DIRECTIONS, DIRECTIONS), "predator-prey hex world")


def lake(gamma=0.9) -> MarkovGame:
    """This site's problem: two boats fish a shared lake (a tragedy of the commons).

    Stock: 0 = low, 1 = medium, 2 = high. Each boat fishes lightly (action 0)
    or heavily (1). A boat's catch depends on the stock and its own effort:
    light 1, 2, 3 and heavy 2, 3.5, 5 (low, medium, high). The stock moves
    with the number of heavy boats h: h = 0: up w.p. 0.7, else same;
    h = 1: up 0.2, same 0.5, down 0.3; h = 2: down 0.7, same 0.3 (a stock
    can't rise above high or fall below low).
    """
    catch = np.array([[1.0, 2.0], [2.0, 3.5], [3.0, 5.0]])     # catch[s, effort]
    moves = {0: {+1: 0.7, 0: 0.3}, 1: {+1: 0.2, 0: 0.5, -1: 0.3}, 2: {-1: 0.7, 0: 0.3}}
    T = np.zeros((3, 2, 2, 3))
    R = np.zeros((2, 3, 2, 2))
    for s in range(3):
        for a1 in range(2):
            for a2 in range(2):
                R[0, s, a1, a2] = catch[s, a1]
                R[1, s, a1, a2] = catch[s, a2]
                for step, p in moves[a1 + a2].items():
                    T[s, a1, a2, min(max(s + step, 0), 2)] += p
    return MarkovGame(T, R, gamma, ("low", "medium", "high"),
                      (("light", "heavy"), ("light", "heavy")), "lake")
