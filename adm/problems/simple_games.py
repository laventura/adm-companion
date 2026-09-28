"""Simple games (ADM Appendix F.10-F.12, plus this site's intersection).

A simple game with k agents stores R[i, a1, ..., ak]: agent i's reward when
agent 1 plays a1, ..., agent k plays ak (all 0-based). There are no states:
each round, the agents act at once and are paid. Mirrors
julia/src/problems/simple_games.jl.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimpleGame:
    R: np.ndarray            # R[i, a1, ..., ak]
    actions: tuple           # actions[i]: labels of agent i's actions
    gamma: float = 0.9       # discount, when the game is repeated
    name: str = ""


def two_player(R1, R2, actions1, actions2, name) -> SimpleGame:
    """Stack two payoff tables (rows: agent 1's action, columns: agent 2's)."""
    return SimpleGame(np.array([R1, R2], dtype=float), (tuple(actions1), tuple(actions2)), name=name)


def prisoners_dilemma() -> SimpleGame:
    """F.10: years in prison, as negative rewards. Actions: cooperate, defect."""
    R1 = [[-1, -4],
          [0, -3]]
    return two_player(R1, np.transpose(R1), ("cooperate", "defect"), ("cooperate", "defect"),
                      "prisoner's dilemma")


def rock_paper_scissors() -> SimpleGame:
    """F.11: +1 to the winner, -1 to the loser, 0 for a draw (zero-sum)."""
    R1 = [[0, -1, 1],        # rock vs rock, paper, scissors
          [1, 0, -1],        # paper
          [-1, 1, 0]]        # scissors
    acts = ("rock", "paper", "scissors")
    return two_player(R1, -np.array(R1), acts, acts, "rock-paper-scissors")


def travelers_dilemma(low=2, high=100) -> SimpleGame:
    """F.12: claims low..high. Equal claims pay the claim; otherwise the lower
    claimant gets its claim + 2 and the higher one gets the lower claim - 2."""
    c = np.arange(low, high + 1)
    mine, theirs = np.meshgrid(c, c, indexing="ij")          # mine[a1, a2] = claim of agent 1
    R1 = np.where(mine == theirs, mine, np.where(mine < theirs, mine + 2, theirs - 2))
    return two_player(R1, R1.T, [f"${v}" for v in c], [f"${v}" for v in c], "traveler's dilemma")


def intersection(win=3.0, crash=-10.0, wait=-1.0) -> SimpleGame:
    """This site's problem: two drivers reach an intersection at once.

    Actions: go, stop. If one goes and the other stops, the driver who goes
    gains `win` and the other gets 0. Both going is a near-crash (`crash`
    each); both stopping wastes time (`wait` each). A game of chicken: a
    traffic light (a correlated equilibrium) beats every Nash equilibrium.
    """
    R1 = [[crash, win],      # go vs go, stop
          [0.0, wait]]       # stop vs go, stop
    return two_player(R1, np.transpose(R1), ("go", "stop"), ("go", "stop"), "intersection")
