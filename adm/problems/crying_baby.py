"""Crying baby (ADM Appendix F.7), implemented from its description.

A caregiver can't see whether the baby is hungry, only whether it cries.
States: 0 = sated, 1 = hungry. Actions: 0 = feed, 1 = sing, 2 = ignore.
Observations: 0 = crying, 1 = quiet.

Arrays: T[s, a, s2] = T(s2 | s, a), R[s, a], O[a, s2, o] = O(o | a, s2)
(the observation depends on the action and the state it led to).
Mirrors julia/src/problems/crying_baby.jl.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscretePOMDP:
    T: np.ndarray
    R: np.ndarray
    O: np.ndarray
    gamma: float
    states: tuple
    actions: tuple
    observations: tuple


def crying_baby(gamma: float = 0.9) -> DiscretePOMDP:
    T = np.zeros((2, 3, 2))
    T[:, 0, 0] = 1.0                       # feeding always sates
    for a in (1, 2):                       # singing and ignoring move the same way
        T[0, a] = [0.9, 0.1]               # a sated baby gets hungry 10% of the time
        T[1, a, 1] = 1.0                   # a hungry baby stays hungry
    p_cry = np.array([[0.1, 0.8],          # feed:   P(crying | sated), P(crying | hungry)
                      [0.0, 0.9],          # sing
                      [0.1, 0.8]])         # ignore
    O = np.stack([p_cry, 1 - p_cry], axis=-1)       # O[a, s2, o]
    # -10 while hungry (the state before the transition), -5 to feed, -0.5 to sing.
    R = -10.0 * np.array([[0.0], [1.0]]) + np.array([[-5.0, -0.5, 0.0]])
    return DiscretePOMDP(T, R, O, gamma, ("sated", "hungry"), ("feed", "sing", "ignore"),
                         ("crying", "quiet"))


def belief_update(p: DiscretePOMDP, b, a, o):
    """b'(s2) proportional to O(o | a, s2) sum_s T(s2 | s, a) b(s) (Ch 19)."""
    b2 = p.O[a, :, o] * (b @ p.T[:, a, :])
    return b2 / b2.sum()
