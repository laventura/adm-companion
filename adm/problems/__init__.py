"""Problem definitions shared across chapters (the Appendix F gallery).

Every finite MDP is returned as a DiscreteMDP, so any solver that takes
(T, R, gamma) works on any problem.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscreteMDP:
    """A finite MDP: T[s, a, s2] = T(s2 | s, a) and R[s, a] = R(s, a).

    `states` and `actions` hold a readable label for each index.
    """

    T: np.ndarray
    R: np.ndarray
    gamma: float
    states: list
    actions: list
