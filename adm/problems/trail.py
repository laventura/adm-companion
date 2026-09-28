"""Mountain trail: a chain MDP where exploration matters (this site's problem).

A hiker stands on one of six trail positions, 0 (the trailhead) to 5 (the
summit). Two actions:
  back:  walk down one position (at the trailhead, rest there). Resting at
         the trailhead pays 0.05 (a snack at the car).
  climb: up one position w.p. 0.35, stay w.p. 0.6, slip down one w.p. 0.05
         (at the trailhead: up 0.35, stay 0.65). At the summit, climbing
         means staying to enjoy the view, which pays 1 (it slips w.p. 0.05).
Discount 0.95. The small, sure reward is next to the start; the large one
needs many unreliable climbs. Structured like the RiverSwim benchmark
(A. L. Strehl and M. L. Littman, J. Comput. Syst. Sci., 2008), with this
site's own numbers. Mirrors julia/src/problems/trail.jl.
"""

import numpy as np

from adm.problems import DiscreteMDP

N_POSITIONS = 6
P_UP, P_SLIP = 0.35, 0.05          # climb: up, slip down; otherwise stay
SNACK, VIEW = 0.05, 1.0            # rewards at the trailhead and the summit
GAMMA = 0.95


def trail(n: int = N_POSITIONS, gamma: float = GAMMA) -> DiscreteMDP:
    """The trail as a DiscreteMDP; actions 0 = back, 1 = climb."""
    T = np.zeros((n, 2, n))
    R = np.zeros((n, 2))
    top = n - 1
    for s in range(n):
        T[s, 0, max(s - 1, 0)] = 1.0              # back: deterministic
        up, down = min(s + 1, top), max(s - 1, 0)
        # += because at the ends two outcomes land on the same position.
        T[s, 1, up] += P_UP
        T[s, 1, down] += P_SLIP
        T[s, 1, s] += 1 - P_UP - P_SLIP
    R[0, 0] = SNACK
    R[top, 1] = VIEW
    return DiscreteMDP(T, R, gamma, list(range(n)), ["back", "climb"])
