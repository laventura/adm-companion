"""The multicaregiver crying baby (ADM Appendix F.14), a two-agent POMG.

A POMG stores T[s, a1, a2, s2], O[a1, a2, s2, o1, o2] (the joint
observation's probability given the joint action and the new state) and
R[i, s, a1, a2]. Mirrors julia/src/problems/multicaregiver.jl.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class POMG:
    T: np.ndarray            # T[s, a1, a2, s2]
    O: np.ndarray            # O[a1, a2, s2, o1, o2]
    R: np.ndarray            # R[i, s, a1, a2]
    gamma: float
    states: tuple
    actions: tuple           # actions[i]: agent i's action labels
    observations: tuple      # observations[i]: agent i's observation labels
    name: str = ""


def multicaregiver(gamma=0.9) -> POMG:
    """F.14 as the appendix defines it. States: 0 = sated, 1 = hungry.
    Actions (each caregiver): 0 = feed, 1 = sing, 2 = ignore.
    Observations (each): 0 = crying, 1 = quiet; both caregivers always
    hear the same thing.

    Either caregiver feeding sates the baby. Otherwise a hungry baby stays
    hungry and a sated one stays sated w.p. 0.5. A hungry baby cries w.p.
    0.9; a sated one never cries (whatever the actions: as defined, singing
    is not more informative). Both caregivers pay 10 while the baby is
    hungry; feeding costs caregiver 1 an extra 2.5 and caregiver 2 an extra
    5; singing costs caregiver 1 0.5 and caregiver 2 0.25.
    """
    T = np.zeros((2, 3, 3, 2))
    O = np.zeros((3, 3, 2, 2, 2))
    R = np.zeros((2, 2, 3, 3))
    extra = np.array([[-2.5, -0.5, 0.0],     # caregiver 1: feed, sing, ignore
                      [-5.0, -0.25, 0.0]])   # caregiver 2
    for a1 in range(3):
        for a2 in range(3):
            fed = a1 == 0 or a2 == 0
            T[0, a1, a2] = [1.0, 0.0] if fed else [0.5, 0.5]
            T[1, a1, a2] = [1.0, 0.0] if fed else [0.0, 1.0]
            # Both hear the same thing: only (cry, cry) and (quiet, quiet) occur.
            O[a1, a2, 0, 1, 1] = 1.0                   # sated: always quiet
            O[a1, a2, 1, 0, 0], O[a1, a2, 1, 1, 1] = 0.9, 0.1
            for s in range(2):
                hungry = -10.0 if s == 1 else 0.0
                R[0, s, a1, a2] = hungry + extra[0, a1]
                R[1, s, a1, a2] = hungry + extra[1, a2]
    acts = ("feed", "sing", "ignore")
    obs = ("crying", "quiet")
    return POMG(T, O, R, gamma, ("sated", "hungry"), (acts, acts), (obs, obs),
                "multicaregiver crying baby")
