"""The lost hiker: this site's Dec-POMDP (two rescuers, one shared reward).

A hiker is lost in the left or right valley. Two rescuers at separate
lookouts can each listen (hear a private report, right with probability
0.8), search the left valley or search the right valley. Carrying the hiker
out takes both rescuers: both searching the right valley earns 20; one alone
earns 5; each rescuer searching the wrong valley costs 10; each listening
rescuer costs 1. Any search ends the episode: a new hiker is lost in a
random valley. Reports are uninformative (50-50) after a search.

Stored as a POMG (adm.problems.multicaregiver) whose two reward rows are
the same shared reward. States: 0 = left, 1 = right. Actions: 0 = listen,
1 = search left, 2 = search right. Observations: 0 = "left", 1 = "right".
Mirrors julia/src/problems/lost_hiker.jl.
"""

import numpy as np

from adm.problems.multicaregiver import POMG


def lost_hiker(gamma=0.9, accuracy=0.8, together=20.0, alone=5.0, wrong=-10.0,
               listen=-1.0) -> POMG:
    T = np.zeros((2, 3, 3, 2))
    O = np.zeros((3, 3, 2, 2, 2))
    R = np.zeros((2, 2, 3, 3))
    for a1 in range(3):
        for a2 in range(3):
            listening = a1 == 0 and a2 == 0
            for s in range(2):
                T[s, a1, a2] = np.eye(2)[s] if listening else [0.5, 0.5]
                found = sum(a == s + 1 for a in (a1, a2))       # searching valley s
                missed = sum(a not in (0, s + 1) for a in (a1, a2))
                r = listen * ((a1 == 0) + (a2 == 0)) + wrong * missed
                r += together if found == 2 else alone if found == 1 else 0.0
                R[:, s, a1, a2] = r
            for s2 in range(2):
                # Each rescuer's report is independent: right w.p. accuracy
                # after both listened, 50-50 otherwise.
                p = accuracy if listening else 0.5
                hear = np.array([p, 1 - p]) if s2 == 0 else np.array([1 - p, p])
                O[a1, a2, s2] = np.outer(hear, hear)
    acts = ("listen", "search left", "search right")
    obs = ("left", "right")
    return POMG(T, O, R, gamma, ("left", "right"), (acts, acts), (obs, obs),
                "lost hiker")
