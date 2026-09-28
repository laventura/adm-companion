"""The fork: a POMDP where information has to be paid for (this site's problem).

A hiker at a fork doesn't know which trail leads home. States: 0 = home is
left, 1 = home is right. Actions: 0 = wait (free, learns nothing), 1 = ask a
passerby (cost 1; the answer is right 85% of the time), 2 = go left, 3 = go
right. Going pays +10 on the right trail and -50 on the wrong one; then a new
hike starts with the home trail equally likely either way. Observations:
0 = "left", 1 = "right" (uninformative 50/50 noise after anything but
asking). Discount 0.95. Structured like the tiger problem (L. P. Kaelbling,
M. L. Littman and A. R. Cassandra, 1998), plus a free wait action.
Mirrors julia/src/problems/fork.jl.
"""

import numpy as np

from adm.problems.crying_baby import DiscretePOMDP


def fork(gamma=0.95, p_correct=0.85, reward=10.0, penalty=-50.0, ask_cost=1.0) -> DiscretePOMDP:
    T = np.zeros((2, 4, 2))
    for s in range(2):
        T[s, 0, s] = T[s, 1, s] = 1.0          # waiting and asking change nothing
        T[s, 2, :] = T[s, 3, :] = 0.5          # going ends the hike; a new one starts
    O = np.full((4, 2, 2), 0.5)                # uninformative by default
    O[1, 0] = [p_correct, 1 - p_correct]       # asked, home is left
    O[1, 1] = [1 - p_correct, p_correct]       # asked, home is right
    R = np.array([[0.0, -ask_cost, reward, penalty],
                  [0.0, -ask_cost, penalty, reward]])
    return DiscretePOMDP(T, R, O, gamma, ("home left", "home right"),
                         ("wait", "ask", "go left", "go right"), ("left", "right"))
