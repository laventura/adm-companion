"""Appendix E: search on deterministic problems, where each action leads to
exactly one next state. Depth-limited forward search, branch and bound,
memoized forward search (dynamic programming) and heuristic search.

Every search returns (best action, value, nodes), where nodes counts the
states the search expanded, so the methods can be compared by work done.
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass
class Search:
    actions: Callable     # actions(s): valid actions; empty = absorbing
    T: Callable           # T(s, a): the next state
    R: Callable           # R(s, a): the reward for taking a in s


MOVES = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}


def maze(rows, goal_reward=10.0, step_reward=-1.0):
    """A deterministic maze. rows are strings: '#' wall, 'G' goal, any
    other character a free cell. States are (row, col). Each move costs
    step_reward; entering the goal also pays goal_reward, and the goal is
    absorbing (no actions). Returns (Search, goal, free cells)."""
    cells = {(r, c) for r, row in enumerate(rows)
             for c, ch in enumerate(row) if ch != "#"}
    goal = next((r, c) for r, row in enumerate(rows)
                for c, ch in enumerate(row) if ch == "G")

    def T(s, a):
        return (s[0] + MOVES[a][0], s[1] + MOVES[a][1])

    def actions(s):
        # Only moves into free cells are valid; the goal has none.
        return [] if s == goal else [a for a in MOVES if T(s, a) in cells]

    def R(s, a):
        return step_reward + (goal_reward if T(s, a) == goal else 0.0)

    return Search(actions, T, R), goal, sorted(cells)


def distances(P: Search, goal, cells):
    """Fewest moves from each cell to the goal (breadth-first search
    backward from the goal; moves are reversible in a maze)."""
    dist, queue = {goal: 0}, deque([goal])
    while queue:
        s = queue.popleft()
        for a in MOVES:
            s2 = P.T(s, a)
            if s2 in cells and s2 not in dist:
                dist[s2] = dist[s] + 1
                queue.append(s2)
    return dist


def forward_search(P: Search, s, d, U):
    """Try every action sequence of length d; value U(s) at the leaves."""
    A = P.actions(s)
    if not A or d <= 0:
        return None, U(s), 1
    best, nodes = (None, float("-inf")), 1
    for a in A:
        _, u, n = forward_search(P, P.T(s, a), d - 1, U)
        nodes += n
        if P.R(s, a) + u > best[1]:
            best = (a, P.R(s, a) + u)
    return best[0], best[1], nodes


def branch_and_bound(P: Search, s, d, U_lo, Q_hi):
    """Forward search that tries actions from the most promising upper
    bound Q_hi(s, a) down, and stops as soon as no remaining action's
    bound beats the best value found so far."""
    A = P.actions(s)
    if not A or d <= 0:
        return None, U_lo(s), 1
    best, nodes = (None, float("-inf")), 1
    for a in sorted(A, key=lambda a: -Q_hi(s, a)):
        if Q_hi(s, a) <= best[1]:
            break           # sorted: every later action is bounded lower
        _, u, n = branch_and_bound(P, P.T(s, a), d - 1, U_lo, Q_hi)
        nodes += n
        if P.R(s, a) + u > best[1]:
            best = (a, P.R(s, a) + u)
    return best[0], best[1], nodes


def memoized_search(P: Search, s, d, U, M=None):
    """Forward search that stores each (depth, state) answer in M, so a
    state reached again by another path with the same depth left is
    looked up instead of searched (dynamic programming)."""
    M = {} if M is None else M
    if (d, s) in M:
        return (*M[(d, s)], 0)          # a lookup expands nothing
    A = P.actions(s)
    if not A or d <= 0:
        best, nodes = (None, U(s)), 1
    else:
        best, nodes = (None, float("-inf")), 1
        for a in A:
            _, u, n = memoized_search(P, P.T(s, a), d - 1, U, M)
            nodes += n
            if P.R(s, a) + u > best[1]:
                best = (a, P.R(s, a) + u)
    M[(d, s)] = best
    return best[0], best[1], nodes


def heuristic_search(P: Search, s, d, U_hi, U_lo, M=None):
    """Memoized branch and bound whose action bound is the immediate
    reward plus an optimistic estimate of what follows,
    R(s, a) + U_hi(T(s, a)). U_hi must never underestimate the value."""
    M = {} if M is None else M
    if (d, s) in M:
        return (*M[(d, s)], 0)
    A = P.actions(s)
    if not A or d <= 0:
        best, nodes = (None, U_lo(s)), 1
    else:
        bound = lambda a: P.R(s, a) + U_hi(P.T(s, a))
        best, nodes = (None, float("-inf")), 1
        for a in sorted(A, key=lambda a: -bound(a)):
            if bound(a) <= best[1]:
                break
            _, u, n = heuristic_search(P, P.T(s, a), d - 1, U_hi, U_lo, M)
            nodes += n
            if P.R(s, a) + u > best[1]:
                best = (a, P.R(s, a) + u)
    M[(d, s)] = best
    return best[0], best[1], nodes
