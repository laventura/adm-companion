"""Partially observable Markov games (ADM Ch. 26): plans, equilibria, DP.

A two-agent POMG stores T[s, a1, a2, s2], O[a1, a2, s2, o1, o2] and
R[i, s, a1, a2] (adm.problems.multicaregiver). A conditional plan is a
nested tuple: (a,) for a last step, or (a, (plan after o = 0, plan after
o = 1, ...)). Tuples are hashable, so plan values can be cached.
"""

from functools import lru_cache
from itertools import product

import numpy as np
from scipy.optimize import linprog

from adm.problems.simple_games import SimpleGame

Array = np.ndarray


def all_plans(n_actions, n_obs, horizon) -> list:
    """Every conditional plan of the given horizon: an action, then a
    (horizon - 1)-step plan for each observation."""
    plans = [(a,) for a in range(n_actions)]
    for _ in range(horizon - 1):
        plans = expand(plans, n_actions, n_obs)
    return plans


def expand(plans, n_actions, n_obs) -> list:
    """All plans one step longer: any first action, then any plan from
    `plans` after each observation (|A| |plans|^|O| of them)."""
    return [(a, subs) for a in range(n_actions)
            for subs in product(plans, repeat=n_obs)]


def plan_evaluator(game):
    """Returns U(p1, p2, s): both agents' utilities when they follow plans
    p1 and p2 from state s (eq. 26.1), cached across calls."""
    nS = game.T.shape[0]

    @lru_cache(maxsize=None)
    def U(p1, p2, s) -> tuple:
        a1, a2 = p1[0], p2[0]
        r = game.R[:, s, a1, a2].copy()
        if len(p1) == 1:                   # last step: reward only
            return tuple(r)
        n1, n2 = game.O.shape[3:]
        for s2 in range(nS):
            p = game.T[s, a1, a2, s2]
            if p == 0:
                continue
            for o1 in range(n1):
                for o2 in range(n2):
                    q = game.O[a1, a2, s2, o1, o2]
                    if q > 0:              # each agent follows its own branch
                        sub = U(p1[1][o1], p2[1][o2], s2)
                        r += game.gamma * p * q * np.array(sub)
        return tuple(r)

    return U


def utility(U, b, p1, p2) -> Array:
    """eq. 26.2: both agents' utilities from the state distribution b."""
    return sum(b[s] * np.array(U(p1, p2, s)) for s in range(len(b)))


def plan_game(game, b, plans1, plans2, U=None) -> SimpleGame:
    """§26.3: the simple game whose actions are whole conditional plans and
    whose rewards are their utilities from b."""
    U = plan_evaluator(game) if U is None else U
    R = np.zeros((2, len(plans1), len(plans2)))
    for k1, p1 in enumerate(plans1):
        for k2, p2 in enumerate(plans2):
            R[:, k1, k2] = utility(U, b, p1, p2)
    labels = (tuple(map(str, plans1)), tuple(map(str, plans2)))
    return SimpleGame(R, labels)


def pure_equilibria(g) -> list:
    """Every pure Nash equilibrium of a two-player simple game: each row
    is a best response to its column and vice versa (ties allowed)."""
    A, B = g.R
    best1 = A >= A.max(axis=0, keepdims=True) - 1e-9   # agent 1 can't gain
    best2 = B >= B.max(axis=1, keepdims=True) - 1e-9   # agent 2 can't gain
    return [tuple(map(int, k)) for k in np.argwhere(best1 & best2)]


def is_dominated(game, plans, i, k, U) -> bool:
    """eq. 26.6: is agent i's plan plans[i][k] dominated? Look for a
    distribution over (the other's plan, state) under which plan k beats
    every alternative by delta; if the best delta is negative, no such
    distribution exists, and the plan can go."""
    mine, theirs = plans[i], plans[1 - i]
    nS = game.T.shape[0]

    def u(pi, pj, s):                      # agent i's value, either role
        return U(pi, pj, s)[0] if i == 0 else U(pj, pi, s)[1]

    # Variables: b over (other plan, state) pairs, then delta.
    pairs = [(pj, s) for pj in theirs for s in range(nS)]
    n = len(pairs)
    rows = []
    for alt in mine:
        if alt == mine[k]:
            continue
        # delta - sum_b b (u(k) - u(alt)) <= 0
        diff = [u(mine[k], pj, s) - u(alt, pj, s) for pj, s in pairs]
        rows.append(np.concatenate([-np.array(diff), [1.0]]))
    if not rows:
        return False
    res = linprog(np.concatenate([np.zeros(n), [-1.0]]),     # maximize delta
                  A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                  A_eq=np.concatenate([np.ones(n), [0.0]])[None, :],
                  b_eq=[1.0], bounds=[(0, None)] * n + [(None, None)],
                  method="highs")
    return -res.fun < -1e-9                # the best delta is negative


def prune_dominated(game, plans, U, rng) -> list:
    """Remove dominated plans, agent by agent in random order, until a
    full pass removes nothing (§26.4)."""
    plans = [list(p) for p in plans]
    changed = True
    while changed:
        changed = False
        for i in rng.permutation(2):
            for k in rng.permutation(len(plans[i])):
                if len(plans[i]) > 1 and is_dominated(game, plans, i, k, U):
                    del plans[i][k]
                    changed = True
                    break
    return plans


def dynamic_programming(game, horizon, rng, U=None):
    """Algorithm 26.4's loop: start from one-step plans; expand and prune,
    horizon - 1 times. Returns the surviving plans and, for each horizon,
    how many plans each agent had before and after pruning."""
    U = plan_evaluator(game) if U is None else U
    nA = game.T.shape[1:3]
    nO = game.O.shape[3:]
    plans = [[(a,) for a in range(nA[i])] for i in range(2)]
    plans = prune_dominated(game, plans, U, rng)
    counts = [(tuple(nA), tuple(map(len, plans)))]
    for _ in range(horizon - 1):
        grown = [expand(plans[i], nA[i], nO[i]) for i in range(2)]
        plans = prune_dominated(game, grown, U, rng)
        counts.append((tuple(map(len, grown)), tuple(map(len, plans))))
    return plans, counts


def evaluate_controllers(game, c1, c2) -> Array:
    """eq. 26.3 by one linear solve. Each controller is (psi, eta) with
    psi[x, a] and eta[x, a, o, x2], as in Ch. 23. Returns
    U[x1, x2, s]: the joint value of each pair of nodes in each state."""
    (psi1, eta1), (psi2, eta2) = c1, c2
    n1, n2, nS = len(psi1), len(psi2), game.T.shape[0]
    idx = lambda x1, x2, s: (x1 * n2 + x2) * nS + s     # flatten (x1, x2, s)
    m = n1 * n2 * nS
    M, r = np.eye(m), np.zeros((2, m))
    for x1, x2, s in product(range(n1), range(n2), range(nS)):
        i = idx(x1, x2, s)
        for a1, a2 in product(range(psi1.shape[1]), range(psi2.shape[1])):
            w = psi1[x1, a1] * psi2[x2, a2]    # both act independently
            if w == 0:
                continue
            r[:, i] += w * game.R[:, s, a1, a2]
            for s2, o1, o2 in product(range(nS), *map(range, game.O.shape[3:])):
                p = w * game.T[s, a1, a2, s2] * game.O[a1, a2, s2, o1, o2]
                if p == 0:
                    continue
                # Each agent moves its own node on its own observation.
                nxt = np.outer(eta1[x1, a1, o1], eta2[x2, a2, o2])
                for y1, y2 in zip(*np.nonzero(nxt)):
                    M[i, idx(y1, y2, s2)] -= game.gamma * p * nxt[y1, y2]
    U = np.linalg.solve(M, r.T).T
    return U.reshape(2, n1, n2, nS)
