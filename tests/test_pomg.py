"""Tests for adm.pomg (Ch 26), pinning the numbers quoted on the page."""

import itertools

import numpy as np

from adm import pomg as P
from adm.problems.multicaregiver import multicaregiver

B = np.array([0.5, 0.5])


def payoffs(g, plans, U):
    sg = P.plan_game(g, B, plans[0], plans[1], U)
    return sorted({tuple(np.round(sg.R[:, a, c], 4)) for a, c in P.pure_equilibria(sg)})


def test_model_is_normalized():
    g = multicaregiver()
    assert np.allclose(g.T.sum(-1), 1) and np.allclose(g.O.sum(axis=(-2, -1)), 1)


def test_plan_evaluation_matches_simulation():
    g, U, rng = multicaregiver(), None, np.random.default_rng(0)
    U = P.plan_evaluator(g)
    p1 = (0, ((0, ((2,), (2,))), (2, ((0,), (2,)))))      # feed; then depends on crying
    p2 = (2, ((2, ((2,), (2,))), (2, ((2,), (2,)))))      # always ignore
    rets = []
    for _ in range(20000):
        s, plans, ret = rng.choice(2, p=B), (p1, p2), np.zeros(2)
        for t in range(3):
            a1, a2 = plans[0][0], plans[1][0]
            ret += g.gamma ** t * g.R[:, s, a1, a2]
            s2 = rng.choice(2, p=g.T[s, a1, a2])
            o = divmod(rng.choice(4, p=g.O[a1, a2, s2].ravel()), 2)
            if t < 2:
                plans = (plans[0][1][o[0]], plans[1][1][o[1]])
            s = s2
        rets.append(ret)
    assert np.allclose(np.mean(rets, axis=0), P.utility(U, B, p1, p2), atol=0.1)


def test_pruning_keeps_the_equilibria():
    g = multicaregiver()
    U = P.plan_evaluator(g)
    full = P.all_plans(3, 2, 2)
    plans, counts = P.dynamic_programming(g, 2, np.random.default_rng(0), U)
    assert counts == [((3, 3), (1, 1)), ((3, 3), (2, 2))]
    assert payoffs(g, [full, full], U) == payoffs(g, plans, U) == [(-7.5, -5.0), (-5.0, -10.0)]
    plans, counts = P.dynamic_programming(g, 4, np.random.default_rng(0), U)
    assert [c[1] for c in counts] == [(1, 1), (2, 2), (4, 4), (32, 32)]


def test_controllers():
    g = multicaregiver()

    def ctrl(acts, nxt):
        psi, eta = np.zeros((2, 3)), np.zeros((2, 3, 2, 2))
        for x in range(2):
            psi[x, acts[x]] = 1
            eta[x, :, 0, nxt[x][0]] = eta[x, :, 1, nxt[x][1]] = 1
        return psi, eta

    idle = ctrl((2, 2), [[0, 0], [0, 0]])
    always = ctrl((0, 0), [[0, 0], [0, 0]])
    assert np.allclose(P.evaluate_controllers(g, idle, idle)[:, 0, 0] @ B, -10 / 0.11, atol=1e-9)
    assert np.allclose(P.evaluate_controllers(g, always, idle)[:, 0, 0] @ B, [-30, -5])
    # Against a caregiver who always feeds, the other's best 2-node controller never feeds.
    cands = [ctrl(a, [n[:2], n[2:]]) for a in itertools.product(range(3), repeat=2)
             for n in itertools.product(range(2), repeat=4)]
    best = max(P.evaluate_controllers(g, always, c)[1, 0, 0] @ B for c in cands)
    assert np.isclose(best, -5.0)
