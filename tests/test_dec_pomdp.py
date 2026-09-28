"""Tests for adm.dec_pomdp (Ch 27), pinning the numbers quoted on the page."""

import itertools

import numpy as np

from adm import dec_pomdp as D
from adm import pomg as P
from adm.problems.lost_hiker import lost_hiker
from adm.problems.multicaregiver import multicaregiver

B = np.array([0.5, 0.5])


def test_model_is_normalized_and_shared():
    g = lost_hiker()
    assert np.allclose(g.T.sum(-1), 1) and np.allclose(g.O.sum(axis=(-2, -1)), 1)
    assert np.array_equal(g.R[0], g.R[1])


def test_exact_optimum_matches_enumeration():
    g = lost_hiker()
    U = P.plan_evaluator(g)
    plans = P.all_plans(3, 2, 2)
    brute = max(P.utility(U, B, p, q)[0] for p in plans for q in plans)
    p1, p2, v = D.exact_best_plans(g, B, 2, U)
    assert np.isclose(v, brute) and np.isclose(v, 7.36)
    p1, p2, v = D.exact_best_plans(g, B, 3, U)
    assert np.isclose(v, P.utility(U, B, p1, p2)[0]) and np.isclose(v, 7.36)
    assert D.exact_best_plans(g, B, 1, U)[2] == 0.0       # one step: nothing to gain


def test_communication_bound():
    g = lost_hiker()
    assert np.isclose(D.centralized_value(g, B, 2), 8.8)
    assert D.centralized_value(g, B, 3) > 11 > 7.36


def test_jesp_and_mbdp_get_stuck_sometimes():
    g = lost_hiker()
    U = P.plan_evaluator(g)
    full = P.all_plans(3, 2, 3)
    vals = [D.iterated_best_response(g, B, [full, full], np.random.default_rng(s), U=U)[2]
            for s in range(20)]
    assert max(vals) <= 7.36 + 1e-9 and min(vals) < 1 and np.isclose(max(vals), 7.36)
    vals = [D.memory_bounded_search(g, B, 3, 3, np.random.default_rng(s), U)[2] for s in range(10)]
    assert all(v <= 7.36 + 1e-9 for v in vals)


def test_team_caregivers_match_the_selfish_equilibrium():
    team = D.shared(multicaregiver())
    assert np.isclose(D.exact_best_plans(team, B, 3)[2], -14.75)


def test_three_node_controllers():
    g = lost_hiker()

    def ctrl(acts, nxt, n):
        psi, eta = np.zeros((n, 3)), np.zeros((n, 3, 2, n))
        for x in range(n):
            psi[x, acts[x]] = 1
            for o in range(2):
                eta[x, :, o, nxt[2 * x + o]] = 1
        return psi, eta

    c = ctrl((0, 1, 2), (1, 2, 0, 0, 0, 0), 3)       # listen; search where told; listen
    assert np.isclose(P.evaluate_controllers(g, c, c)[0, 0, 0] @ B, 38.737, atol=1e-3)
    best2 = max(P.evaluate_controllers(g, c2, c2)[0, 0, 0] @ B
                for c2 in (ctrl(a, n, 2) for a in itertools.product(range(3), repeat=2)
                           for n in itertools.product(range(2), repeat=4)))
    assert np.isclose(best2, 0.0)
