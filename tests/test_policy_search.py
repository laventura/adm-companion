"""Policy search (Ch 10) and the simple regulator problem."""

import numpy as np
import pytest

from adm.policy_search import (
    cross_entropy, evolution_strategies, genetic_search, hooke_jeeves,
    isotropic_evolution_strategies, rank_weights,
)
from adm.problems import simple_regulator as sr

U_STAR = sr.exact_value([-1.0, 0.0])            # theta_1 = -1 cancels the state


def test_regulator_exact_value_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for theta in ([0.0, 1.0], [-0.5, 0.3], [-1.0, 0.0]):
        mc = sr.rollout_returns(np.array(theta), 200_000, rng).mean()
        assert mc == pytest.approx(sr.exact_value(theta), rel=0.01)
    # E[s_1^2] = 0.1, then 0.01 per step: -(0.1 + 0.01 (0.9 + ... + 0.9^9))
    assert U_STAR == pytest.approx(-(0.1 + 0.01 * sum(0.9**t
                                                      for t in range(1, 10))),
                                   abs=1e-6)


def test_rank_weights():
    w = rank_weights(4)
    assert w.sum() == pytest.approx(0) and np.all(np.diff(w) <= 0)
    raw = np.log(3) - np.log([1, 2])
    assert w[:2] == pytest.approx(raw / raw.sum() - 0.25)
    assert w[2] == w[3] == pytest.approx(-0.25)


def test_hooke_jeeves_climbs_a_concave_quadratic():
    U = lambda t: -(t[0] - 1.3) ** 2 - 2 * (t[1] + 0.4) ** 2
    trace = []
    theta = hooke_jeeves(U, [0.0, 0.0], 1.0, 0.5, 1e-4, trace)
    assert theta == pytest.approx([1.3, -0.4], abs=1e-3)
    assert trace[-1][1] <= 1e-4 * 2


@pytest.mark.parametrize("method", ["hooke", "genetic", "cem", "es", "iso"])
def test_methods_reach_the_regulator_optimum(method):
    rng = np.random.default_rng(1)
    U = sr.exact_value                          # noise-free objective
    if method == "hooke":
        theta = hooke_jeeves(U, [0.0, 1.0], 0.75, 0.75, 0.001)
    elif method == "genetic":
        theta = genetic_search(U, rng.normal([0, 3], 1.0, (25, 2)), 0.1, 5,
                               30, rng)
    elif method == "cem":
        # With fewer samples the spread can collapse before reaching the
        # optimum (25 samples: 3 of 20 seeds succeed; 200: all of them).
        theta = cross_entropy(U, [0, 3], 2 * np.eye(2), 200, 40, 15, rng)[0]
    elif method == "es":
        theta = evolution_strategies(U, [0, 3], np.log([1.0, 1.0]), 10, 0.5,
                                     60, rng)[0]
    else:
        theta = isotropic_evolution_strategies(U, [0, 3], 0.25, 10, 0.02,
                                               60, rng)
    assert sr.exact_value(theta) > U_STAR - 0.01


def test_cross_entropy_trace_shrinks_the_spread():
    trace = []
    rng = np.random.default_rng(2)
    cross_entropy(sr.exact_value, [0, 3], 2 * np.eye(2), 25, 5, 8, rng,
                  trace)
    spreads = [np.trace(cov) for _, cov, _, _ in trace]
    assert spreads[-1] < spreads[0] / 100      # 5 of 25 elites: fast shrink


def test_mountain_car_switching_rule():
    from adm.problems import mountain_car as mc
    s = np.array([[-0.5, 0.01], [-0.5, -0.01], [0.0, 0.0]])
    assert mc.switching_rule([0.0, 0.0], s).tolist() == [1, -1, -1]
    assert mc.switching_rule([0.0, 0.001], s[2:])[0] == 1
    # "Push with the motion" from the ten standard starts (the page quotes
    # 152.8 with a 200-step cap; two cars escape at 202 and 210 steps).
    steps = mc.steps_to_goal(lambda x: mc.switching_rule([0.0, 0.0], x),
                             mc.STANDARD_STARTS, 200)
    assert steps.tolist() == [188, 192, 197, 201, 201, 171, 103, 95, 91, 89]


def test_genetic_search_keeps_the_best():
    rng = np.random.default_rng(3)
    trace = []
    genetic_search(sr.exact_value, rng.normal([0, 3], 1.0, (10, 2)), 0.2, 3,
                   5, rng, trace)
    best = [max(sr.exact_value(t) for t in pop) for pop, _ in trace]
    assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:]))
