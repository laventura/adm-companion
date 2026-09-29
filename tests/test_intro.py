"""Tests for adm.intro (Ch 1), pinning the numbers quoted on the page."""

import numpy as np

from adm import imitation as im
from adm import intro
from adm.mdp import policy_iteration
from adm.problems.hex_world import STANDARD_REWARDS, STANDARD_TILES, hex_world

H = hex_world()
N = len(H.R)
START = [i for i, t in enumerate(STANDARD_TILES) if t not in STANDARD_REWARDS]
B = np.zeros(N)
B[START] = 1 / len(START)


def value(policy):
    return im.policy_value(H.T, H.R, H.gamma, np.eye(6)[policy], B)


def test_explicit_rule_is_worse_than_planning():
    opt, _, _ = policy_iteration(H.T, H.R, H.gamma, np.zeros(N, int))
    rule = intro.head_for(STANDARD_TILES, (2, 9))
    assert np.isclose(value(opt), 4.526, atol=1e-3)
    assert np.isclose(value(rule), 2.766, atol=1e-3)


def test_simulated_return_matches_exact_value():
    opt, _, _ = policy_iteration(H.T, H.R, H.gamma, np.zeros(N, int))
    u = intro.simulated_return(H, opt, B, seed=0, n=4000, depth=80)
    assert abs(u - value(opt)) < 0.3


def test_local_search_improves():
    policy, history = intro.local_search(H, B, np.random.default_rng(0), proposals=300)
    assert history[-1] > history[0] and all(np.diff(history) >= 0)
    assert value(policy) > 4.0
