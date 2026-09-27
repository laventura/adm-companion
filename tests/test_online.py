"""Online planning (Ch 9) on Hex World, mountain car and a point robot."""

import numpy as np
import pytest

from adm.mdp import lookahead as q_table
from adm.mdp import policy_evaluation_exact, policy_iteration
from adm.online import (
    MonteCarloTreeSearch, branch_and_bound, deterministic_forward_search,
    double_integrator, forward_search, heuristic_search,
    labeled_heuristic_search, mpc_plan, rollout, rollout_lookahead,
    sparse_sampling,
)
from adm.problems import mountain_car as mc
from adm.problems.hex_world import (
    STANDARD_REWARDS, STANDARD_TILES, hex_world, optimistic_values,
)
from adm.value_approx import (
    MultilinearValue, approximate_value_iteration, greedy_actions,
)

MDP = hex_world()
PI_STAR, U_STAR, _ = policy_iteration(MDP.T, MDP.R, MDP.gamma,
                                      np.zeros(len(MDP.T), int))
Q_STAR = q_table(MDP.T, MDP.R, MDP.gamma, U_STAR)
U_HI = optimistic_values(STANDARD_TILES, STANDARD_REWARDS, MDP.gamma)
Q_HI = MDP.R + MDP.gamma * MDP.T @ U_HI
U_LO = policy_evaluation_exact(MDP.T, MDP.R, MDP.gamma, Q_HI.argmax(axis=1))
DECISIONS = [i for i, t in enumerate(STANDARD_TILES)
             if t not in STANDARD_REWARDS]


def test_bounds_bracket_the_optimum():
    assert np.all(U_HI >= U_STAR - 1e-12) and np.all(U_LO <= U_STAR + 1e-12)
    assert U_HI[STANDARD_TILES.index((2, 9))] == 10     # on the +10 tile
    # (2, 8) isn't a tile, so (2, 7) is 3 moves from the +10 tile.
    assert U_HI[STANDARD_TILES.index((2, 7))] == pytest.approx(10 * 0.9**3)


def test_forward_search_with_exact_leaves_is_optimal():
    for s in DECISIONS:
        a, u = forward_search(MDP, s, 1, lambda s2: U_STAR[s2])
        assert a == PI_STAR[s] and u == pytest.approx(U_STAR[s])


def test_branch_and_bound_matches_forward_search_with_fewer_leaves():
    for d in (1, 2, 3):
        calls = {"fs": 0, "bb": 0}

        def leaf(name):
            def U(s):
                calls[name] += 1
                return U_LO[s]
            return U
        for s in DECISIONS:
            a_fs, u_fs = forward_search(MDP, s, d, leaf("fs"))
            a_bb, u_bb = branch_and_bound(MDP, s, d, leaf("bb"),
                                          lambda s, a: Q_HI[s, a])
            assert a_bb == a_fs and u_bb == pytest.approx(u_fs)
        assert calls["bb"] < calls["fs"]


def test_sparse_sampling_approaches_forward_search():
    rng = np.random.default_rng(0)
    s = STANDARD_TILES.index((1, 8))
    _, u_fs = forward_search(MDP, s, 1, lambda s2: U_LO[s2])
    _, u_ss = sparse_sampling(MDP, s, 1, 400, lambda s2: U_LO[s2], rng)
    assert u_ss == pytest.approx(u_fs, abs=0.1)


def test_rollouts():
    rng = np.random.default_rng(1)
    east = MDP.actions.index("E")
    s = STANDARD_TILES.index((2, 9))              # on the +10 tile
    assert rollout(MDP, s, lambda s, rng: east, 5, rng) == 10.0
    a = rollout_lookahead(MDP, s, lambda s, rng: east, 3, 2, rng)
    assert 0 <= a < 6


def test_mcts_finds_the_optimal_action():
    s = STANDARD_TILES.index((0, 5))     # best action leads by 1.0 in Q*
    planner = MonteCarloTreeSearch(MDP, 10, 2000, 10.0,
                                   lambda s2: U_LO[s2],
                                   np.random.default_rng(2))
    assert planner(s) == PI_STAR[s]
    # The first run only adds the root; paths can also loop back to s (the
    # "tree" is really a graph), so the root is counted at least m - 1 times.
    assert planner.N[s].sum() >= 2000 - 1


def test_heuristic_search_stays_optimistic_and_converges():
    rng = np.random.default_rng(3)
    for s in DECISIONS:
        a, U = heuristic_search(MDP, s, U_HI, 10, 50, rng)
        assert a == PI_STAR[s]
        assert np.all(U >= U_STAR - 1e-9)   # an upper bound stays one


def test_labeled_heuristic_search_solves():
    rng = np.random.default_rng(4)
    for s in DECISIONS:
        a, U, sims = labeled_heuristic_search(MDP, s, U_HI, 10, 0.01, rng)
        assert a == PI_STAR[s] and sims < 100
        assert U[s] == pytest.approx(U_STAR[s], abs=0.1)


def test_deterministic_forward_search_matches_greedy_at_depth_1():
    lo, hi = [-1.2, -0.07], [0.6, 0.07]
    grid = MultilinearValue(lo, hi, (11, 11))
    approximate_value_iteration(grid, grid.points(), mc.step, mc.ACTIONS,
                                -1.0, 1.0, mc.is_goal, 200)
    s = np.array([-0.5, 0.0])
    a, _, count = deterministic_forward_search(mc.step, mc.is_goal,
                                               mc.ACTIONS, grid, s, 1)
    assert count == 3
    assert a == greedy_actions(grid, s[None], mc.step, mc.ACTIONS, -1.0,
                               1.0, mc.is_goal)[0]
    # With no value beyond the horizon, every plan looks the same.
    _, u, count = deterministic_forward_search(
        mc.step, mc.is_goal, mc.ACTIONS, lambda s: np.zeros(len(s)), s, 4)
    assert u == -4 and count == 81


def test_mpc_plan_avoids_the_obstacle_and_reaches_the_goal():
    s0, goal = np.zeros(4), np.array([8.0, 6.0, 0.0, 0.0])
    centers = np.repeat([[4.0, 3.0]], 12, axis=0)
    a = mpc_plan(s0, goal, centers, 1.5, 0.5)
    p, v = double_integrator(s0, a)
    assert np.all(np.abs(a) <= 0.5 + 1e-9)
    assert np.min(np.linalg.norm(p - centers, axis=1)) >= 1.5 - 1e-6
    assert np.linalg.norm(p[-1] - goal[:2]) < 0.1
