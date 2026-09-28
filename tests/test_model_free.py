"""Tests for adm/model_free.py (Ch 17)."""

import numpy as np
import pytest

from adm import model_free as mf
from adm.mdp import policy_iteration, policy_evaluation_exact
from adm.problems import mountain_car as mc
from adm.problems.trail import trail
from adm.value_approx import fourier_basis

TRAIL = trail()


def test_incremental_mean_by_hand():
    xs = [12, 6, 9, 3]
    assert np.allclose(mf.incremental_mean(xs, lambda m: 1 / m), [12, 9, 9, 7.5])
    assert np.allclose(mf.incremental_mean(xs, lambda m: 0.5), [6, 6, 7.5, 5.25])


def test_q_learning_and_sarsa_updates_by_hand():
    q, s = mf.QLearning(2, 3, 0.9, 0.2), mf.Sarsa(2, 3, 0.9, 0.2)
    for learner in (q, s):
        learner.Q[0, 0] = 5.0
        learner.Q[1] = [1.0, 4.0, 3.0]
        learner.update(0, 0, 2.0, 1, 2)            # next action a' = 2 (the third)
    assert q.Q[0, 0] == pytest.approx(5.12)          # target 2 + 0.9 * max = 5.6
    assert s.Q[0, 0] == pytest.approx(4.94)          # target 2 + 0.9 * Q[1, 2] = 4.7


def test_sarsa_lambda_spreads_credit_along_the_path():
    L = mf.SarsaLambda(4, 1, 0.9, 0.1, 0.8)
    L.start()
    L.update(0, 0, 0.0, 1, 0)
    L.update(1, 0, 0.0, 2, 0)
    L.update(2, 0, 5.0, 3, 0)                        # surprise 5 at the third step
    assert np.allclose(L.Q[:3, 0], 0.1 * 5 * np.array([0.72**2, 0.72, 1]))


def test_q_learning_converges_on_the_trail_with_optimism():
    pi_star, U_star, _ = policy_iteration(TRAIL.T, TRAIL.R, TRAIL.gamma, np.zeros(6, int))
    L = mf.QLearning(6, 2, TRAIL.gamma, 0.1, q0=20)
    mf.run_tabular(TRAIL, L, 20_000, 0, np.random.default_rng(0), epsilon=0.1)
    assert L.Q.argmax(axis=1).tolist() == pi_star.tolist()


def test_plain_epsilon_greedy_gets_stuck_on_the_trail():
    L = mf.QLearning(6, 2, TRAIL.gamma, 0.1)
    mf.run_tabular(TRAIL, L, 5000, 0, np.random.default_rng(1), epsilon=0.1)
    assert L.Q.argmax(axis=1)[0] == 0                # still prefers resting


def test_potential_shaping_keeps_the_optimum_and_bonuses_do_not():
    pos = np.arange(6.0)
    T, R, g = TRAIL.T, TRAIL.R, TRAIL.gamma

    def shaped_policy(F):
        Rs = R + (T * F[:, None, :]).sum(axis=2)
        return policy_iteration(T, Rs, g, np.zeros(6, int))[0]
    pi_star = policy_iteration(T, R, g, np.zeros(6, int))[0]
    assert np.array_equal(shaped_policy(g * pos[None, :] - pos[:, None]), pi_star)
    bonus = shaped_policy(5.0 * (pos[None, :] > pos[:, None]))
    assert not np.array_equal(bonus, pi_star)
    assert policy_evaluation_exact(T, R, g, bonus)[0] < 1.0


def test_potential_shaping_function():
    F = mf.potential_shaping(np.array([0.0, 1.0, 2.0]), 0.9)
    assert F(0, 1) == pytest.approx(0.9) and F(1, 0) == pytest.approx(-1.0)


def test_gradient_q_learning_and_replay_on_mountain_car():
    phi = fourier_basis(3, [mc.X_BOUNDS[0], mc.V_BOUNDS[0]], [mc.X_BOUNDS[1], mc.V_BOUNDS[1]])
    feats = lambda s: phi(np.asarray(s)[None])[0]   # noqa: E731

    def step(s, a):
        s2 = mc.step(s, mc.ACTIONS[a])
        return s2, -1.0, bool(mc.is_goal(s2))

    def late(**kw):
        rng = np.random.default_rng(0)
        L = mf.GradientQLearning(feats, 49, 3, 1.0, 0.3, **kw)
        lengths = [mf.run_episode(L, step, np.array([rng.uniform(-0.6, -0.4), 0.0]), rng) for _ in range(60)]
        return np.mean(lengths[-20:])
    plain, replay = late(), late(replay=5000, batch=32)
    assert plain < 400                                   # it learns (the cap is 1,000)
    assert replay < plain
