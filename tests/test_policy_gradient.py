"""Policy gradient estimation (Ch 11) on the simple regulator."""

import numpy as np
import pytest

from adm.policy_gradient import (
    baseline_gradient, discounted_returns, finite_difference_gradient,
    likelihood_ratio_gradient, regression_gradient, reward_to_go_gradient,
    rewards_to_go,
)
from adm.problems import simple_regulator as sr

THETA = np.array([-0.5, 0.5])
TRUE = sr.exact_gradient(THETA)


def estimates(fn, m, reps, seed=0):
    out = []
    for i in range(reps):
        rng = np.random.default_rng(seed + i)
        S, A, R = sr.sample_trajectories(THETA, m, rng)
        out.append(fn(sr.grad_log_policy(THETA, S, A), R, sr.GAMMA))
    return np.array(out)


def test_exact_gradient_matches_differences_of_exact_value():
    for theta in ([-0.5, 0.5], [0.0, 1.0], [-1.3, 0.2]):
        theta = np.array(theta)
        numeric = [(sr.exact_value(theta + h) - sr.exact_value(theta - h))
                   / 2e-6 for h in 1e-6 * np.eye(2)]
        assert sr.exact_gradient(theta) == pytest.approx(numeric, rel=1e-6)


def test_grad_log_policy_matches_numeric_derivative():
    s, a = np.array([0.4]), np.array([-0.1])

    def log_pi(theta):
        sd = abs(theta[1]) + 1e-5
        return (-0.5 * ((a - theta[0] * s) / sd) ** 2 - np.log(sd))[0]
    numeric = [(log_pi(THETA + h) - log_pi(THETA - h)) / 2e-6
               for h in 1e-6 * np.eye(2)]
    assert sr.grad_log_policy(THETA, s, a)[0] == pytest.approx(numeric,
                                                               rel=1e-5)


def test_rewards_to_go_and_returns():
    r = np.array([[1.0, 2.0, 3.0]])
    assert rewards_to_go(r, 0.5)[0] == pytest.approx([1 + 1 + 0.75,
                                                      2 + 1.5, 3])
    assert discounted_returns(r, 0.5)[0] == pytest.approx(1 + 1 + 0.75)


@pytest.mark.parametrize("fn", [likelihood_ratio_gradient,
                                reward_to_go_gradient, baseline_gradient])
def test_trajectory_estimators_are_close_to_the_true_gradient(fn):
    # Unbiased: with 100,000 rollouts the likelihood ratio's standard error
    # is about 0.09 per component (sd about 9 per rollout batch of 10).
    est = estimates(fn, 100_000, 1)[0]
    assert est == pytest.approx(TRUE, abs=0.3)


def test_variance_falls_with_reward_to_go_and_baseline():
    sd = {fn.__name__: estimates(fn, 10, 300).std(axis=0).sum()
          for fn in (likelihood_ratio_gradient, reward_to_go_gradient,
                     baseline_gradient)}
    assert (sd["baseline_gradient"] < sd["reward_to_go_gradient"]
            < sd["likelihood_ratio_gradient"])


def test_common_random_numbers_tame_finite_differences():
    def fd(seed, shared):
        rng = np.random.default_rng(seed)
        if shared:                       # same random numbers for every U
            U = lambda t: sr.rollout_returns(t, 10,
                                             np.random.default_rng(seed)).mean()
        else:
            U = lambda t: sr.rollout_returns(t, 10, rng).mean()
        return finite_difference_gradient(U, THETA, 0.01)
    fresh = np.array([fd(s, False) for s in range(200)])
    shared = np.array([fd(s, True) for s in range(200)])
    assert shared.std(axis=0).sum() < fresh.std(axis=0).sum() / 10
    assert shared.mean(axis=0) == pytest.approx(TRUE, abs=0.3)


def test_regression_gradient_recovers_a_linear_function():
    g = np.array([2.0, -3.0, 0.5])
    U = lambda t: 7.0 + g @ t
    est = regression_gradient(U, np.zeros(3), 0.1, 6,
                              np.random.default_rng(0))
    assert est == pytest.approx(g)
    assert finite_difference_gradient(U, np.zeros(3), 0.1) == pytest.approx(g)
