"""Policy gradient optimization (Ch 12) on the simple regulator."""

import numpy as np
import pytest

from adm.policy_gradient import baseline_gradient
from adm.policy_optimization import (
    advantage_estimates, clamped_gradient, clamped_surrogate_update,
    clip_gradient, discounted_rewards_to_go, fisher_information,
    natural_step, restricted_step, scale_gradient, surrogate_constraint,
    surrogate_objective, trust_region_update,
)
from adm.problems import simple_regulator as sr

G = sr.GAMMA
THETA = np.array([-0.5, 0.5])


def batch(theta=THETA, m=50, seed=0):
    return sr.sample_trajectories(theta, m, np.random.default_rng(seed))


def test_scale_and_clip():
    g = np.array([3.0, 0.5])
    assert np.linalg.norm(scale_gradient(g, 1.0)) == pytest.approx(1.0)
    assert scale_gradient(g, 10.0) == pytest.approx(g)     # short: unchanged
    s = scale_gradient(g, 1.0)
    assert s[0] / s[1] == pytest.approx(6.0)               # same direction
    assert clip_gradient(g, -1, 1) == pytest.approx([1.0, 0.5])


def test_restricted_and_natural_steps_meet_their_constraints():
    g, F, eps = np.array([2.0, 1.0]), np.diag([4.0, 0.5]), 0.1
    step = restricted_step(np.zeros(2), g, eps)
    assert 0.5 * step @ step == pytest.approx(eps)
    assert step == pytest.approx([0.4, 0.2])
    nat = natural_step(np.zeros(2), g, F, eps)
    assert 0.5 * nat @ F @ nat == pytest.approx(eps)
    assert nat == pytest.approx(np.array([0.5, 2.0]) * np.sqrt(0.2 / 3))


def test_natural_step_is_invariant_to_rescaling_a_parameter():
    # Parameterize with phi = D theta. Gradients map by D^-1 and the Fisher
    # matrix by D^-1 F D^-1, so the natural step in theta is unchanged.
    g, F, D = np.array([2.0, 1.0]), np.array([[4.0, 1.0], [1.0, 0.5]]), \
        np.diag([1.0, 10.0])
    Dinv = np.linalg.inv(D)
    step_theta = natural_step(np.zeros(2), g, F, 0.1)
    step_phi = natural_step(np.zeros(2), Dinv @ g, Dinv @ F @ Dinv, 0.1)
    # (Here F^-1 g = (0, 2), so compare absolutely: the tiny ridge in
    # natural_step makes the zero component about 1e-6 instead of 0.)
    assert Dinv @ step_phi == pytest.approx(step_theta, abs=1e-5)
    # The plain restricted step is not invariant.
    plain_phi = restricted_step(np.zeros(2), Dinv @ g, 0.1)
    assert not np.allclose(Dinv @ plain_phi, restricted_step(np.zeros(2), g,
                                                             0.1))


def test_policy_functions():
    s, a = np.array([0.4, -0.2]), np.array([0.1, 0.3])
    numeric = np.stack([(sr.log_policy(THETA + h, s, a)
                         - sr.log_policy(THETA - h, s, a)) / 2e-6
                        for h in 1e-6 * np.eye(2)], axis=-1)
    assert sr.grad_log_policy(THETA, s, a) == pytest.approx(numeric, rel=1e-5)
    assert sr.policy_kl(THETA, THETA, s) == pytest.approx([0, 0])
    # KL(N(-0.2, 0.25) || N(-0.24, 0.16)) at s = 0.4
    kl = sr.policy_kl(THETA, np.array([-0.6, 0.4]), np.array([0.4]))[0]
    # (The policy adds 1e-5 to each standard deviation, hence rel=1e-3.)
    assert kl == pytest.approx(np.log(0.8) + (0.25 + 0.04**2) / 0.32 - 0.5,
                               rel=1e-3)


def test_fisher_is_symmetric_positive_semidefinite():
    S, A, _ = batch()
    F = fisher_information(THETA, S, A, sr.grad_log_policy)
    assert F == pytest.approx(F.T)
    assert np.all(np.linalg.eigvalsh(F) >= -1e-12)


def test_surrogates_at_the_current_policy():
    S, A, R = batch()
    q = discounted_rewards_to_go(R, G)
    assert surrogate_objective(THETA, THETA, S, A, q, sr.log_policy) == \
        pytest.approx(q.mean())
    assert surrogate_constraint(THETA, THETA, S, G, sr.policy_kl) == \
        pytest.approx(0)
    adv = advantage_estimates(R, G)
    assert adv.mean(axis=0) == pytest.approx(np.zeros(R.shape[1]), abs=1e-12)


def test_trust_region_step_respects_the_constraint():
    S, A, R = batch()
    grad = baseline_gradient(sr.grad_log_policy(THETA, S, A), R, G)
    adv = advantage_estimates(R, G)
    new = trust_region_update(THETA, grad, S, A, adv, G, sr.log_policy,
                              sr.grad_log_policy, sr.policy_kl, 0.2)
    assert surrogate_constraint(THETA, new, S, G, sr.policy_kl) <= 0.2
    assert (surrogate_objective(THETA, new, S, A, adv, sr.log_policy)
            > surrogate_objective(THETA, THETA, S, A, adv, sr.log_policy))


def test_clamped_gradient():
    S, A, R = batch()
    adv = advantage_estimates(R, G)
    # At theta2 = theta nothing is clamped: it is the plain gradient.
    unclamped = (sr.grad_log_policy(THETA, S, A) * adv[..., None]).mean(
        axis=(0, 1))
    assert clamped_gradient(THETA, THETA, S, A, adv, sr.log_policy,
                            sr.grad_log_policy, 0.2) == pytest.approx(unclamped)
    # Far from theta, with eps = 0, most steps are clamped.
    far = clamped_gradient(THETA, THETA + [0.3, -0.2], S, A, adv,
                           sr.log_policy, sr.grad_log_policy, 0.0)
    assert np.linalg.norm(far) < np.linalg.norm(unclamped)


@pytest.mark.parametrize("rule", ["natural", "trpo", "ppo"])
def test_update_rules_reach_the_optimum(rule):
    rng = np.random.default_rng(0)
    theta = np.array([0.0, 1.0])
    for _ in range(30):
        S, A, R = sr.sample_trajectories(theta, 10, rng)
        grad = baseline_gradient(sr.grad_log_policy(theta, S, A), R, G)
        adv = advantage_estimates(R, G)
        if rule == "natural":
            F = fisher_information(theta, S, A, sr.grad_log_policy)
            theta = natural_step(theta, grad, F, 0.2)
        elif rule == "trpo":
            theta = trust_region_update(theta, grad, S, A, adv, G,
                                        sr.log_policy, sr.grad_log_policy,
                                        sr.policy_kl, 0.2)
        else:
            theta = clamped_surrogate_update(theta, S, A, adv, sr.log_policy,
                                             sr.grad_log_policy, 0.2, 0.05,
                                             10)
    assert sr.exact_value(theta) > sr.exact_value([-1.0, 0.0]) - 0.05
