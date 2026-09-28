"""Actor-critic methods (Ch 13)."""

import numpy as np
import pytest

from adm.actor_critic import (
    GuidedTreeSearch, actor_critic_step, actor_gradient, critic_gradient,
    deterministic_policy_gradients, generalized_advantages, root_targets,
    softmax, td_residuals, tree_search_actor_critic_step,
)
from adm.mdp import policy_iteration
from adm.policy_gradient import rewards_to_go
from adm.problems import simple_regulator as sr
from adm.problems.hex_world import STANDARD_REWARDS, STANDARD_TILES, hex_world

G = sr.GAMMA
feat = lambda S: np.stack([np.ones_like(S), S**2], axis=-1)


def test_td_residuals_and_gae_by_hand():
    values = np.array([[1.0, 2.0, 4.0]])
    R = np.array([[0.5, -1.0, 7.0]])
    delta = td_residuals(values, R, 0.5)
    assert delta[0] == pytest.approx([0.5 + 1 - 1, -1 + 2 - 2])
    # A_1 = delta_1 + (gamma lam) delta_2 with gamma lam = 0.25
    adv = generalized_advantages(delta, 0.5, 0.5)
    assert adv[0] == pytest.approx([0.5 + 0.25 * -1.0, -1.0])
    assert generalized_advantages(delta, 0.5, 0.0) == pytest.approx(delta)


def test_gae_with_lambda_one_telescopes_to_reward_to_go():
    rng = np.random.default_rng(0)
    values, R = rng.normal(size=(4, 6)), rng.normal(size=(4, 6))
    adv = generalized_advantages(td_residuals(values, R, G), G, 1.0)
    # sum_l gamma^l delta_(k+l) = rtg over steps k..d-1
    #   + gamma^(d-k) U(s_d) - U(s_k)
    rtg_short = rewards_to_go(R[:, :-1], G)
    d = R.shape[1]
    tail = G ** (d - 1 - np.arange(d - 1)) * values[:, -1:]
    assert adv == pytest.approx(rtg_short + tail - values[:, :-1])


def test_critic_gradient_vanishes_at_the_least_squares_fit():
    S, A, R = sr.sample_trajectories(np.array([-0.5, 0.5]), 200,
                                     np.random.default_rng(1))
    X, y = feat(S).reshape(-1, 2), rewards_to_go(R, G).reshape(-1)
    phi = np.linalg.lstsq(X, y, rcond=None)[0]
    grad = critic_gradient(feat(S) @ phi, feat(S), R, G)
    assert grad == pytest.approx(np.zeros(2), abs=1e-9)


def test_a_good_critic_with_td_residuals_beats_monte_carlo_on_variance():
    theta = np.array([-0.5, 0.5])
    S, A, R = sr.sample_trajectories(theta, 20_000, np.random.default_rng(2))
    phi = np.linalg.lstsq(feat(S).reshape(-1, 2),
                          rewards_to_go(R, G).reshape(-1), rcond=None)[0]
    spread = {}
    for lam in (0.0, 1.0):
        E = []
        for i in range(200):
            S, A, R = sr.sample_trajectories(theta, 10,
                                             np.random.default_rng(10 + i))
            adv = generalized_advantages(
                td_residuals(feat(S) @ phi, R, G), G, lam)
            E.append(actor_gradient(sr.grad_log_policy(theta, S[:, :-1],
                                                       A[:, :-1]), adv, G))
        spread[lam] = np.array(E).std(axis=0).sum()
    assert spread[0.0] < spread[1.0]


def test_actor_critic_learns_the_regulator():
    rng = np.random.default_rng(3)
    theta, phi = np.array([0.0, 1.0]), np.zeros(2)
    for _ in range(60):
        S, A, R = sr.sample_trajectories(theta, 10, rng)
        theta, phi = actor_critic_step(theta, phi, S, A, R, G,
                                       sr.grad_log_policy, feat, 0.5, 0.1,
                                       0.002)
    assert sr.exact_value(theta) > sr.exact_value([-1.0, 0.0]) - 0.1


def test_deterministic_policy_gradient_on_the_regulator():
    policy = lambda th, S: th[0] * S
    dpolicy = lambda th, S: S[..., None]
    qf = lambda S, A: np.stack([np.ones_like(S), S**2, (S + A)**2], axis=-1)
    dqda = lambda ph, S, A: 2 * ph[2] * (S + A)
    # The actor gradient is s * 2 phi_3 (s + theta s): with phi_3 < 0 it
    # pushes the gain toward -1 from either side.
    S = np.array([[0.5, -0.2]])
    ga, _ = deterministic_policy_gradients(
        np.array([0.0]), np.array([0, -1.0, -1.0]), S, np.zeros_like(S),
        np.zeros_like(S), 1.0, policy, dpolicy, qf, dqda)
    assert ga[0] == pytest.approx(2 * -1.0 * (0.25 + 0.04) / 1)
    rng, theta, phi = np.random.default_rng(4), np.array([0.0]), \
        np.array([0.0, -1.0, -1.0])
    for _ in range(50):
        S, A, R = sr.sample_trajectories(np.array([theta[0], 0.5]), 5, rng)
        ga, gc = deterministic_policy_gradients(theta, phi, S, A, R, G,
                                                policy, dpolicy, qf, dqda)
        theta = theta + 0.05 * ga / max(1.0, np.linalg.norm(ga))
        phi = phi - 0.05 * gc
    assert theta[0] == pytest.approx(-1.0, abs=0.1)


def test_tree_search_actor_critic_on_hex_world():
    mdp = hex_world()
    pi_star, U_star, _ = policy_iteration(mdp.T, mdp.R, mdp.gamma,
                                          np.zeros(len(mdp.T), int))
    starts = [i for i, t in enumerate(STANDARD_TILES)
              if t not in STANDARD_REWARDS]
    rng = np.random.default_rng(0)
    logits, values = np.zeros((len(mdp.T), 6)), np.zeros(len(mdp.T))
    tree = GuidedTreeSearch(mdp, softmax(logits), values, 10, 5.0, rng)
    for _ in range(30):
        tree.simulate(starts[0], 10)
    pi, u = root_targets(tree, starts[0])
    assert pi.sum() == pytest.approx(1) and np.isfinite(u)
    for _ in range(30):
        logits, values = tree_search_actor_critic_step(
            mdp, logits, values, starts, 100, 10, 5.0, 1.0, 0.5, rng)
    agree = np.mean(logits.argmax(axis=1)[starts] == pi_star[starts])
    assert agree >= 0.8
    assert np.abs(values[starts] - U_star[starts]).mean() < 0.5
