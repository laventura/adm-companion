"""Tests for adm/imitation.py and the park problem (Ch 18)."""

import numpy as np
import pytest

from adm import imitation as im
from adm.mdp import policy_iteration
from adm.problems.park import PHI_TRUE, left_edge_starts, park

MDP, BETA = park("A")
T, R, G = MDP.T, MDP.R, MDP.gamma
N = len(R)
END = N - 1
B = left_edge_starts("A")
PI_E, U_E, _ = policy_iteration(T, R, G, np.zeros(N, int))
P_E = im.greedy_probs(PI_E, 4)
ALL = np.zeros(N)
ALL[[s for s in range(END) if MDP.states[s] != (0, 7)]] = 1
ALL /= ALL.sum()


def test_park_is_valid_and_reward_is_linear():
    assert np.allclose(T.sum(axis=2), 1)
    assert np.allclose(R, BETA @ PHI_TRUE)


def test_values_and_feature_expectations_agree():
    v = im.policy_value(T, R, G, P_E, B)
    assert v == pytest.approx(B @ U_E)
    mu = im.feature_expectations(T, BETA, G, P_E, B)
    assert mu @ PHI_TRUE == pytest.approx(v)            # linear reward: value = phi . mu


def test_behavioral_cloning_counts_and_uniform_elsewhere():
    P = im.behavioral_cloning([(0, 1), (0, 1), (0, 2), (5, 3)], N, 4)
    assert np.allclose(P[0], [0, 2 / 3, 1 / 3, 0]) and np.allclose(P[5], [0, 0, 0, 1])
    assert np.allclose(P[7], 0.25)


def test_dagger_improves_on_cloning():
    gains = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        demos = im.rollouts(T, P_E, B, 3, 60, rng, end=END)
        pols = im.dagger(T, PI_E, B, demos, 6, 3, 60, rng, end=END)
        gains.append(im.policy_value(T, R, G, pols[-1], B) - im.policy_value(T, R, G, pols[0], B))
    assert np.mean(gains) > 3


def test_smile_with_enough_data_is_near_the_expert():
    rng = np.random.default_rng(0)
    _, final = im.smile(T, PI_E, B, 10, 30, 60, 0.2, rng, end=END)
    assert im.policy_value(T, R, G, final, B) > im.policy_value(T, R, G, P_E, B) - 1.5


def test_max_margin_weights_and_mixture_on_a_toy():
    t, phi = im.max_margin_weights(np.array([1.0, 0.0]), [np.array([0.0, 1.0])])
    assert np.allclose(phi, np.array([1, -1]) / np.sqrt(2), atol=1e-4) and t == pytest.approx(np.sqrt(2), abs=1e-4)
    lam = im.mixture_weights(np.array([0.25, 0.75]), [np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert np.allclose(lam, [0.25, 0.75], atol=1e-5)


def test_soft_policy_is_a_distribution_and_sharpens_with_scale():
    P1 = im.soft_policy(T, R / 10, G)
    P2 = im.soft_policy(T, R * 10, G)
    assert np.allclose(P1.sum(axis=1), 1)
    assert np.mean(P2.argmax(axis=1)[:END] == PI_E[:END]) > 0.9   # nearly greedy


def test_max_entropy_irl_recovers_a_transferable_reward():
    demos = im.rollouts(T, P_E, ALL, 30, 80, np.random.default_rng(0), end=END)
    phi, _, norms = im.max_entropy_irl(T, BETA, G, demos, ALL, 80, 0.05, 600)
    assert phi[0] > phi[1] > phi[2]                     # path better than grass better than mud
    assert norms[-1] < norms[0] / 20
    mdp_b, beta_b = park("B")
    pi_b, _, _ = policy_iteration(mdp_b.T, beta_b @ phi, G, np.zeros(N, int))
    pi_star, _, _ = policy_iteration(mdp_b.T, mdp_b.R, G, np.zeros(N, int))
    b_b = left_edge_starts("B")
    learned = im.policy_value(mdp_b.T, mdp_b.R, G, im.greedy_probs(pi_b, 4), b_b)
    best = im.policy_value(mdp_b.T, mdp_b.R, G, im.greedy_probs(pi_star, 4), b_b)
    assert learned > best - 0.5


def test_max_margin_mixture_is_close_to_the_expert():
    demos = im.rollouts(T, P_E, B, 30, 80, np.random.default_rng(0), end=END)
    mu_E = im.demo_feature_expectations(demos, BETA, G)
    lam, pols, _, _, _ = im.max_margin_irl(T, BETA, G, mu_E, B, k_max=15)
    value = sum(l * im.policy_value(T, R, G, P, B) for l, P in zip(lam, pols))
    assert value > im.policy_value(T, R, G, P_E, B) - 2
