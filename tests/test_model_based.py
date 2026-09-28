"""Tests for adm/model_based.py and the trail problem (Ch 16)."""

import numpy as np
import pytest

from adm import model_based as mb
from adm.mdp import policy_iteration
from adm.problems.hex_world import STANDARD_REWARDS, STANDARD_TILES, hex_world
from adm.problems.trail import trail

TRAIL = trail()
HEX = hex_world()
HEX_TERMINAL = len(HEX.R) - 1
HEX_DECISIONS = [i for i, t in enumerate(STANDARD_TILES) if t not in STANDARD_REWARDS]


def test_trail_is_a_valid_mdp_and_climbing_is_optimal():
    assert np.allclose(TRAIL.T.sum(axis=2), 1)
    policy, U, _ = policy_iteration(TRAIL.T, TRAIL.R, TRAIL.gamma, np.zeros(6, int))
    assert policy.tolist() == [1] * 6                  # climb everywhere
    assert U[0] > TRAIL.R[0, 0] / (1 - TRAIL.gamma)    # beats resting forever


def test_maximum_likelihood_by_hand():
    m = mb.MaximumLikelihoodModel(3, 2, 0.9)
    for s, a, r, s2 in [(0, 1, 0, 1), (0, 1, 0, 0), (0, 1, 0, 1), (1, 0, 2, 0), (0, 0, 1, 0)]:
        m.update(s, a, r, s2)
    T, R = m.model()
    assert np.allclose(T[0, 1], [1 / 3, 2 / 3, 0]) and R[1, 0] == 2 and R[0, 0] == 1
    assert np.all(T[2] == 0) and np.all(R[2] == 0)     # never tried: no reward, no future


def test_rmax_is_optimistic_about_unknown_pairs():
    m = mb.RmaxModel(3, 2, 0.9, m=2, r_max=1.0)
    m.update(0, 0, 0.0, 1)
    m.update(0, 0, 0.0, 1)                             # (0, 0) now known
    T, R = m.model()
    assert np.allclose(T[0, 0], [0, 1, 0]) and R[0, 0] == 0
    assert T[1, 1, 1] == 1 and R[1, 1] == 1            # unknown: self-loop paying r_max
    U = np.zeros(3)
    mb.solve(T, R, 0.9, U, tol=1e-10)
    assert U[1] == pytest.approx(1 / (1 - 0.9))


def test_bayesian_model_mean_and_samples():
    m = mb.BayesianModel(3, 1, 0.9)
    for s2 in (2, 2, 2, 2, 0):
        m.update(0, 0, 1.0, s2)
    T, R = m.model()
    assert np.allclose(T[0, 0], np.array([2, 1, 5]) / 8)   # (1 + counts) / (3 + 5)
    assert R[0, 0] == pytest.approx(6 / 7)
    Ts, Rs = m.sample(np.random.default_rng(0))
    assert np.allclose(Ts.sum(axis=2), 1) and np.all((Rs >= 0) & (Rs <= 1))
    draws = np.array([m.sample(np.random.default_rng(i))[0][0, 0] for i in range(4000)])
    assert np.allclose(draws.mean(axis=0), T[0, 0], atol=0.01)


def test_estimates_converge_with_data():
    rng = np.random.default_rng(1)
    m = mb.MaximumLikelihoodModel(len(HEX.R), 6, HEX.gamma)
    mb.learn(HEX, m, lambda *args: 0, 20_000, HEX_DECISIONS[0], rng,
             behavior=lambda s, r: r.integers(6),
             reset=lambda s, r: r.choice(HEX_DECISIONS) if s == HEX_TERMINAL else None)
    T, _ = m.model()
    err = np.abs(T - HEX.T).sum(axis=2)[HEX_DECISIONS]  # per-pair total variation x 2
    n = m.counts()[HEX_DECISIONS]
    # Random behavior visits some tiles rarely; well-visited pairs are accurate.
    assert err[n >= 200].max() < 0.25 and err.mean() < 0.15


@pytest.mark.parametrize("make", [lambda: mb.full_update, lambda: mb.randomized_update(5),
                                  lambda: mb.prioritized_update(5)],
                         ids=["full", "randomized", "prioritized"])
def test_planners_track_the_model_solution(make):
    rng, world = np.random.default_rng(2), np.random.default_rng(3)
    m = mb.MaximumLikelihoodModel(len(HEX.R), 6, HEX.gamma)
    _, backups, U = mb.learn(HEX, m, make(), 1500, HEX_DECISIONS[0], rng,
                             behavior=lambda s, r: r.integers(6), world_rng=world,
                             reset=lambda s, r: r.choice(HEX_DECISIONS) if s == HEX_TERMINAL else None)
    T, R = m.model()
    _, exact, _ = policy_iteration(T, R, HEX.gamma, np.zeros(len(HEX.R), int))
    assert np.abs(U - exact)[HEX_DECISIONS].mean() < 0.05


def test_prioritized_beats_random_early():
    def error(planner):
        rng, world = np.random.default_rng(4), np.random.default_rng(5)
        m = mb.MaximumLikelihoodModel(len(HEX.R), 6, HEX.gamma)
        _, _, U = mb.learn(HEX, m, planner, 100, HEX_DECISIONS[0], rng,
                           behavior=lambda s, r: r.integers(6), world_rng=world,
                           reset=lambda s, r: r.choice(HEX_DECISIONS) if s == HEX_TERMINAL else None)
        T, R = m.model()
        exact = U.copy()
        mb.solve(T, R, HEX.gamma, exact, tol=1e-10)
        return np.abs(U - exact).mean()
    assert error(mb.prioritized_update(5)) < error(mb.randomized_update(5))


def climbs(model_maker, planner_maker, epsilon, seeds=8, steps=1500):
    """Share of runs whose last 300 steps earn more than 0.5 per step."""
    wins = 0
    for seed in range(seeds):
        r, _, _ = mb.learn(TRAIL, model_maker(), planner_maker(), steps, 0,
                           np.random.default_rng(seed), epsilon=epsilon)
        wins += r[-300:].mean() > 0.5
    return wins / seeds


def test_directed_exploration_finds_the_summit():
    g = TRAIL.gamma
    assert climbs(lambda: mb.MaximumLikelihoodModel(6, 2, g), lambda: mb.full_update, 0.1) <= 0.25
    assert climbs(lambda: mb.RmaxModel(6, 2, g, 10, 1.0), lambda: mb.full_update, 0.0) >= 0.75
    assert climbs(lambda: mb.BayesianModel(6, 2, g), lambda: mb.posterior_sampling_update, 0.0) >= 0.75
