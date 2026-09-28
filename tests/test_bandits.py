"""Tests for adm/bandits.py (Ch 15)."""

import numpy as np
import pytest
from scipy import stats

from adm.bandits import (
    UCB1, EpsilonGreedy, ExploreThenCommit, Greedy, Quantile, Softmax, Thompson,
    argmax_split, evaluate, optimal_values, posterior_mean, q_values, simulate,
)

STRATEGIES = [Greedy(), EpsilonGreedy(0.2, 0.9), ExploreThenCommit(3), Softmax(4, 1.1),
              Quantile(0.9), UCB1(0.5), Thompson()]


def brute_force(strategy, w, l, t, h, optimal=False):
    """Expected remaining reward by plain recursion over every outcome:
    an independent check of the layered evaluation in `evaluate`."""
    if t == h:
        return 0.0
    rho = posterior_mean(w, l)
    q = []
    for a in range(len(w)):
        win_w, loss_l = w.copy(), l.copy()
        win_w[a] += 1
        loss_l[a] += 1
        q.append(rho[a] * (1 + brute_force(strategy, win_w, l, t + 1, h, optimal))
                 + (1 - rho[a]) * brute_force(strategy, w, loss_l, t + 1, h, optimal))
    q = np.array(q)
    return q.max() if optimal else float(strategy.probs(w, l, t) @ q)


def test_posterior_mean_and_ties():
    assert posterior_mean(np.array([5, 0]), np.array([3, 0])).tolist() == [0.6, 0.5]
    assert argmax_split(np.array([0.3, 0.7, 0.7])).tolist() == [0, 0.5, 0.5]


@pytest.mark.parametrize("s", STRATEGIES, ids=lambda s: type(s).__name__)
def test_probs_are_distributions(s):
    rng = np.random.default_rng(0)
    w, l = rng.integers(0, 6, (20, 3)), rng.integers(0, 6, (20, 3))
    p = s.probs(w, l, 4)
    assert p.shape == (20, 3) and np.all(p >= 0)
    assert np.allclose(p.sum(axis=1), 1)


def test_worked_example_probabilities():
    w, l = np.array([5, 1]), np.array([3, 1])
    assert np.allclose(EpsilonGreedy(0.2).probs(w, l, 0), [0.9, 0.1])
    assert Quantile(0.9).probs(w, l, 0).tolist() == [0, 1]   # B's wider posterior
    assert UCB1(0.5).probs(w, l, 0).tolist() == [0, 1]
    assert UCB1(0.0).probs(w, l, 0).tolist() == [1, 0]       # c = 0 is greedy


def test_thompson_probabilities_match_sampling():
    w, l = np.array([5, 1, 2]), np.array([3, 1, 4])
    exact = Thompson().probs(w, l, 0)
    draws = np.random.default_rng(1).beta(w + 1, l + 1, size=(400_000, 3)).argmax(axis=1)
    assert np.allclose(exact, np.bincount(draws, minlength=3) / 400_000, atol=4e-3)


@pytest.mark.parametrize("s", STRATEGIES, ids=lambda s: type(s).__name__)
def test_evaluate_matches_brute_force(s):
    zero = np.zeros(2, dtype=int)
    assert evaluate(s, 2, 5) == pytest.approx(brute_force(s, zero, zero, 0, 5), rel=1e-12)


def test_bayes_optimal_matches_brute_force_and_table():
    zero = np.zeros(2, dtype=int)
    assert evaluate(None, 2, 2, optimal=True) == pytest.approx(13 / 12)
    assert evaluate(None, 2, 6, optimal=True) == pytest.approx(
        brute_force(None, zero, zero, 0, 6, optimal=True), rel=1e-12)
    U = optimal_values(12, 14)
    assert U[12, 0, 0, 0, 0] == pytest.approx(evaluate(None, 2, 12, optimal=True), rel=1e-12)
    # With one pull left the optimal choice is greedy: Q_1 = rho.
    q1 = q_values(U, 1)[:, 5, 3, 1, 1]
    assert np.allclose(q1, [0.6, 0.5])


def test_optimal_beats_every_strategy():
    best = evaluate(None, 2, 12, optimal=True)
    for s in STRATEGIES:
        assert evaluate(s, 2, 12) <= best + 1e-12


@pytest.mark.parametrize("s", [UCB1(0.5), Thompson(), EpsilonGreedy(0.1)], ids=lambda s: type(s).__name__)
def test_simulation_matches_exact(s):
    # Average reward over bandits drawn from the uniform prior.
    rng = np.random.default_rng(2)
    theta = rng.random((100_000, 2))
    _, R, (w, l) = simulate(theta, s, 10, rng)
    assert np.all(w.sum(axis=1) + l.sum(axis=1) == 10)
    total = R.sum(axis=1)
    assert abs(total.mean() - evaluate(s, 2, 10)) < 4 * total.std() / np.sqrt(len(total))


def test_directed_exploration_beats_fixed_epsilon_long_run():
    rng = np.random.default_rng(3)
    theta = rng.random((400, 3))

    def regret(s):
        A, _, _ = simulate(theta, s, 1000, np.random.default_rng(4))
        return (theta.max(axis=1)[:, None] - np.take_along_axis(theta, A, axis=1)).sum(axis=1).mean()
    assert regret(Thompson()) < 0.5 * regret(EpsilonGreedy(0.1))
    assert regret(UCB1(0.5)) < 0.5 * regret(Greedy())


def test_quantile_uses_beta_quantile():
    w, l = np.array([[2, 9]]), np.array([[1, 3]])
    q = stats.beta.ppf(0.95, w + 1, l + 1)
    expected = np.eye(2)[q.argmax()]                  # one-hot on the higher quantile
    assert np.array_equal(Quantile(0.95).probs(w, l, 0)[0], expected)
