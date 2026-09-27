"""Also pins the numbers quoted in the Ch 4 page."""

import numpy as np
import pytest

from adm.learning import (
    bayes_estimate, beta_update, count_table, expectation_maximization, gaussian_ml,
    impute_mode, kernel_density, log_likelihood, ml_estimate, sample_dataset,
)
from adm.problems.commute import COMMUTE, PARENTS

CARD = COMMUTE.card


@pytest.fixture(scope="module")
def mornings():
    return sample_dataset(COMMUTE, 3000, np.random.default_rng(4))


def test_counts_and_ml_on_a_tiny_dataset():
    data = [{"S": 0, "R": 0, "J": 0, "L": 0, "A": 0}, {"S": 0, "R": 1, "J": 1, "L": 1, "A": 0},
            {"S": 0, "R": 1, "J": 0, "L": 0, "A": 0}]
    counts = count_table(CARD, PARENTS, data)
    assert counts["J"][0, 1].tolist() == [1, 1] and counts["R"].tolist() == [1, 2]
    ml = ml_estimate(CARD, PARENTS, data)
    assert np.allclose(ml.cpt["J"][0, 1], [0.5, 0.5])
    assert np.allclose(ml.cpt["J"][1, 1], [0.5, 0.5])       # never seen: uniform fallback


def test_ml_approaches_truth(mornings):
    ml = ml_estimate(CARD, PARENTS, mornings)
    assert abs(ml.cpt["S"][1] - 0.1) < 0.02 and abs(ml.cpt["L"][1, 1] - 0.5) < 0.05


def test_bayes_estimate_is_smoothed_counts():
    assert np.allclose(bayes_estimate(CARD, PARENTS, []).cpt["J"], 0.5)   # no data: uniform
    data = [{"S": 1, "R": 0, "J": 1, "L": 1, "A": 1}] * 3
    est = bayes_estimate(CARD, PARENTS, data, alpha=1.0)
    assert np.isclose(est.cpt["S"][1], (3 + 1) / (3 + 2))


def test_beta_numbers():
    assert beta_update(1, 1, 3, 7) == (4, 8)       # posterior mean 4 / 12 = 1/3
    assert beta_update(5, 5, 3, 7) == (8, 12)      # posterior mean 8 / 20 = 0.4


def test_gaussian_ml_and_kde():
    x = np.array([30.0, 34.0, 29.0, 52.0, 35.0])
    mu, var = gaussian_ml(x)
    assert np.isclose(mu, x.mean()) and np.isclose(var, x.var())   # np.var divides by m too
    grid = np.linspace(0, 90, 4001)
    density = kernel_density(x, 3.0)(grid)
    assert abs(density.sum() * (grid[1] - grid[0]) - 1) < 1e-3


def hide(data, variable, fraction, rng):
    return [{**r, variable: None} if rng.random() < fraction else dict(r) for r in data]


def test_em(mornings):
    rng = np.random.default_rng(5)
    partial = hide(mornings, "S", 0.7, rng)
    start = bayes_estimate(CARD, PARENTS, [])          # uniform CPTs
    em, history = expectation_maximization(start, partial, iterations=60)
    assert np.all(np.diff(history) >= -1e-9)           # EM never lowers the likelihood
    assert abs(history[-1] - history[-2]) < 1e-3       # converged
    # EM should land near what full data would give (sampling noise aside).
    full = ml_estimate(CARD, PARENTS, mornings)
    assert abs(em.cpt["S"][1] - full.cpt["S"][1]) < 0.02
    assert abs(em.cpt["A"][1, 1] - full.cpt["A"][1, 1]) < 0.04
    imputed = ml_estimate(CARD, PARENTS, impute_mode(partial))
    assert abs(imputed.cpt["S"][1] - 0.1) > 0.05       # mode imputation: badly biased
    # With nothing missing, one EM step is exactly maximum likelihood.
    one_step, _ = expectation_maximization(start, mornings, iterations=1)
    assert np.allclose(one_step.cpt["J"], full.cpt["J"])
    assert log_likelihood(full, mornings) >= log_likelihood(start, mornings)
