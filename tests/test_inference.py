"""Also pins the numbers quoted in the Ch 3 page."""

from itertools import combinations, product

import numpy as np
import pytest

from adm.bayes_net import joint_table
from adm.inference import (
    Factor, condition, exact_inference, factor_product, gaussian_condition, gibbs_sampling,
    likelihood_weighted_sampling, naive_bayes_posterior, rejection_sampling, sum_out,
    topological_order, variable_elimination,
)
from adm.problems.coffee import COFFEE, LIKELIHOODS, PRIOR
from adm.problems.commute import COMMUTE, COMMUTE_TIMES_COV, COMMUTE_TIMES_MEAN


def brute_force(bn, x, evidence):
    """P(x | evidence) by slicing and summing the full joint table (Ch 2)."""
    names = list(bn.parents)
    joint = joint_table(bn)
    index = tuple(evidence.get(v, slice(None)) for v in names)
    kept = [v for v in names if v not in evidence]
    sliced = joint[index]
    p = sliced.sum(axis=tuple(i for i, v in enumerate(kept) if v != x))
    return p / p.sum()


@pytest.mark.parametrize("bn", [COMMUTE, COFFEE], ids=["commute", "coffee"])
def test_exact_methods_match_brute_force_everywhere(bn):
    names = list(bn.parents)
    for x in names:
        rest = [v for v in names if v != x]
        for k in range(len(rest) + 1):
            for observed in combinations(rest, k):
                for values in product(*(range(bn.card[v]) for v in observed)):
                    ev = dict(zip(observed, values))
                    expected = brute_force(bn, x, ev)
                    hidden = [v for v in names if v != x and v not in ev]
                    assert np.allclose(exact_inference(bn, (x,), ev).table, expected)
                    assert np.allclose(variable_elimination(bn, (x,), ev, hidden).table, expected)


def test_factor_operations():
    f = Factor(("A", "B"), np.array([[1.0, 2.0], [3.0, 4.0]]))
    g = Factor(("B", "C"), np.array([[10.0, 20.0], [30.0, 40.0]]))
    h = factor_product(f, g)
    assert h.variables == ("A", "B", "C") and h.table[1, 0, 1] == 3.0 * 20.0
    assert sum_out(h, "B").table[0, 0] == 1 * 10 + 2 * 30
    assert condition(h, {"B": 1}).variables == ("A", "C")
    assert topological_order(COMMUTE.parents).index("J") > topological_order(COMMUTE.parents).index("S")


EVIDENCE = {"L": 1, "A": 1}   # late, and the radio warned of a storm


def test_samplers_approach_the_exact_answer():
    exact = exact_inference(COMMUTE, ("S",), EVIDENCE).table[1]
    rng = np.random.default_rng(3)
    assert abs(rejection_sampling(COMMUTE, ("S",), EVIDENCE, 20_000, rng).table[1] - exact) < 0.03
    assert abs(likelihood_weighted_sampling(COMMUTE, ("S",), EVIDENCE, 5_000, rng).table[1] - exact) < 0.03
    assert abs(gibbs_sampling(COMMUTE, ("S",), EVIDENCE, 5_000, rng).table[1] - exact) < 0.04


def test_naive_bayes_worked_example():
    post = naive_bayes_posterior(PRIOR, LIKELIHOODS, [1, 1, 0])   # slow, noisy, taste fine
    assert np.allclose(post, np.array([0.0036, 0.0084, 0.0168]) / 0.0288)
    assert np.allclose(post, exact_inference(COFFEE, ("C",), {"F": 1, "N": 1, "W": 0}).table)


def test_gaussian_conditioning():
    mu, S = gaussian_condition(COMMUTE_TIMES_MEAN, COMMUTE_TIMES_COV, [1], [0], [50.0])
    assert np.allclose(mu, [51.25]) and np.allclose(S, [[64.0]])
    # Independent check: the conditional covariance is the inverse of the
    # matching block of the precision matrix (a Schur-complement identity).
    precision = np.linalg.inv(COMMUTE_TIMES_COV)
    assert np.allclose(S, np.linalg.inv(precision[np.ix_([1], [1])]))
