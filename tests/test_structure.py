"""Also pins the numbers quoted in the Ch 5 page."""

from itertools import product
from math import log

import numpy as np
import pytest

from adm.learning import sample_dataset
from adm.problems.commute import COMMUTE, PARENTS
from adm.structure import (
    all_dags, bayesian_score, family_scores, graph_score, is_acyclic, k2_search,
    local_search, markov_equivalent, neighbors, score_component,
)

CARD, NODES = COMMUTE.card, list(PARENTS)
TWIN = {**PARENTS, "A": (), "S": ("A",)}        # alert -> storm instead: Markov equivalent


@pytest.fixture(scope="module")
def mornings():
    return sample_dataset(COMMUTE, 3000, np.random.default_rng(1))


def test_score_component_hand_numbers():
    assert np.isclose(score_component(np.array([[19.0, 1.0], [5.0, 5.0]]), 1.0), -13.9676, atol=1e-4)
    assert np.isclose(score_component(np.array([24.0, 6.0]), 1.0), -16.7282, atol=1e-4)
    assert np.isclose(score_component(np.array([2.0, 1.0]), 1.0), -log(12))


def test_score_equals_sequential_prediction():
    """With one pseudocount per cell, exp(score) is the probability of the data
    when each record is predicted from the (count + 1) / (total + r) of the
    records before it."""
    records = [1, 0, 0, 1, 1, 1, 0, 1]
    counts, p = np.zeros(2), 1.0
    for x in records:
        p *= (counts[x] + 1) / (counts.sum() + 2)
        counts[x] += 1
    assert np.isclose(score_component(counts, 1.0), log(p))


def test_family_table_matches_direct_score(mornings):
    for prior in ("uniform", "bdeu"):
        table = family_scores(CARD, mornings[:500], prior)
        assert np.isclose(graph_score(table, PARENTS), bayesian_score(CARD, PARENTS, mornings[:500], prior))


def test_bdeu_is_score_equivalent(mornings):
    assert markov_equivalent(PARENTS, TWIN)
    assert np.isclose(bayesian_score(CARD, PARENTS, mornings, "bdeu"), bayesian_score(CARD, TWIN, mornings, "bdeu"))
    assert not np.isclose(bayesian_score(CARD, PARENTS, mornings), bayesian_score(CARD, TWIN, mornings))


def test_graph_utilities():
    assert is_acyclic(PARENTS) and not is_acyclic({"A": ("B",), "B": ("A",)})
    assert [sum(1 for _ in all_dags(NODES[:n])) for n in range(1, 6)] == [1, 3, 25, 543, 29281]
    assert sum(markov_equivalent(g, PARENTS) for g in all_dags(NODES)) == 2
    assert not markov_equivalent(PARENTS, {**PARENTS, "J": ("S",), "R": ("J",)})   # breaks the v-structure
    # Three nodes: 25 DAGs fall into 11 Markov equivalence classes.
    dags = list(all_dags(["X", "Y", "Z"]))
    classes = []
    for g in dags:
        if not any(markov_equivalent(g, c) for c in classes):
            classes.append(g)
    assert len(classes) == 11
    # 10 pairs: an absent pair can be added one way or the other.
    assert len(list(neighbors({x: () for x in NODES}))) == 20


def test_searches_find_the_true_class(mornings):
    table = family_scores(CARD, mornings, "bdeu")
    score = lambda g: graph_score(table, g)
    assert markov_equivalent(k2_search(score, ["S", "R", "J", "L", "A"]), PARENTS)
    assert not markov_equivalent(k2_search(score, ["L", "J", "A", "S", "R"]), PARENTS)
    found, steps = local_search(score, {x: () for x in NODES})
    assert markov_equivalent(found, PARENTS) and np.all(np.diff(steps) > 0)
