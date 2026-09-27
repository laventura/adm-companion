"""Also pins the numbers quoted in the Ch 2 page."""

from itertools import combinations

import numpy as np
import pytest

from adm.bayes_net import active_trail, d_separated, joint_table, num_parameters, probability
from adm.problems.commute import COMMUTE, PARENTS

ORDER = list(PARENTS)          # axes of joint_table: S, R, J, L, A


def test_chain_rule_and_counts():
    p = probability(COMMUTE, {"S": 1, "R": 0, "J": 1, "L": 1, "A": 1})
    assert np.isclose(p, 0.1 * 0.8 * 0.7 * 0.5 * 0.95)
    assert num_parameters(COMMUTE) == 10
    joint = joint_table(COMMUTE)
    assert joint.size - 1 == 31 and np.isclose(joint.sum(), 1)


def test_marginals_and_explaining_away():
    joint = joint_table(COMMUTE)
    p_jam = joint.sum(axis=(0, 1, 3, 4))[1]
    p_late = joint.sum(axis=(0, 1, 2, 4))[1]
    assert np.isclose(p_jam, 0.254) and np.isclose(p_late, 0.1643)
    p_storm_given_jam = joint[1, :, 1].sum() / joint[:, :, 1].sum()
    p_storm_given_jam_road = joint[1, 1, 1].sum() / joint[:, 1, 1].sum()
    assert np.isclose(p_storm_given_jam, 0.074 / 0.254)
    assert np.isclose(p_storm_given_jam_road, 0.09 / 0.63)


@pytest.mark.parametrize("x, y, given, separated", [
    ("S", "R", (), True),        # collider J unobserved: blocked
    ("S", "R", ("J",), False),   # observing the collider opens it
    ("S", "R", ("L",), False),   # so does observing a descendant
    ("A", "L", (), False),       # A <- S -> J -> L
    ("A", "L", ("S",), True),
    ("A", "L", ("J",), True),
    ("A", "R", ("J",), False),   # A <- S -> J <- R, collider observed
])
def test_d_separation_cases(x, y, given, separated):
    assert d_separated(PARENTS, x, y, given) is separated


def test_trail_is_reported():
    assert active_trail(PARENTS, "A", "L") == ["A", "S", "J", "L"]


def _independent(joint, x, y, given):
    """Numerically check P(x, y | z) = P(x | z) P(y | z) for all values."""
    keep = [ORDER.index(v) for v in (x, y, *given)]
    drop = tuple(i for i in range(len(ORDER)) if i not in keep)
    m = np.moveaxis(joint.sum(axis=drop), np.argsort(np.argsort(keep)), range(len(keep)))
    pz = m.sum(axis=(0, 1), keepdims=True)
    return np.allclose(m * pz, m.sum(axis=1, keepdims=True) * m.sum(axis=0, keepdims=True))


def test_d_separation_matches_the_numbers():
    joint = joint_table(COMMUTE)
    for x, y in combinations(ORDER, 2):
        rest = [v for v in ORDER if v not in (x, y)]
        for k in range(len(rest) + 1):
            for given in combinations(rest, k):
                assert d_separated(PARENTS, x, y, given) == _independent(joint, x, y, given), (x, y, given)
