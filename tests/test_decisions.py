"""Also pins the numbers quoted in the Ch 6 page."""

from itertools import product

import numpy as np
import pytest

from adm.decisions import (
    action_utilities, best_action, certainty_equivalent, expected_utility,
    exponential_utility, value_of_information,
)
from adm.problems.commute import ACTIONS, COMMUTE_DECISION as BN, COMMUTE_UTILITY as U

EU = lambda ev: action_utilities(BN, "D", ("L",), U, ev)
VOI = lambda ev, o: value_of_information(BN, "D", ("L",), U, ev, o)


def test_lotteries_and_risk():
    assert expected_utility([0.5, 0.5], [0.0, 1.0]) == 0.5
    assert np.isclose(certainty_equivalent([0, 100], [0.5, 0.5], 0.01), 37.99, atol=0.01)
    assert certainty_equivalent([0, 100], [0.5, 0.5], 0.02) < certainty_equivalent([0, 100], [0.5, 0.5], 0.01) < 50
    assert np.isclose(exponential_utility(0, 0.3), 0.0)


def test_commute_decisions():
    assert np.allclose(EU({}), [-1.643, -2.254, -6.0])
    assert ACTIONS[best_action(BN, "D", ("L",), U, {})[0]] == "leave on time"
    assert ACTIONS[best_action(BN, "D", ("L",), U, {"A": 1})[0]] == "leave early"
    assert np.allclose(EU({"J": 1}), [-5.0, -3.0, -6.0]) and np.allclose(EU({"J": 0}), [-0.5, -2.0, -6.0])


def test_value_of_information():
    assert np.isclose(VOI({}, "J"), 0.508)
    assert np.isclose(VOI({}, "A"), 0.0892, atol=1e-4)
    assert VOI({}, "J") > VOI({}, "R") > VOI({}, "S") > VOI({}, "A") > 0
    assert np.isclose(VOI({"J": 1}, "A"), 0.0)          # the jam already settles the choice
    # VOI is never negative, whatever you already know.
    for a, j in product([None, 0, 1], repeat=2):
        ev = {k: v for k, v in (("A", a), ("J", j)) if v is not None}
        for o in ("S", "R", "A", "J"):
            if o not in ev:
                assert VOI(ev, o) >= -1e-12


def test_allais_choices_are_inconsistent_with_any_utility():
    """1A over 1B and 2B over 2A can't both hold for any utilities u0 < u1 < u5
    of $0, $1M and $5M (Allais, 1953)."""
    rng = np.random.default_rng(0)
    for u0, u1, u5 in np.sort(rng.random((2000, 3)), axis=1):
        prefers_1a = u1 > 0.89 * u1 + 0.10 * u5 + 0.01 * u0
        prefers_2b = 0.10 * u5 + 0.90 * u0 > 0.11 * u1 + 0.89 * u0
        assert not (prefers_1a and prefers_2b)
