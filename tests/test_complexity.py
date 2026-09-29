"""Tests for adm.complexity (Appendix C)."""

from itertools import product

from adm import complexity as C

FORMULA = [(1, 2, -3), (-1, 3, 4), (-2, -4, 3), (1, -2, -4), (2, 3, 4), (-1, -3, -4)]
# All 8 sign patterns over x1..x3: each assignment breaks exactly one clause.
UNSAT = [tuple(s * i for s, i in zip(signs, (1, 2, 3))) for signs in product((1, -1), repeat=3)]


def test_brute_force_finds_and_checks():
    x, checked = C.brute_force_sat(FORMULA, 4)
    assert x == (False, False, False, True) and checked == 2
    assert C.count_satisfying(FORMULA, 4) == 6


def test_unsatisfiable_formula_needs_every_assignment():
    assert C.brute_force_sat(UNSAT, 3) == (None, 8)
    for x in product((False, True), repeat=3):
        broken = [c for c in UNSAT if not C.satisfies([c], x)]
        assert len(broken) == 1


def test_conditional_plan_count():
    # Crying baby: 3 actions, 2 observations
    assert [C.conditional_plan_count(3, 2, h) for h in (1, 2, 3)] == [3, 27, 2187]
