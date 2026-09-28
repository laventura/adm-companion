"""Tests for adm.games (Ch 24), pinning the numbers quoted on the page."""

import itertools

import numpy as np

from adm import games as G
from adm.problems.simple_games import (SimpleGame, intersection, prisoners_dilemma,
                                       rock_paper_scissors, travelers_dilemma)


def test_utility_matches_brute_force_three_players():
    rng = np.random.default_rng(0)
    game = SimpleGame(rng.normal(size=(3, 2, 3, 4)), (("a", "b"), tuple("xyz"), tuple("pqrs")))
    pis = [rng.dirichlet(np.ones(n)) for n in (2, 3, 4)]
    for i in range(3):
        brute = sum(game.R[(i,) + a] * pis[0][a[0]] * pis[1][a[1]] * pis[2][a[2]]
                    for a in itertools.product(range(2), range(3), range(4)))
        assert np.isclose(G.utility(game, pis, i), brute)


def test_dominant_strategies():
    assert G.dominant_strategy_equilibrium(prisoners_dilemma()) == (1, 1)       # both defect
    assert G.dominant_strategy_equilibrium(rock_paper_scissors()) is None
    assert G.dominant_strategy_equilibrium(intersection()) is None


def test_intersection_equilibria():
    game = intersection()
    eqs = G.support_enumeration(game)
    assert len(eqs) == 3                                  # two pure, one mixed
    mixed = [e for e in eqs if 0 < e[0][0] < 1][0]
    assert np.allclose(mixed[0], [2 / 7, 5 / 7]) and np.isclose(G.utility(game, list(mixed), 0), -5 / 7)
    light = G.correlated_equilibrium(game, "egalitarian")
    assert np.allclose(G.correlated_utilities(game, light), [1.5, 1.5], atol=1e-7)
    assert np.isclose(G.correlated_utilities(game, G.correlated_equilibrium(game)).sum(), 3.0)


def test_nash_solver_and_rps():
    rps = rock_paper_scissors()
    pis, r = G.nash_equilibrium(rps, np.random.default_rng(0))
    assert r < 1e-6 and np.allclose(pis[0], 1 / 3, atol=1e-4)
    assert np.allclose(G.correlated_equilibrium(rps), 1 / 9, atol=1e-7)


def test_travelers_dilemma():
    td = travelers_dilemma()
    two = np.eye(99)[0]
    assert G.regret(td, [two, two]) == 0                  # $2 is an equilibrium
    h = G.iterated_best_response(td, 120)
    claims = [int(np.argmax(p[0])) + 2 for p in h[1:]]
    assert claims[:3] == [96, 95, 94] and claims.index(2) == 94   # round 95 reaches $2
    pis, r = G.nash_equilibrium(td, np.random.default_rng(0), starts=1)
    assert r < 1e-5                                       # only approximately an equilibrium
    assert int(np.argmax(pis[0])) + 2 == 51               # ... and far from $2


def test_iterated_best_response_cycles_in_rps():
    h = G.iterated_best_response(rock_paper_scissors(), 7)
    assert [int(np.argmax(p[0])) for p in h[1:]] == [0, 1, 2, 0, 1, 2, 0]


def test_fictitious_play():
    ix = intersection()
    a = G.fictitious_play(ix, 2000)["actions"]
    assert np.all(a[:, 0] == a[:, 1])                     # identical counts: lockstep play
    assert abs((a[:, 0] == 0).mean() - 2 / 7) < 0.01
    rps = rock_paper_scissors()
    a = G.fictitious_play(rps, 2000)["actions"]
    assert np.abs(np.bincount(a[:, 0], minlength=3) / 2000 - 1 / 3).max() < 0.01


def test_projection_and_softmax():
    assert np.allclose(G.project_simplex(np.array([0.7, 0.5, -0.1])), [0.6, 0.4, 0.0])
    td = travelers_dilemma()
    assert np.allclose(G.hierarchical_softmax(td, 0.3, 0)[0], 1 / 99)
    p = G.softmax_response(rock_paper_scissors(), [None, np.array([0.5, 0.3, 0.2])], 0, 0.0)
    assert np.allclose(p, 1 / 3)
