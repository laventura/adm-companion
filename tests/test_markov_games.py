"""Tests for adm.markov_games (Ch 25), pinning the numbers quoted on the page."""

import itertools

import numpy as np

from adm import markov_games as M
from adm.mdp import policy_iteration
from adm.problems.markov_games import (PREDATOR_PREY_TILES, PREDATOR_START, PREY_START, lake,
                                       predator_prey)

LIGHT_UNLESS_HIGH = np.eye(2)[[0, 0, 1]]
ALWAYS_HEAVY = np.eye(2)[[1, 1, 1]]


def test_models_are_distributions():
    for g in (lake(), predator_prey()):
        assert np.allclose(g.T.sum(-1), 1)


def test_evaluation_matches_simulation():
    g, rng = lake(), np.random.default_rng(0)
    U = M.policy_evaluation(g, [LIGHT_UNLESS_HIGH, LIGHT_UNLESS_HIGH])
    rets = []
    for _ in range(3000):
        s, ret = 2, 0.0
        for t in range(120):
            a = (int(LIGHT_UNLESS_HIGH[s, 1]), int(LIGHT_UNLESS_HIGH[s, 1]))
            ret += g.gamma ** t * g.R[(0, s) + a]
            s = M.step(g, s, a, rng)
        rets.append(ret)
    assert abs(np.mean(rets) - U[0, 2]) < 3 * np.std(rets) / np.sqrt(3000)


def test_lake_has_exactly_two_pure_equilibria():
    g, found = lake(), []
    for c1, c2 in itertools.product(itertools.product(range(2), repeat=3), repeat=2):
        pols = [np.eye(2)[list(c1)], np.eye(2)[list(c2)]]
        if M.exploitability(g, pols) < 1e-9:
            found.append((c1, c2))
    assert found == [((0, 0, 1), (0, 0, 1)), ((1, 1, 1), (1, 1, 1))]
    # The good equilibrium is also the team optimum.
    T, R = g.T.reshape(3, 4, 3), (g.R[0] + g.R[1]).reshape(3, 4)
    pi, U = policy_iteration(T, R, g.gamma, np.zeros(3, int))[:2]
    good = M.policy_evaluation(g, [LIGHT_UNLESS_HIGH] * 2)
    assert np.allclose(U, good.sum(0)) and list(pi) == [0, 0, 3]


def test_nash_program_finds_an_equilibrium():
    pols, e = M.nash_equilibrium(lake(), np.random.default_rng(0))
    assert e < 1e-6


def test_iterated_best_response():
    g = lake()
    h = M.iterated_best_response(g, 3)
    assert all(np.array_equal(p[0], LIGHT_UNLESS_HIGH) for p in h[1:])
    pp = predator_prey()
    h = M.iterated_best_response(pp, 12)
    assert M.exploitability(pp, h[-1]) > 50            # it keeps chasing its tail


def test_learning_in_the_lake():
    g = lake()
    fp = M.fictitious_play(g, 5000, np.random.default_rng(1), s0=2)["trace"][-500:]
    assert all((s == 2) == (a1 == 1 and a2 == 1) for s, a1, a2 in fp)   # good equilibrium
    _, pols, _ = M.nash_q_learning(g, 10000, np.random.default_rng(0), s0=2, eps=0.2)
    assert M.exploitability(g, pols) < 1e-6
    _, pols, _ = M.nash_q_learning(g, 10000, np.random.default_rng(0), s0=2)
    assert M.exploitability(g, pols) > 1                 # the book's schedule explores too little


def test_predator_prey_fictitious_play_improves():
    pp = predator_prey()
    n = len(PREDATOR_PREY_TILES)
    s0 = PREDATOR_PREY_TILES.index(PREDATOR_START) * n + PREDATOR_PREY_TILES.index(PREY_START)
    ex = []
    for steps in (1000, 20000):
        N = M.fictitious_play(pp, steps, np.random.default_rng(0), s0=s0, reset=10)["N"]
        ex.append(M.exploitability(pp, [N[i] / N[i].sum(1, keepdims=True) for i in range(2)]))
    assert ex[1] < ex[0] - 20
