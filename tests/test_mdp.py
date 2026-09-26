"""Also pins the numbers quoted in the Ch 7 page's worked example and solved problems."""

import numpy as np
import pytest

from adm.mdp import (
    greedy,
    lookahead,
    policy_evaluation_exact,
    policy_evaluation_iterative,
    policy_iteration,
    value_iteration,
)
from adm.page import show_source
from adm.problems.gridworld import ROVER


def two_state(entries):
    """Build (T, R) from {(s, a): (next-state probabilities, reward)}."""
    T, R = np.zeros((2, 2, 2)), np.zeros((2, 2))
    for (s, a), (probs, reward) in entries.items():
        T[s, a], R[s, a] = probs, reward
    return T, R


# Drone: states High=0, Low=1; actions deliver=0, charge=1.
DRONE = two_state({
    (0, 0): ([0.6, 0.4], 10), (0, 1): ([1.0, 0.0], 0),
    (1, 0): ([0.0, 1.0], -6), (1, 1): ([0.8, 0.2], 0),
})
# Machine: states Working=0, Broken=1; actions run=0, repair=1.
MACHINE = two_state({
    (0, 0): ([0.9, 0.1], 5), (0, 1): ([1.0, 0.0], -1),
    (1, 0): ([0.0, 1.0], 0), (1, 1): ([0.7, 0.3], -4),
})


def test_drone_worked_example():
    T, R = DRONE
    U, history = value_iteration(T, R, 0.9)
    assert np.allclose(history[1], [10, 0]) and np.allclose(history[2], [15.4, 7.2])
    assert np.allclose(U, [69.492, 61.017], atol=1e-3)
    assert np.allclose(policy_evaluation_exact(T, R, 0.9, np.array([0, 0])), [-25.217, -60.0], atol=1e-3)
    policy, U_pi, policies = policy_iteration(T, R, 0.9, np.array([0, 0]))
    assert [p.tolist() for p in policies] == [[0, 0], [1, 1], [0, 1]]
    assert np.allclose(U_pi, U, atol=1e-4)


def test_machine_problem():
    T, R = MACHINE
    U = np.zeros(2)
    expected = [([5, 0], [0, 0]), ([9.05, 0], [0, 0]), ([12.3305, 1.7015], [0, 1])]
    for values, actions in expected:
        policy, U = greedy(T, R, 0.9, U)
        assert np.allclose(U, values) and policy.tolist() == actions


def test_exact_evaluation_problem():
    T = np.array([[[0.5, 0.5]], [[0.0, 1.0]]])  # one action
    R = np.array([[2.0], [1.0]])
    assert np.allclose(policy_evaluation_exact(T, R, 0.5, np.array([0, 0])), [10 / 3, 2])


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99])
def test_value_and_policy_iteration_agree_on_rover(gamma):
    T, R = ROVER.mdp()
    delta = 1e-8
    U, _ = value_iteration(T, R, gamma, delta=delta)
    policy, U_pi, _ = policy_iteration(T, R, gamma, np.zeros(len(U), dtype=int))
    assert np.max(np.abs(U - U_pi)) < delta * gamma / (1 - gamma)  # residual bound
    assert np.array_equal(greedy(T, R, gamma, U)[0], policy)
    assert np.allclose(policy_evaluation_iterative(T, R, gamma, policy), U_pi)
    assert np.allclose(lookahead(T, R, gamma, U_pi).max(axis=1), U_pi)  # Bellman optimality


def test_show_source(capsys):
    show_source(lookahead)
    out = capsys.readouterr().out
    assert out.startswith("```python\ndef lookahead(") and out.rstrip().endswith("```")
