"""Approximate value functions (Ch 8) and the mountain car problem."""

from itertools import groupby

import numpy as np
import pytest

from adm.problems import mountain_car as mc
from adm.value_approx import (
    KernelValue, LinearRegressionValue, MultilinearValue, NearestNeighborValue,
    NeuralNetValue, SimplexValue, approximate_value_iteration, fourier_basis,
    gaussian, greedy_actions, inverse_distance, polynomial_basis,
)

LO, HI = np.array([-1.2, -0.07]), np.array([0.6, 0.07])
rng = np.random.default_rng(0)


def test_mountain_car_step():
    s = mc.step([-0.5, 0.0], 1)
    v = 0.001 - 0.0025 * np.cos(-1.5)
    assert np.allclose(s, [-0.5 + v, v])
    assert mc.step([0.59, 0.07], 1)[0] == 0.6 and mc.is_goal([0.6, 0.0])
    assert mc.step([-1.2, -0.07], -1)[0] == -1.2        # left wall
    assert mc.step([-1.0, 0.07], 1)[1] == 0.07          # top speed
    # Vectorized over states and actions.
    many = mc.step(np.array([[-0.5, 0.0], [0.0, 0.0]]), np.array([1, -1]))
    assert np.allclose(many[0], s)


def test_unit_cell_hand_example():
    # The worked example on the page: corner values 10, 20, 30, 60.
    for cls in (MultilinearValue, SimplexValue):
        U = cls([0, 0], [1, 1], (2, 2))
        U.fit(U.points(), [10, 30, 20, 60])     # order (0,0),(0,1),(1,0),(1,1)
        expected = 27.5 if cls is MultilinearValue else 29.5
        assert U(np.array([[0.25, 0.6]]))[0] == pytest.approx(expected)


@pytest.mark.parametrize("cls", [MultilinearValue, SimplexValue])
def test_interpolation_exact_on_grid_and_linear(cls):
    U = cls([0, 0, 0], [2, 1, 3], (3, 4, 5))
    P = U.points()
    linear = lambda s: 1 + 2 * s[:, 0] - 3 * s[:, 1] + 0.5 * s[:, 2]
    U.fit(P, linear(P))
    assert np.allclose(U(P), linear(P))
    q = rng.uniform([0, 0, 0], [2, 1, 3], (200, 3))
    assert np.allclose(U(q), linear(q))            # linear data: exact


def test_multilinear_exact_on_bilinear():
    U = MultilinearValue([0, 0], [1, 1], (5, 5))
    f = lambda s: s[:, 0] * s[:, 1] + s[:, 0]
    U.fit(U.points(), f(U.points()))
    q = rng.uniform(0, 1, (100, 2))
    assert np.allclose(U(q), f(q))


def test_simplex_matches_book_style_weights():
    # Point (0.3, 0.7, 0.2) in the unit cube: the walk steps up axis 2
    # (0.7), then axis 1 (0.3), then axis 3 (0.2); weights 0.3, 0.4, 0.1,
    # 0.2 on corners (0,0,0), (0,1,0), (1,1,0), (1,1,1).
    U = SimplexValue([0, 0, 0], [1, 1, 1], (2, 2, 2))
    theta = np.zeros((2, 2, 2))
    theta[0, 0, 0], theta[0, 1, 0], theta[1, 1, 0], theta[1, 1, 1] = (
        1, 10, 100, 1000)
    U.theta = theta
    assert U(np.array([[0.3, 0.7, 0.2]]))[0] == pytest.approx(
        0.3 + 4 + 10 + 200)


def test_local_averagers():
    S = np.array([[0.0], [1.0], [3.0]])
    nn = NearestNeighborValue(S, 1).fit(S, [5, 7, 9])
    assert np.allclose(nn(np.array([[0.4], [2.1]])), [5, 9])
    assert NearestNeighborValue(S, 2).fit(S, [5, 7, 9])(
        np.array([[0.4]]))[0] == 6
    k = KernelValue(S, inverse_distance()).fit(S, [5, 7, 9])
    w = k.weights(np.array([[0.5], [1.0]]))
    assert np.allclose(w.sum(axis=1), 1)
    # Inverse distances 2, 2, 0.4 from s = 0.5 -> weights 2/4.4, 2/4.4, 0.4/4.4
    assert np.allclose(w[0], np.array([2, 2, 0.4]) / 4.4)
    assert k(np.array([[1.0]]))[0] == pytest.approx(7, abs=1e-4)
    g = KernelValue(S, gaussian(0.5)).fit(S, [5, 7, 9])
    assert 5 < g(np.array([[0.5]]))[0] < 7


def test_regression_bases():
    assert polynomial_basis(2, [0, 0], [1, 1])(np.zeros((3, 2))).shape == (3, 6)
    assert fourier_basis(2, [0, 0], [1, 1])(np.zeros((3, 2))).shape == (3, 25)
    # A cubic in two variables is fit exactly by a degree-3 basis.
    S = rng.uniform(LO, HI, (60, 2))
    f = lambda s: 3 + s[:, 0] ** 3 - 50 * s[:, 0] * s[:, 1]
    U = LinearRegressionValue(polynomial_basis(3, LO, HI)).fit(S, f(S))
    q = rng.uniform(LO, HI, (50, 2))
    assert np.allclose(U(q), f(q))


def test_neural_net_fits_a_smooth_curve():
    S = np.linspace(-1, 1, 25)[:, None]
    u = np.sin(3 * S[:, 0])
    U = NeuralNetValue(1, hidden=12, steps=4000, lr=0.02).fit(S, u)
    assert np.sqrt(np.mean((U(S) - u) ** 2)) < 0.05


@pytest.fixture(scope="module")
def mountain_car_41():
    U = MultilinearValue(LO, HI, (41, 41))
    history = []
    approximate_value_iteration(U, U.points(), mc.step, mc.ACTIONS, -1.0,
                                1.0, mc.is_goal, 400, history)
    return U, history


def test_avi_mountain_car_converges_and_drives_out(mountain_car_41):
    U, history = mountain_car_41
    assert np.max(np.abs(history[-1] - history[-2])) < 1e-5
    policy = lambda s: greedy_actions(U, s, mc.step, mc.ACTIONS, -1.0, 1.0,
                                      mc.is_goal)
    steps = mc.steps_to_goal(policy, [[-0.5, 0.0]])[0]
    assert steps == 108
    # The value is about minus the number of steps to the goal.
    assert abs(U(np.array([[-0.5, 0.0]]))[0] + steps) < 10
    # The car can't drive straight out: the policy rocks it right, left,
    # then right again (the page quotes 22, 39 and 46 steps).
    path = mc.rollout(lambda s: policy(s[None])[0], [-0.5, 0.0])
    actions = policy(path[:-1])
    runs = [(a, len(list(g))) for a, g in groupby(actions)]
    assert [n for _, n in runs[:3]] == [22, 39, 46]
    assert [a for a, _ in runs[:3]] == [1, -1, 1]


def test_nearest_neighbor_drifts_without_discount():
    # With gamma = 1, a sample state whose successors all stay nearest to
    # itself backs up u <- -1 + u forever.
    G = MultilinearValue(LO, HI, (21, 21)).points()
    scaled = lambda s: (np.asarray(s) - LO) / (HI - LO)

    class Scaled(NearestNeighborValue):
        def __call__(self, states):
            return super().__call__(scaled(states))
    U = Scaled(scaled(G), 1)
    history = []
    approximate_value_iteration(U, G, mc.step, mc.ACTIONS, -1.0, 1.0,
                                mc.is_goal, 300, history)
    assert history[-1].min() == -300
