"""Tests for adm.neural (Appendix D)."""

import numpy as np
import pytest

from adm import neural as N


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "tanh", "sigmoid", "identity"])
def test_backprop_matches_finite_differences(act):
    rng = np.random.default_rng(1)
    params = N.init_mlp([3, 5, 4, 2], rng)
    X, Y = rng.normal(size=(7, 3)), rng.normal(size=(7, 2))
    _, grads = N.mse_loss(params, X, Y, act, beta=0.01)
    h = 1e-6
    for k, (W, b) in enumerate(params):
        for P, G in ((W, grads[k][0]), (b, grads[k][1])):
            for idx in np.ndindex(P.shape):
                old = P[idx]
                P[idx] = old + h
                up = N.mse_loss(params, X, Y, act, beta=0.01)[0]
                P[idx] = old - h
                down = N.mse_loss(params, X, Y, act, beta=0.01)[0]
                P[idx] = old
                assert np.isclose((up - down) / (2 * h), G[idx], atol=1e-5)


def test_softmax():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    s = N.softmax(z)
    assert np.allclose(s.sum(axis=1), 1) and np.allclose(s[1], 1 / 3)
    assert np.allclose(N.softmax(z[:1] + 5), s[:1])


def test_training_fits_to_noise_level():
    rng = np.random.default_rng(0)
    f = lambda x: np.sin(2 * x) * np.exp(-x ** 2 / 4)
    X = rng.uniform(-3, 3, (60, 1))
    Y = f(X) + rng.normal(0, 0.05, X.shape)
    params, losses = N.train(N.init_mlp([1, 16, 16, 1], rng), X, Y, "tanh",
                             alpha=0.01, epochs=1500)
    grid = np.linspace(-3, 3, 200)[:, None]
    assert losses[-1] < 0.004 and losses[-1] < losses[0] / 20
    assert np.mean((N.forward(params, grid, "tanh")[0] - f(grid)) ** 2) < 0.005


def test_linear_layers_collapse():
    rng = np.random.default_rng(2)
    params = N.init_mlp([2, 6, 3], rng)
    X = rng.normal(size=(5, 2))
    (W1, b1), (W2, b2) = params
    assert np.allclose(N.forward(params, X, "identity")[0], X @ (W1 @ W2) + (b1 @ W2 + b2))
