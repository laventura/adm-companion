"""Appendix D: a small feedforward network (multilayer perceptron) in NumPy,
with backpropagation written out by hand and Adam updates.

Shapes: a batch X has one sample per row, (n, d_in). A layer is a pair
(W, b) with W of shape (d_in, d_out), so a layer computes X @ W + b.
Hidden layers apply the activation; the output layer is linear.
"""

import numpy as np

Array = np.ndarray

# Each activation is (phi, dphi): the function and its derivative.
ACTIVATIONS = {
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0) * 1.0),
    "leaky_relu": (lambda z: np.where(z > 0, z, 0.1 * z),
                   lambda z: np.where(z > 0, 1.0, 0.1)),
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)),
                lambda z: np.exp(-z) / (1 + np.exp(-z)) ** 2),
    "identity": (lambda z: z, lambda z: np.ones_like(z)),
}


def softmax(z, axis=-1) -> Array:
    """exp(z_i) / sum_j exp(z_j). Subtracting the max first changes
    nothing mathematically but avoids overflow in exp."""
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def init_mlp(sizes, rng) -> list:
    """Layers for sizes = [d_in, h1, ..., d_out]. Weights ~ N(0, 1/d_in),
    so each unit's input starts with variance about 1."""
    return [(rng.normal(0, 1 / np.sqrt(m), (m, n)), np.zeros(n))
            for m, n in zip(sizes[:-1], sizes[1:])]


def forward(params, X, act="tanh"):
    """Returns the output and a cache of (layer input, pre-activation)
    per layer, which the backward pass needs."""
    phi = ACTIVATIONS[act][0]
    A, cache = X, []
    for k, (W, b) in enumerate(params):
        Z = A @ W + b                   # (n, d_in) @ (d_in, d_out)
        cache.append((A, Z))
        A = Z if k == len(params) - 1 else phi(Z)   # linear output layer
    return A, cache


def backward(params, cache, dA, act="tanh") -> list:
    """Chain rule from the output back: dA is dLoss/dOutput, (n, d_out).
    Returns [(dW, db)] matching params."""
    dphi = ACTIVATIONS[act][1]
    grads = [None] * len(params)
    for k in reversed(range(len(params))):
        A_prev, Z = cache[k]
        # Through the activation (skipped at the linear output layer)
        dZ = dA if k == len(params) - 1 else dA * dphi(Z)
        W = params[k][0]
        # Z = A_prev @ W + b: each weight's gradient sums over the batch
        grads[k] = (A_prev.T @ dZ, dZ.sum(axis=0))
        dA = dZ @ W.T                   # hand the gradient to layer k - 1
    return grads


def mse_loss(params, X, Y, act="tanh", beta=0.0):
    """Mean squared error plus beta * (sum of squared weights), and its
    gradient. Biases are not penalized."""
    Yhat, cache = forward(params, X, act)
    err = Yhat - Y
    loss = (err ** 2).sum() / len(X) + beta * sum((W ** 2).sum()
                                                  for W, _ in params)
    grads = backward(params, cache, 2 * err / len(X), act)
    grads = [(dW + 2 * beta * W, db) for (dW, db), (W, _) in
             zip(grads, params)]
    return loss, grads


def train(params, X, Y, act="tanh", alpha=0.01, epochs=2000, beta=0.0,
          batch=None, rng=None):
    """Adam on the loss above: each step moves each parameter by alpha
    times its running mean gradient over the root of its running mean
    squared gradient. batch=None uses all data every step.
    Returns trained params and the loss per epoch."""
    rng = rng or np.random.default_rng(0)
    b1, b2, eps = 0.9, 0.999, 1e-8
    flat = [p for layer in params for p in layer]       # W1, b1, W2, ...
    flat = [p.copy() for p in flat]
    m = [np.zeros_like(p) for p in flat]
    v = [np.zeros_like(p) for p in flat]
    t, losses = 0, []
    for _ in range(epochs):
        order = rng.permutation(len(X))
        size = batch or len(X)
        for start in range(0, len(X), size):
            idx = order[start:start + size]
            layers = list(zip(flat[::2], flat[1::2]))
            _, grads = mse_loss(layers, X[idx], Y[idx], act, beta)
            t += 1
            for i, g in enumerate(p for layer in grads for p in layer):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g ** 2
                mhat, vhat = m[i] / (1 - b1 ** t), v[i] / (1 - b2 ** t)
                flat[i] -= alpha * mhat / (np.sqrt(vhat) + eps)
        layers = list(zip(flat[::2], flat[1::2]))
        losses.append(mse_loss(layers, X, Y, act, beta)[0])
    return layers, np.array(losses)
