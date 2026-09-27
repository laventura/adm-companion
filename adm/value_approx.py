"""Approximate value functions (ADM Ch. 8).

Every approximator U_theta here works the same way:
  U(states)      -> one value per state; `states` has shape (n, d)
  U.fit(S, u)    -> adjust theta so that U(S) matches the targets u
Local methods (nearest neighbor, kernel smoothing, multilinear and simplex
interpolation) store one value per sample state; global methods (linear
regression, neural networks) have parameters not tied to any one state.

Lines stay under 80 characters because the Ch 8 page prints this source.
"""

from itertools import product

import numpy as np

Array = np.ndarray


class NearestNeighborValue:
    """U(s) = mean of theta over the k sample states closest to s."""

    def __init__(self, S: Array, k: int = 1):
        self.S, self.k = np.asarray(S, dtype=float), k
        self.theta = np.zeros(len(self.S))

    def fit(self, S: Array, u: Array):
        self.theta = np.asarray(u, dtype=float)   # local: theta = targets
        return self

    def __call__(self, states: Array) -> Array:
        # dist[i, j] = distance from query state i to sample state j.
        diff = np.asarray(states)[:, None, :] - self.S[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        nearest = np.argsort(dist, axis=1)[:, :self.k]
        return self.theta[nearest].mean(axis=1)


class KernelValue:
    """U(s) = sum_i beta_i(s) theta_i, with beta_i(s) proportional to a
    kernel k(s, s_i) and normalized so the weights sum to 1."""

    def __init__(self, S: Array, kernel):
        self.S, self.kernel = np.asarray(S, dtype=float), kernel
        self.theta = np.zeros(len(self.S))

    def fit(self, S: Array, u: Array):
        self.theta = np.asarray(u, dtype=float)
        return self

    def weights(self, states: Array) -> Array:
        """beta[i, j]: weight of sample state j when evaluating state i."""
        diff = np.asarray(states)[:, None, :] - self.S[None, :, :]
        k = self.kernel(np.linalg.norm(diff, axis=-1))
        return k / k.sum(axis=1, keepdims=True)

    def __call__(self, states: Array) -> Array:
        return self.weights(states) @ self.theta


def inverse_distance(eps: float = 1e-6):
    """Kernel k = 1 / max(d, eps): closer sample states count for more."""
    return lambda d: 1.0 / np.maximum(d, eps)


def gaussian(sigma: float):
    """Kernel k = exp(-d^2 / (2 sigma^2)); sigma sets how far it reaches."""
    return lambda d: np.exp(-d**2 / (2 * sigma**2))


class _Grid:
    """A regular grid from `lo` to `hi` with `shape` points per axis.
    theta holds one value per grid point, shaped like the grid."""

    def __init__(self, lo, hi, shape):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.shape = tuple(shape)
        self.width = (self.hi - self.lo) / (np.array(self.shape) - 1)
        self.theta = np.zeros(self.shape)

    def points(self) -> Array:
        """All grid points as rows, in the same (C) order as theta.ravel()."""
        axes = [np.linspace(l, h, n)
                for l, h, n in zip(self.lo, self.hi, self.shape)]
        return np.array(list(product(*axes)))

    def fit(self, S: Array, u: Array):
        # S must be self.points(), so the targets fill the grid in order.
        self.theta = np.asarray(u, dtype=float).reshape(self.shape)
        return self

    def _cell(self, states: Array):
        """Lower-corner index of each state's cell, and its position inside
        the cell as a fraction from 0 to 1 along each axis."""
        z = (np.asarray(states, float) - self.lo) / self.width
        # Clip so states on the upper edge use the last cell (fraction 1).
        base = np.clip(np.floor(z).astype(int), 0, np.array(self.shape) - 2)
        return base, np.clip(z - base, 0.0, 1.0)


class MultilinearValue(_Grid):
    """Interpolate linearly along each axis in turn: each of the cell's 2^d
    corners is weighted by the volume of the box opposite it."""

    def __call__(self, states: Array) -> Array:
        base, f = self._cell(states)
        u = np.zeros(len(base))
        # corner is a 0/1 vector: 1 means the upper neighbor on that axis.
        for corner in product((0, 1), repeat=len(self.shape)):
            c = np.array(corner)
            weight = np.prod(np.where(c == 1, f, 1 - f), axis=1)
            u += weight * self.theta[tuple((base + c).T)]
        return u


class SimplexValue(_Grid):
    """Split each cell into d! simplexes and interpolate among the d + 1
    corners of the one containing s (Freudenthal triangulation)."""

    def __call__(self, states: Array) -> Array:
        base, f = self._cell(states)
        n, d = f.shape
        rows = np.arange(n)
        # Walk from the lower corner to the upper corner, stepping up one
        # axis at a time, largest fraction first. The weights are the gaps
        # between successive sorted fractions (they sum to 1).
        order = np.argsort(-f, axis=1)
        f_sorted = np.concatenate(
            [np.ones((n, 1)), f[rows[:, None], order], np.zeros((n, 1))],
            axis=1)
        vertex = base.copy()
        u = (f_sorted[:, 0] - f_sorted[:, 1]) * self.theta[tuple(vertex.T)]
        for j in range(d):
            vertex[rows, order[:, j]] += 1
            w = f_sorted[:, j + 1] - f_sorted[:, j + 2]
            u += w * self.theta[tuple(vertex.T)]
        return u


class LinearRegressionValue:
    """U(s) = theta . beta(s), with theta chosen by least squares."""

    def __init__(self, basis):
        self.basis = basis        # basis(states) -> (n, number of features)
        self.theta = None

    def fit(self, S: Array, u: Array):
        X = self.basis(np.asarray(S, float))   # row i is beta(s_i)
        # lstsq minimizes ||X theta - u||^2 (via the pseudoinverse of X).
        self.theta = np.linalg.lstsq(X, u, rcond=None)[0]
        return self

    def __call__(self, states: Array) -> Array:
        return self.basis(np.asarray(states, float)) @ self.theta


def polynomial_basis(degree: int, lo, hi):
    """All monomials x1^i x2^j ... with total degree <= degree. States are
    first rescaled to [-1, 1] per axis so high powers stay well scaled."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)

    def basis(states):
        z = 2 * (states - lo) / (hi - lo) - 1
        d = z.shape[1]
        powers = [p for p in product(range(degree + 1), repeat=d)
                  if sum(p) <= degree]
        return np.stack([np.prod(z**np.array(p), axis=1) for p in powers],
                        axis=1)
    return basis


def fourier_basis(order: int, lo, hi):
    """Products of 1, sin and cos terms up to `order` periods across each
    axis's range: every combination of one term per axis."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)

    def basis(states):
        z = (states - lo) / (hi - lo)          # 0 to 1 across each axis
        per_axis = []
        for k in range(z.shape[1]):
            t = 2 * np.pi * z[:, k:k + 1]
            i = np.arange(1, order + 1)
            per_axis.append(np.hstack([np.full((len(z), 1), 0.5),
                                       np.sin(i * t), np.cos(i * t)]))
        feats = per_axis[0]
        for more in per_axis[1:]:              # all pairwise products
            feats = (feats[:, :, None] * more[:, None, :]).reshape(
                len(z), -1)
        return feats
    return basis


class NeuralNetValue:
    """U(s) = w2 . tanh(W1 s + b1) + b2: one hidden layer, fitted by
    gradient descent on the squared error (Adam step sizes)."""

    def __init__(self, d: int, hidden: int = 16, steps: int = 3000,
                 lr: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p = {"W1": rng.normal(0, 1, (d, hidden)),
                  "b1": np.zeros(hidden),
                  "w2": rng.normal(0, 1 / np.sqrt(hidden), hidden),
                  "b2": np.zeros(())}
        self.steps, self.lr = steps, lr

    def __call__(self, states: Array) -> Array:
        h = np.tanh(np.asarray(states, float) @ self.p["W1"] + self.p["b1"])
        return h @ self.p["w2"] + self.p["b2"]

    def fit(self, S: Array, u: Array):
        S, u = np.asarray(S, float), np.asarray(u, float)
        m = {k: 0 * v for k, v in self.p.items()}   # Adam: running mean
        v = {k: 0 * x for k, x in self.p.items()}   # and running square
        for t in range(1, self.steps + 1):
            h = np.tanh(S @ self.p["W1"] + self.p["b1"])
            err = h @ self.p["w2"] + self.p["b2"] - u  # prediction - target
            # Chain rule for the mean squared error, layer by layer.
            dh = np.outer(err, self.p["w2"]) * (1 - h**2) * 2 / len(u)
            g = {"w2": 2 * h.T @ err / len(u), "b2": 2 * err.mean(),
                 "W1": S.T @ dh, "b1": dh.sum(axis=0)}
            for k in self.p:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k]**2
                m_hat, v_hat = m[k] / (1 - 0.9**t), v[k] / (1 - 0.999**t)
                self.p[k] = self.p[k] - self.lr * m_hat / (
                    np.sqrt(v_hat) + 1e-8)
        return self


def approximate_value_iteration(U, S: Array, step, actions, reward: float,
                                gamma: float, is_terminal, k_max: int,
                                history: list | None = None):
    """Alternate Bellman backups at the sample states S with refitting U.

    For a deterministic problem, the backup at s is
      max_a [ reward + gamma * U(step(s, a)) ],
    with U = 0 once the next state is terminal. Terminal sample states
    keep the value 0. Appends each iteration's targets to `history`.
    """
    S = np.asarray(S, float)
    done = is_terminal(S)
    u = np.zeros(len(S))
    U.fit(S, u)
    for _ in range(k_max):
        q = []
        for a in actions:
            S2 = step(S, a)
            q.append(reward + gamma * np.where(is_terminal(S2), 0.0, U(S2)))
        u = np.where(done, 0.0, np.max(q, axis=0))
        U.fit(S, u)
        if history is not None:
            history.append(u.copy())
    return U


def greedy_actions(U, states: Array, step, actions, reward: float,
                   gamma: float, is_terminal) -> Array:
    """For each state, the action with the best one-step lookahead value
    under U (the policy of equation 8.1, for a deterministic problem)."""
    states = np.asarray(states, float)
    q = []
    for a in actions:
        S2 = step(states, a)
        q.append(reward + gamma * np.where(is_terminal(S2), 0.0, U(S2)))
    # q has one row per action; argmax over rows picks the best action.
    return np.asarray(actions)[np.argmax(q, axis=0)]
