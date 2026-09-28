"""Bernoulli bandits and exploration strategies (ADM Ch. 15).

Arm a pays 1 with unknown probability theta_a. With a uniform prior, the
belief about arm a after w_a wins and l_a losses is Beta(w_a + 1, l_a + 1),
so the counts (w, l) are the whole belief state.

Every strategy maps the counts to action probabilities with
`probs(w, l, t)`, where w and l have shape (..., n_arms) and t is the
number of pulls so far. That one interface serves both simulation
(`act` samples from it) and exact evaluation (`evaluate` averages over it).
"""

import itertools

import numpy as np
from scipy import stats

Array = np.ndarray


def posterior_mean(w: Array, l: Array) -> Array:
    """rho_a = P(win | w_a, l_a) = (w_a + 1) / (w_a + l_a + 2) (eq. 15.1)."""
    return (w + 1) / (w + l + 2)


def argmax_split(scores: Array) -> Array:
    """Probabilities that put equal weight on every arm tied for the
    highest score (last axis), so ties are broken uniformly at random."""
    best = np.isclose(scores, scores.max(axis=-1, keepdims=True), rtol=0, atol=1e-12)
    return best / best.sum(axis=-1, keepdims=True)


class Strategy:
    """Base class: `act` samples an arm from `probs` for each bandit."""

    def probs(self, w, l, t):
        raise NotImplementedError

    def act(self, w, l, t, rng):
        p = self.probs(w, l, t)                       # (..., n_arms)
        u = rng.random(p.shape[:-1] + (1,))
        # Inverse-CDF draw: count the cumulative probabilities below u.
        return np.minimum((u > np.cumsum(p, axis=-1)).sum(axis=-1), p.shape[-1] - 1)


class Greedy(Strategy):
    """Always an arm with the highest posterior mean."""

    def probs(self, w, l, t):
        return argmax_split(posterior_mean(w, l))


class EpsilonGreedy(Strategy):
    """A random arm with probability eps (decayed as eps * decay^t,
    eq. 15.2), otherwise a greedy one."""

    def __init__(self, eps, decay=1.0):
        self.eps, self.decay = eps, decay

    def probs(self, w, l, t):
        eps = self.eps * self.decay**t
        n = w.shape[-1]
        return eps / n + (1 - eps) * argmax_split(posterior_mean(w, l))


class ExploreThenCommit(Strategy):
    """Uniformly random arms for the first k pulls, then greedy."""

    def __init__(self, k):
        self.k = k

    def probs(self, w, l, t):
        if t < self.k:
            return np.full(w.shape, 1 / w.shape[-1])
        return argmax_split(posterior_mean(w, l))


class Softmax(Strategy):
    """P(a) proportional to exp(lam * rho_a), with the precision growing
    as lam * factor^t."""

    def __init__(self, lam, factor=1.0):
        self.lam, self.factor = lam, factor

    def probs(self, w, l, t):
        z = self.lam * self.factor**t * posterior_mean(w, l)
        z = np.exp(z - z.max(axis=-1, keepdims=True))  # subtract max: no overflow
        return z / z.sum(axis=-1, keepdims=True)


class Quantile(Strategy):
    """The arm whose posterior has the highest alpha-quantile: optimism
    under uncertainty when alpha > 0.5."""

    def __init__(self, alpha):
        self.alpha = alpha

    def probs(self, w, l, t):
        return argmax_split(stats.beta.ppf(self.alpha, w + 1, l + 1))


class UCB1(Strategy):
    """The arm maximizing rho_a + c sqrt(log N / N_a) (eq. 15.3), with
    the Beta pseudocounts N_a = w_a + l_a + 2 and N = sum_a N_a."""

    def __init__(self, c):
        self.c = c

    def probs(self, w, l, t):
        Na = w + l + 2
        N = Na.sum(axis=-1, keepdims=True)
        return argmax_split(posterior_mean(w, l) + self.c * np.sqrt(np.log(N) / Na))


# Gauss-Legendre nodes on [0, 1]. Beta densities and CDFs with integer
# parameters are polynomials, so this rule integrates them exactly while
# the total degree stays below 2 * 128.
_X, _WX = np.polynomial.legendre.leggauss(128)
_X, _WX = (_X + 1) / 2, _WX / 2


class Thompson(Strategy):
    """Posterior (Thompson) sampling: draw theta_a from each posterior and
    pull the arm with the largest draw. Its probabilities are
    P(a) = integral of pdf_a(x) prod_{b != a} cdf_b(x) dx."""

    def probs(self, w, l, t):
        x = _X[:, None]                                # (nodes, 1): broadcasts over arms
        pdf = stats.beta.pdf(x, w[..., None, :] + 1, l[..., None, :] + 1)
        cdf = stats.beta.cdf(x, w[..., None, :] + 1, l[..., None, :] + 1)
        # prod over the other arms = prod over all arms / own cdf, done in
        # logs; an arm's own cdf can be 0 only where its pdf is also 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            others = np.exp(np.log(cdf).sum(axis=-1, keepdims=True) - np.log(cdf))
        others = np.nan_to_num(others)
        p = (_WX[:, None] * pdf * others).sum(axis=-2)
        return p / p.sum(axis=-1, keepdims=True)

    def act(self, w, l, t, rng):
        return rng.beta(w + 1, l + 1).argmax(axis=-1)


def simulate(theta: Array, strategy: Strategy, h: int, rng):
    """Run a strategy on bandits side by side for h pulls.

    theta has shape (m, n_arms): row i is bandit i's win probabilities.
    Returns (actions, rewards), each of shape (m, h), and the final
    counts (w, l).
    """
    m, n = theta.shape
    w, l = np.zeros((m, n), dtype=int), np.zeros((m, n), dtype=int)
    rows = np.arange(m)
    A, R = np.zeros((m, h), dtype=int), np.zeros((m, h), dtype=int)
    for t in range(h):
        a = strategy.act(w, l, t, rng)
        r = (rng.random(m) < theta[rows, a]).astype(int)
        w[rows, a] += r                      # paired indexing: one arm per row
        l[rows, a] += 1 - r
        A[:, t], R[:, t] = a, r
    return A, R, (w, l)


def _layer(t: int, n: int) -> Array:
    """All count vectors (w_1, l_1, ..., w_n, l_n) with total t, as rows.
    Stars and bars: choose where the 2n - 1 dividers go among t + 2n - 1
    slots; the gaps between dividers are the counts."""
    k = 2 * n
    rows = []
    for bars in itertools.combinations(range(t + k - 1), k - 1):
        edges = (-1,) + bars + (t + k - 1,)
        rows.append([edges[i + 1] - edges[i] - 1 for i in range(k)])
    return np.array(rows, dtype=int).reshape(-1, k)


def evaluate(strategy, n: int, h: int, optimal=False):
    """Exact expected total reward of a strategy over h pulls, averaged
    over bandits drawn from the uniform prior (theta_a ~ U(0, 1)).

    Backward induction over belief states (the counts), one layer of
    total t at a time. With optimal=True it computes the Bayes-optimal
    value instead (eq. 15.6), ignoring `strategy`. Returns the value at
    the start (all counts 0).
    """
    V = np.zeros((h + 1,) * (2 * n))           # V[w_1, l_1, ..., w_n, l_n]
    for t in range(h - 1, -1, -1):
        S = _layer(t, n)                       # (m, 2n)
        w, l = S[:, 0::2], S[:, 1::2]          # wins and losses, (m, n)
        rho = posterior_mean(w, l)
        Q = np.zeros(w.shape)
        for a in range(n):
            win, loss = S.copy(), S.copy()
            win[:, 2 * a] += 1
            loss[:, 2 * a + 1] += 1
            # V[tuple(columns)] looks up one value per row of counts.
            Q[:, a] = (rho[:, a] * (1 + V[tuple(win.T)])
                       + (1 - rho[:, a]) * V[tuple(loss.T)])
        if optimal:
            V[tuple(S.T)] = Q.max(axis=1)
        else:
            V[tuple(S.T)] = (strategy.probs(w, l, t) * Q).sum(axis=1)
    return V[(0,) * (2 * n)]


def optimal_values(r_max: int, k_max: int):
    """Bayes-optimal values of a two-armed bandit with r pulls left.

    Returns U with U[r, w1, l1, w2, l2] for r = 0..r_max and counts up
    to k_max (states whose next counts would exceed k_max are left 0,
    so keep k_max >= largest total count + r). Q values for the two arms
    are then q_values(U, r, ...). Recursion (eq. 15.6):
      U_r(c) = max_a [rho_a (1 + U_{r-1}(c + win_a))
                      + (1 - rho_a) U_{r-1}(c + loss_a)].
    """
    K = k_max + 1
    U = np.zeros((r_max + 1, K, K, K, K))
    for r in range(1, r_max + 1):
        U[r, :-1, :-1, :-1, :-1] = np.max(q_values(U, r), axis=0)
    return U


def q_values(U, r):
    """Q[a, w1, l1, w2, l2] with r pulls left, for counts below k_max
    (the arrays are one shorter on each axis than U's count axes)."""
    K = U.shape[1]
    w1, l1, w2, l2 = np.meshgrid(*(np.arange(K - 1),) * 4, indexing="ij")
    prev = U[r - 1]
    rho1, rho2 = posterior_mean(w1, l1), posterior_mean(w2, l2)
    # Slicing shifts one count by 1: prev[1:, :-1, :-1, :-1] is
    # U_{r-1} at (w1 + 1, l1, w2, l2), a win for arm 1.
    q1 = rho1 * (1 + prev[1:, :-1, :-1, :-1]) + (1 - rho1) * prev[:-1, 1:, :-1, :-1]
    q2 = rho2 * (1 + prev[:-1, :-1, 1:, :-1]) + (1 - rho2) * prev[:-1, :-1, :-1, 1:]
    return np.array([q1, q2])
