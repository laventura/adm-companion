"""Appendix A: information measures, gradient ascent and the contraction
property of the Bellman backup. Monte Carlo and importance sampling live
in adm.validation (Ch. 14).
"""

import numpy as np

Array = np.ndarray


def entropy(p, base=2.0) -> float:
    """H(P) = -sum_x P(x) log P(x); terms with P(x) = 0 contribute 0."""
    p = np.asarray(p, dtype=float)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum() / np.log(base))


def cross_entropy(p, q, base=2.0) -> float:
    """H(P, Q) = -sum_x P(x) log Q(x): the average surprise of outcomes
    from P under a model Q. Infinite if Q misses something P can do."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    nz = p > 0
    if np.any(q[nz] == 0):
        return np.inf
    return float(-(p[nz] * np.log(q[nz])).sum() / np.log(base))


def kl_divergence(p, q, base=2.0) -> float:
    """D_KL(P || Q) = H(P, Q) - H(P) >= 0, zero only when P = Q."""
    return cross_entropy(p, q, base) - entropy(p, base)


def gradient_ascent(grad, x0, alpha, k_max) -> Array:
    """x <- x + alpha grad f(x), k_max times. Returns every iterate."""
    xs = [np.asarray(x0, dtype=float)]
    for _ in range(k_max):
        xs.append(xs[-1] + alpha * np.asarray(grad(xs[-1])))
    return np.array(xs)


def bellman_backup(T, R, gamma, U) -> Array:
    """(B U)(s) = max_a [R(s, a) + gamma sum_s2 T(s2 | s, a) U(s2)]."""
    return (R + gamma * T @ U).max(axis=1)      # T @ U sums over s2
