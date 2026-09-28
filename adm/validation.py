"""Policy validation tools (ADM Ch. 14): Monte Carlo estimates with their
error bars, importance sampling, and Pareto fronts. The collision
avoidance problem in adm/problems/collision_avoidance.py supplies the
exact evaluation, robust dynamic programming and adversary.
"""

import numpy as np
from scipy import stats

Array = np.ndarray


def mc_estimate(x: Array, level: float = 0.95):
    """Mean of samples x with its standard error (eq. 14.5), relative
    standard error (eq. 14.6) and a normal confidence interval.
    Returns (mean, se, relative_se, (low, high))."""
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))      # sample sd / sqrt(n)
    z = stats.norm.ppf(0.5 + level / 2)       # 1.96 for 95%
    rel = se / mean if mean != 0 else np.inf
    return mean, se, rel, (mean - z * se, mean + z * se)


def beta_interval(k: int, n: int, level: float = 0.95):
    """Bayesian interval for a probability after k successes in n trials,
    from the Beta(1 + k, 1 + n - k) posterior of a uniform prior (Ch 4).
    Unlike the normal interval, it never goes below 0."""
    post = stats.beta(1 + k, 1 + n - k)
    return post.ppf(0.5 - level / 2), post.ppf(0.5 + level / 2)


def importance_estimate(f: Array, log_w: Array):
    """Importance sampling estimate of E_P[f] from samples drawn from a
    proposal P', with log_w = log P(tau) - log P'(tau) (eq. 14.12).

    Returns (mean, se, ess). The effective sample size
    ess = (sum w)^2 / sum w^2 says how many direct samples the weighted
    ones are worth for estimating the weights' spread; it collapses when
    a few huge weights dominate.
    """
    w = np.exp(np.asarray(log_w, dtype=float))
    fw = np.asarray(f, dtype=float) * w       # each sample's contribution
    n = len(fw)
    return fw.mean(), fw.std(ddof=1) / np.sqrt(n), w.sum()**2 / (w**2).sum()


def pareto_front(points: Array) -> Array:
    """Boolean mask of the rows of points (n, k) that no other row
    dominates, where every metric is to be minimized. Row j dominates
    row i if it is no worse on every metric and better on at least one."""
    P = np.asarray(points, dtype=float)
    # no_worse[i, j]: row j is <= row i everywhere; better: < somewhere.
    no_worse = (P[None, :, :] <= P[:, None, :]).all(axis=2)
    better = (P[None, :, :] < P[:, None, :]).any(axis=2)
    return ~(no_worse & better).any(axis=1)
