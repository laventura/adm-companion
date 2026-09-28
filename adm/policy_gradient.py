"""Policy gradient estimation (ADM Ch. 11): estimate the gradient of
U(theta), the expected discounted return, from simulated trajectories.

Two kinds of estimator:
- finite difference and regression only need U estimates (black box);
- likelihood ratio estimators need grad log pi(a | s) along m sampled
  trajectories, passed as an array g of shape (m, d, n), together with
  the rewards r of shape (m, d): row i is trajectory i, column k step k+1,
  and n is the number of policy parameters.

Lines stay under 80 characters because the Ch 11 page prints this source.
"""

import numpy as np

Array = np.ndarray


def finite_difference_gradient(U, theta, delta: float) -> Array:
    """Forward differences: (U(theta + delta e_i) - U(theta)) / delta for
    each coordinate i. U may be noisy; each call is a fresh estimate."""
    theta = np.asarray(theta, dtype=float)
    u0 = U(theta)
    grad = np.zeros(len(theta))
    for i in range(len(theta)):
        step = np.zeros(len(theta))
        step[i] = delta
        grad[i] = (U(theta + step) - u0) / delta
    return grad


def regression_gradient(U, theta, delta: float, m: int, rng) -> Array:
    """Perturb theta in m random directions of length delta, record how U
    changes, and fit the gradient by least squares: grad ~ dTheta^+ dU."""
    theta = np.asarray(theta, dtype=float)
    directions = rng.standard_normal((m, len(theta)))
    dtheta = delta * directions / np.linalg.norm(directions, axis=1,
                                                 keepdims=True)
    u0 = U(theta)
    du = np.array([U(theta + dt) - u0 for dt in dtheta])
    return np.linalg.pinv(dtheta) @ du


def discounted_returns(r: Array, gamma: float) -> Array:
    """R(tau) = sum_k gamma^(k-1) r_k for each trajectory (row)."""
    return r @ gamma ** np.arange(r.shape[1])


def rewards_to_go(r: Array, gamma: float) -> Array:
    """rtg[i, k] = sum over l >= k of gamma^(l-k) r[i, l]: the discounted
    reward from step k on, discounted back to step k."""
    rtg = np.zeros_like(r)
    running = np.zeros(r.shape[0])
    for k in range(r.shape[1] - 1, -1, -1):      # sweep backward in time
        running = r[:, k] + gamma * running
        rtg[:, k] = running
    return rtg


def likelihood_ratio_gradient(g: Array, r: Array, gamma: float) -> Array:
    """Average over trajectories of (sum_k grad log pi_k) * R(tau)."""
    score = g.sum(axis=1)                        # (m, n): grad log p(tau)
    return (score * discounted_returns(r, gamma)[:, None]).mean(axis=0)


def reward_to_go_gradient(g: Array, r: Array, gamma: float) -> Array:
    """Like the likelihood ratio, but each action is credited only with
    rewards that come after it: sum_k grad log pi_k gamma^(k-1) rtg_k."""
    discount = gamma ** np.arange(r.shape[1])     # gamma^(k-1)
    weights = discount * rewards_to_go(r, gamma)  # (m, d)
    return (g * weights[:, :, None]).sum(axis=1).mean(axis=0)


def baseline_gradient(g: Array, r: Array, gamma: float) -> Array:
    """Reward-to-go minus, for each parameter i, the constant baseline that
    minimizes the estimate's variance: E[l_i^2 rtg] / E[l_i^2], where
    l_i = gamma^(k-1) d/d theta_i log pi. Pooled over all steps here."""
    discount = gamma ** np.arange(r.shape[1])
    ell = g * discount[None, :, None]              # (m, d, n)
    rtg = rewards_to_go(r, gamma)[:, :, None]      # (m, d, 1)
    base = (ell**2 * rtg).sum(axis=(0, 1)) / (ell**2).sum(axis=(0, 1))
    return (ell * (rtg - base)).sum(axis=1).mean(axis=0)
