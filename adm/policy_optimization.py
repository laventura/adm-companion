"""Policy gradient optimization (ADM Ch. 12): turn gradient estimates into
policy updates that make steady progress without overshooting.

Updates take a gradient estimate `grad` from any Chapter 11 estimator.
Those that reuse sampled trajectories take arrays S, A of shape (m, d)
(states and actions of m trajectories), a matching array `weights` of
per-step credits (for example discounted rewards-to-go or advantages),
and the policy's functions: logp(theta, s, a), grad_logp(theta, s, a)
(shape s.shape + (n,)) and kl(theta, theta2, s), elementwise over arrays.

Lines stay under 80 characters because the Ch 12 page prints this source.
"""

import numpy as np

from adm.policy_gradient import rewards_to_go

Array = np.ndarray


def scale_gradient(g: Array, max_norm: float) -> Array:
    """Shrink g to length max_norm if it is longer; keeps its direction."""
    norm = np.linalg.norm(g)
    return g * min(max_norm / norm, 1.0) if norm > 0 else g


def clip_gradient(g: Array, lo: float, hi: float) -> Array:
    """Clamp each component of g to [lo, hi]; can change its direction."""
    return np.clip(g, lo, hi)


def restricted_step(theta: Array, grad: Array, eps: float) -> Array:
    """Best step within the ball (1/2)||step||^2 <= eps for a linear model
    of U with this gradient: move a distance sqrt(2 eps) along it."""
    return theta + grad * np.sqrt(2 * eps / (grad @ grad))


def fisher_information(theta, S, A, grad_logp) -> Array:
    """F = E[grad log p(tau) grad log p(tau)^T], where grad log p(tau) is
    the sum of the policy's scores along trajectory tau (one per row)."""
    scores = grad_logp(theta, S, A).sum(axis=1)     # (m, n)
    return scores.T @ scores / len(scores)


def natural_step(theta: Array, grad: Array, F: Array, eps: float) -> Array:
    """Best step within the ellipse (1/2) step^T F step <= eps: move along
    u = F^-1 grad, scaled so the constraint holds with equality."""
    # A tiny ridge keeps the solve stable when few trajectories make F
    # nearly singular.
    u = np.linalg.solve(F + 1e-8 * np.eye(len(F)), grad)
    return theta + u * np.sqrt(2 * eps / (grad @ u))


def discounted_rewards_to_go(R: Array, gamma: float) -> Array:
    """gamma^(k-1) times the reward-to-go at step k: an estimate of the
    action value Q, weighted by the discounted visitation of step k."""
    return gamma ** np.arange(R.shape[1]) * rewards_to_go(R, gamma)


def advantage_estimates(R: Array, gamma: float) -> Array:
    """Discounted reward-to-go minus its average over the trajectories at
    the same step: a simple estimate of the advantage A = Q - U."""
    q = discounted_rewards_to_go(R, gamma)
    return q - q.mean(axis=0, keepdims=True)


def surrogate_objective(theta, theta2, S, A, weights, logp) -> float:
    """Estimate of U(theta2) - const from trajectories sampled with theta:
    importance weights pi_theta2 / pi_theta times each step's credit."""
    w = np.exp(logp(theta2, S, A) - logp(theta, S, A))
    return float((w * weights).mean())


def surrogate_constraint(theta, theta2, S, gamma, kl) -> float:
    """Average discounted KL divergence between the old and new policies
    at the visited states."""
    discount = gamma ** np.arange(S.shape[1])
    return float((discount * kl(theta, theta2, S)).mean())


def trust_region_update(theta, grad, S, A, weights, gamma, logp, grad_logp,
                        kl, eps: float, shrink: float = 0.5,
                        max_halvings: int = 30) -> Array:
    """TRPO: take the natural gradient step, then shrink it toward theta
    until the surrogate objective improves and the KL constraint holds.
    Everything reuses the same trajectories."""
    F = fisher_information(theta, S, A, grad_logp)
    candidate = natural_step(theta, grad, F, eps)
    f0 = surrogate_objective(theta, theta, S, A, weights, logp)
    for _ in range(max_halvings):
        f = surrogate_objective(theta, candidate, S, A, weights, logp)
        g = surrogate_constraint(theta, candidate, S, gamma, kl)
        if g <= eps and f > f0:
            return candidate
        candidate = theta + shrink * (candidate - theta)
    return theta                     # no acceptable step found: stay put


def clamped_objective(theta, theta2, S, A, weights, logp,
                      eps: float) -> float:
    """PPO's pessimistic surrogate: at each step, the smaller of w * credit
    and clamp(w, 1 - eps, 1 + eps) * credit, where w is the probability
    ratio pi_theta2 / pi_theta."""
    w = np.exp(logp(theta2, S, A) - logp(theta, S, A))
    return float(np.minimum(w * weights,
                            np.clip(w, 1 - eps, 1 + eps) * weights).mean())


def clamped_gradient(theta, theta2, S, A, weights, logp, grad_logp,
                     eps: float) -> Array:
    """Gradient at theta2 of PPO's clamped surrogate, from trajectories
    sampled with theta. A step's contribution is w * grad log pi_theta2 *
    credit, or zero once the probability ratio w has moved beyond 1 +- eps
    in the direction the credit favors."""
    w = np.exp(logp(theta2, S, A) - logp(theta, S, A))
    clamped = (((weights > 0) & (w > 1 + eps))
               | ((weights < 0) & (w < 1 - eps)))
    per_step = np.where(clamped, 0.0, w * weights)
    # w * grad log pi_theta2 = grad pi_theta2 / pi_theta (ADM eq. 12.23)
    return (grad_logp(theta2, S, A) * per_step[..., None]).mean(axis=(0, 1))


def clamped_surrogate_update(theta, S, A, weights, logp, grad_logp,
                             eps: float, alpha: float,
                             k_max: int) -> Array:
    """PPO: several gradient steps on the clamped surrogate, all reusing
    the same trajectories sampled with theta."""
    theta2 = np.array(theta, dtype=float)
    for _ in range(k_max):
        theta2 = theta2 + alpha * clamped_gradient(
            theta, theta2, S, A, weights, logp, grad_logp, eps)
    return theta2
