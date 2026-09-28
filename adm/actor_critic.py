"""Actor-critic methods (ADM Ch. 13): learn a policy (the actor) together
with a value function (the critic) that supplies low-noise advantages.

Trajectory arrays follow Chapters 11–12: S, A, R of shape (m, d), row i
is trajectory i and column k is step k + 1. Critics are linear in their
parameters phi: U(phi, s) = phi . features(s), so the gradient with respect
to phi is the feature vector.

Lines stay under 80 characters because the Ch 13 page prints this source.
"""

import numpy as np

from adm.online import sample_step
from adm.policy_gradient import rewards_to_go
from adm.policy_optimization import scale_gradient

Array = np.ndarray


def td_residuals(values: Array, R: Array, gamma: float) -> Array:
    """delta_k = r_k + gamma U(s_(k+1)) - U(s_k) for steps 1..d-1, given
    the critic's values U(s_k) at every visited state (shape (m, d)).
    The last step has no next state in the data, so it is dropped."""
    return R[:, :-1] + gamma * values[:, 1:] - values[:, :-1]


def generalized_advantages(delta: Array, gamma: float,
                           lam: float) -> Array:
    """A_k = sum over l >= 0 of (gamma lam)^l delta_(k+l), computed by a
    backward sweep. lam = 0 gives the TD residual; lam = 1 the discounted
    reward-to-go minus the critic's value (Monte Carlo)."""
    adv = np.zeros_like(delta)
    running = np.zeros(delta.shape[0])
    for k in range(delta.shape[1] - 1, -1, -1):
        running = delta[:, k] + gamma * lam * running
        adv[:, k] = running
    return adv


def actor_gradient(scores: Array, adv: Array, gamma: float) -> Array:
    """Policy gradient sum_k grad log pi_k gamma^(k-1) A_k, averaged over
    trajectories. scores has shape (m, d', n) and adv (m, d')."""
    discount = gamma ** np.arange(adv.shape[1])
    return (scores * (discount * adv)[..., None]).sum(axis=1).mean(axis=0)


def critic_gradient(values: Array, features: Array, R: Array,
                    gamma: float) -> Array:
    """Gradient of (1/2) sum_k (U(s_k) - rtg_k)^2, averaged over
    trajectories: pull the critic toward the observed reward-to-go.
    features has shape (m, d, p): the gradient of U with respect to phi."""
    err = values - rewards_to_go(R, gamma)          # (m, d)
    return (err[..., None] * features).sum(axis=1).mean(axis=0)


def actor_critic_step(theta, phi, S, A, R, gamma, grad_logp, features,
                      lam: float, alpha_theta: float, alpha_phi: float):
    """One update of both: ascend the actor's policy gradient (with
    generalized advantages from the critic), scaled to length at most 1
    as in Chapter 12, and descend the critic's squared error. Returns the
    new (theta, phi)."""
    feats = features(S)                             # (m, d, p)
    values = feats @ phi                            # U_phi at every state
    adv = generalized_advantages(td_residuals(values, R, gamma), gamma,
                                 lam)
    scores = grad_logp(theta, S[:, :-1], A[:, :-1])
    grad = scale_gradient(actor_gradient(scores, adv, gamma), 1.0)
    theta = theta + alpha_theta * grad
    phi = phi - alpha_phi * critic_gradient(values, feats, R, gamma)
    return theta, phi


def deterministic_policy_gradients(theta, phi, S, A, R, gamma, policy,
                                   dpolicy, q_features, dq_da):
    """Deterministic policy gradient with a critic Q_phi(s, a) =
    phi . q_features(s, a). The actor ascends
      sum_k gamma^(k-1) (d pi / d theta)(s_k) (dQ/da)(s_k, pi(s_k)),
    and the critic descends the squared TD residual
      delta = r + gamma Q(s', pi(s')) - Q(s, a).
    Returns (actor gradient, critic gradient)."""
    discount = gamma ** np.arange(S.shape[1])
    pi_s = policy(theta, S)
    actor = (discount[None, :, None] * dpolicy(theta, S)
             * dq_da(phi, S, pi_s)[..., None]).sum(axis=1).mean(axis=0)
    f_sa = q_features(S[:, :-1], A[:, :-1])         # Q at taken actions
    s2 = S[:, 1:]
    f_next = q_features(s2, policy(theta, s2))       # Q at the next state
    delta = R[:, :-1] + gamma * f_next @ phi - f_sa @ phi
    critic = (delta[..., None] * (gamma * f_next - f_sa)).sum(axis=1)
    return actor, critic.mean(axis=0)


def softmax(logits: Array) -> Array:
    z = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


class GuidedTreeSearch:
    """Monte Carlo tree search guided by a policy prior and a value
    estimate (as in AlphaGo Zero): actions are chosen by
      Q(s, a) + c pi(a | s) sqrt(N(s)) / (1 + N(s, a)),
    and new leaves are valued by the critic instead of a rollout."""

    def __init__(self, mdp, prior: Array, values: Array, d: int, c: float,
                 rng):
        self.mdp, self.prior, self.values = mdp, prior, values
        self.d, self.c, self.rng = d, c, rng
        self.N, self.Q = {}, {}

    def simulate(self, s: int, d: int) -> float:
        if d <= 0:
            return self.values[s]
        if s not in self.N:                 # new leaf: the critic values it
            n_actions = self.mdp.T.shape[1]
            self.N[s], self.Q[s] = np.zeros(n_actions), np.zeros(n_actions)
            return self.values[s]
        N, Q = self.N[s], self.Q[s]
        score = Q + self.c * self.prior[s] * np.sqrt(N.sum()) / (1 + N)
        a = int(np.argmax(score))
        s2, r = sample_step(self.mdp, s, a, self.rng)
        q = r + self.mdp.gamma * self.simulate(s2, d - 1)
        N[a] += 1
        Q[a] += (q - Q[a]) / N[a]
        return q


def root_targets(tree: GuidedTreeSearch, s: int, eta: float = 1.0):
    """Training targets from a finished search at its root s:
    pi_MCTS(a | s) proportional to N(s, a)^eta, and U_MCTS(s) = the
    largest Q(s, a) among the actions tried."""
    N = tree.N[s]
    weights = N**eta
    return weights / weights.sum(), tree.Q[s][N > 0].max()


def tree_search_actor_critic_step(mdp, logits: Array, values: Array,
                                  starts, m: int, d: int, c: float,
                                  alpha_theta: float, alpha_phi: float,
                                  rng, eta: float = 1.0):
    """One round: search from each start state, then move the policy at
    that state toward the search's visit distribution (for softmax logits
    the cross-entropy gradient is pi_theta - pi_MCTS) and its value toward
    U_MCTS. Tabular: one logit per (state, action), one value per state.
    Only the roots are trained, where the search statistics are richest
    (as in AlphaGo Zero)."""
    logits, values = logits.copy(), values.copy()
    for s0 in starts:
        tree = GuidedTreeSearch(mdp, softmax(logits), values, d, c, rng)
        for _ in range(m):
            tree.simulate(s0, d)
        pi_target, u_target = root_targets(tree, s0, eta)
        logits[s0] -= alpha_theta * (softmax(logits[s0]) - pi_target)
        values[s0] -= alpha_phi * (values[s0] - u_target)
    return logits, values
