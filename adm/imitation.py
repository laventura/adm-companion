"""Imitation learning (ADM Ch. 18) for finite MDPs.

Policies are arrays probs[s, a] of action probabilities. Features are an
array beta[s, a, k], and a linear reward is R_phi = beta @ phi. Everything
is evaluated exactly with linear solves, so methods can be compared
without simulation noise.
"""

import numpy as np
from scipy.optimize import minimize

from adm.mdp import policy_iteration

Array = np.ndarray


# --- Evaluation --------------------------------------------------------------------------------

def policy_value(T, R, gamma, probs, b):
    """Expected discounted return of a stochastic policy from start
    distribution b: b . (I - gamma T_pi)^-1 R_pi."""
    T_pi = np.einsum("sa,sat->st", probs, T)         # average T over actions
    R_pi = (probs * R).sum(axis=1)
    return b @ np.linalg.solve(np.eye(len(b)) - gamma * T_pi, R_pi)


def feature_expectations(T, beta, gamma, probs, b):
    """mu_pi = E[sum_k gamma^(k-1) beta(s_k, a_k)] from b (eq. 18.5), exactly:
    the value of each feature treated as a reward."""
    T_pi = np.einsum("sa,sat->st", probs, T)
    beta_pi = np.einsum("sa,sak->sk", probs, beta)   # expected features per state
    return b @ np.linalg.solve(np.eye(len(b)) - gamma * T_pi, beta_pi)


def greedy_probs(policy, n_actions):
    """A deterministic policy (array of actions) as probabilities."""
    return np.eye(n_actions)[policy]


# --- Demonstrations and behavioral cloning ----------------------------------------------------

def rollouts(T, probs, b, m, d, rng, label=None, end=None):
    """m trajectories of up to d steps from b, acting with `probs`. Each
    trajectory is a list of (s, a). If label(s) is given, the recorded
    action is label(s) (the expert's choice) while the action taken still
    comes from probs, as in DAgger. Rollouts stop at the state `end`."""
    trajs = []
    n_s, n_a = probs.shape
    for _ in range(m):
        s, traj = rng.choice(n_s, p=b), []
        for _ in range(d):
            if s == end:
                break
            a = rng.choice(n_a, p=probs[s])
            traj.append((s, a if label is None else label(s)))
            s = rng.choice(n_s, p=T[s, a])
        trajs.append(traj)
    return trajs


def behavioral_cloning(pairs, n_states, n_actions):
    """Maximum likelihood tabular policy (eq. 18.1): action counts per
    state, normalized. States never demonstrated get a uniform policy."""
    N = np.zeros((n_states, n_actions))
    for s, a in pairs:
        N[s, a] += 1
    seen = N.sum(axis=1, keepdims=True)
    return np.where(seen > 0, N / np.maximum(seen, 1), 1 / n_actions)


def dagger(T, expert, b, demos, k_max, m, d, rng, end=None):
    """Data set aggregation (algorithm 18.2): clone, run the clone, have
    the expert label the states it visits, add them, clone again.
    expert[s] is the expert's action. Returns the policies after each
    iteration (the first is plain behavioral cloning)."""
    n_s, n_a = T.shape[0], T.shape[1]
    data = [pair for traj in demos for pair in traj]
    policies = [behavioral_cloning(data, n_s, n_a)]
    for _ in range(k_max - 1):
        new = rollouts(T, policies[-1], b, m, d, rng, label=lambda s: expert[s], end=end)
        data += [pair for traj in new for pair in traj]
        policies.append(behavioral_cloning(data, n_s, n_a))
    return policies


def smile(T, expert, b, k_max, m, d, beta, rng, end=None):
    """Stochastic mixing iterative learning (algorithm 18.3). Iteration k
    runs the current mixture, has the expert label the visited states,
    clones only that new data, and mixes the clone in. The expert keeps
    weight (1 - beta)^k and clone i gets beta (1 - beta)^(i-1). Returns the
    mixtures after each iteration and the final one with the expert's share
    renormalized away."""
    n_s, n_a = T.shape[0], T.shape[1]
    expert_probs = greedy_probs(expert, n_a)
    clones, mixtures = [], []
    current = expert_probs
    for k in range(1, k_max + 1):
        new = rollouts(T, current, b, m, d, rng, label=lambda s: expert[s], end=end)
        clones.append(behavioral_cloning([p for tr in new for p in tr], n_s, n_a))
        w = np.array([beta * (1 - beta) ** (i - 1) for i in range(1, k + 1)])
        current = (1 - beta) ** k * expert_probs + np.tensordot(w, np.array(clones), axes=1)
        mixtures.append(current)
    final = np.tensordot(w / w.sum(), np.array(clones), axes=1)
    return mixtures, final


# --- Inverse reinforcement learning -------------------------------------------------------------

def demo_feature_expectations(demos, beta, gamma):
    """mu_E from trajectories: average of sum_k gamma^(k-1) beta(s_k, a_k)."""
    return np.mean([sum(gamma**k * beta[s, a] for k, (s, a) in enumerate(traj))
                    for traj in demos], axis=0)


def max_margin_weights(mu_E, mus):
    """Solve eq. 18.6: maximize t subject to phi.mu_E >= phi.mu_i + t for
    every earlier policy i and ||phi|| <= 1. Returns (t, phi)."""
    k = len(mu_E)
    cons = [{"type": "ineq", "fun": lambda x, mu=mu: x[:k] @ (mu_E - mu) - x[k]} for mu in mus]
    cons.append({"type": "ineq", "fun": lambda x: 1 - x[:k] @ x[:k]})
    x0 = np.r_[(mu_E - np.mean(mus, axis=0)) / (np.linalg.norm(mu_E - np.mean(mus, axis=0)) + 1e-12), 0.0]
    res = minimize(lambda x: -x[k], x0, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 500})
    return res.x[k], res.x[:k]


def mixture_weights(mu_E, mus):
    """Solve eq. 18.7: the convex combination of the mus closest to mu_E."""
    M = np.array(mus)
    k = len(M)
    res = minimize(lambda lam: np.sum((mu_E - lam @ M) ** 2), np.full(k, 1 / k),
                   bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda lam: lam.sum() - 1}],
                   method="SLSQP", options={"ftol": 1e-14, "maxiter": 500})
    return res.x


def max_margin_irl(T, beta, gamma, mu_E, b, eps=1e-3, k_max=30):
    """Maximum margin inverse reinforcement learning (algorithm 18.5), with
    exact planning and exact feature expectations. Starts from a uniformly
    random policy. Returns (lam, policies, mus, margins, phis)."""
    n_s, n_a = T.shape[0], T.shape[1]
    policies = [np.full((n_s, n_a), 1 / n_a)]
    mus = [feature_expectations(T, beta, gamma, policies[0], b)]
    margins, phis = [], []
    for _ in range(k_max):
        t, phi = max_margin_weights(mu_E, mus)
        margins.append(t)
        phis.append(phi)
        if t <= eps:
            break
        pi, _, _ = policy_iteration(T, beta @ phi, gamma, np.zeros(n_s, int))  # plan for R_phi
        policies.append(greedy_probs(pi, n_a))
        mus.append(feature_expectations(T, beta, gamma, policies[-1], b))
    return mixture_weights(mu_E, mus), policies, mus, margins, phis


def soft_policy(T, R, gamma, iters=2000, tol=1e-10):
    """Maximum entropy ("soft") optimal policy: Q = R + gamma T V with
    V(s) = log sum_a exp Q(s, a), and pi(a | s) = exp(Q(s, a) - V(s))."""
    V = np.zeros(T.shape[0])
    for _ in range(iters):
        Q = R + gamma * T @ V
        Qmax = Q.max(axis=1, keepdims=True)
        V_new = (Qmax + np.log(np.exp(Q - Qmax).sum(axis=1, keepdims=True)))[:, 0]
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = R + gamma * T @ V
    return np.exp(Q - V[:, None])


def horizon_feature_expectations(T, beta, gamma, probs, b, d):
    """Expected sum_{k=1}^{d} gamma^(k-1) beta(s_k, a_k) by pushing the
    state distribution forward d steps (the dynamic program of eq. 18.18)."""
    dist, mu = b.copy(), np.zeros(beta.shape[2])
    T_pi = np.einsum("sa,sat->st", probs, T)
    beta_pi = np.einsum("sa,sak->sk", probs, beta)
    for k in range(d):
        mu += gamma**k * dist @ beta_pi
        dist = dist @ T_pi
    return mu


def max_entropy_irl(T, beta, gamma, demos, b, d, alpha, k_max, phi0=None):
    """Maximum entropy inverse reinforcement learning (algorithm 18.6), per
    demonstration: ascend the log-likelihood, whose gradient is the
    demonstrations' discounted feature counts minus those expected under
    the maximum entropy policy for the current reward (eq. 18.17).
    Returns (phi, policy, history of gradient norms)."""
    mu_D = demo_feature_expectations(demos, beta, gamma)
    phi = np.zeros(beta.shape[2]) if phi0 is None else phi0.copy()
    norms = []
    for _ in range(k_max):
        probs = soft_policy(T, beta @ phi, gamma)
        grad = mu_D - horizon_feature_expectations(T, beta, gamma, probs, b, d)
        norms.append(np.linalg.norm(grad))
        phi += alpha * grad
    return phi, soft_policy(T, beta @ phi, gamma), norms
