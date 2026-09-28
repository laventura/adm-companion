"""Model-based reinforcement learning (ADM Ch. 16).

The agent keeps counts of what it has seen and plans on the model they
imply. Arrays follow adm/mdp.py: T[s, a, s2] and R[s, a]. Counts:
N[s, a, s2] transitions observed and rho[s, a] the sum of rewards.

Planners share one signature, planner(model, U, s) -> number of Bellman
backups done, and update U in place, so they can be swapped freely.
"""

import numpy as np

from adm.mdp import lookahead

Array = np.ndarray


class MaximumLikelihoodModel:
    """Counts, and the maximum likelihood model they imply (eqs. 16.1-16.2):
    T(s2 | s, a) = N(s, a, s2) / N(s, a) and R(s, a) = rho(s, a) / N(s, a).
    Pairs (s, a) never tried get T = 0 and R = 0: no reward, no future."""

    def __init__(self, n_states, n_actions, gamma):
        self.N = np.zeros((n_states, n_actions, n_states))
        self.rho = np.zeros((n_states, n_actions))
        self.gamma = gamma
        self._cache = None                            # (T, R) until the next update

    def update(self, s, a, r, s2):
        self.N[s, a, s2] += 1
        self.rho[s, a] += r
        self._cache = None

    def counts(self):
        return self.N.sum(axis=2)                     # N(s, a)

    def model(self):
        """(T, R) as arrays; untried pairs are all zeros."""
        if self._cache is None:
            self._cache = self._estimate()
        return self._cache

    def _estimate(self):
        n = self.counts()
        safe = np.maximum(n, 1)                       # avoid 0 / 0
        return self.N / safe[:, :, None], self.rho / safe

    def q_values(self, U, s):
        """Q(s, a) for one state s under the model, as a vector over a."""
        T, R = self.model()
        return R[s] + self.gamma * T[s] @ U


class RmaxModel(MaximumLikelihoodModel):
    """R-MAX (eqs. 16.3-16.4): a pair tried fewer than m times is modeled as
    paying r_max forever (a self-loop), so its value is r_max / (1 - gamma)
    and the greedy policy is drawn toward it until it is known."""

    def __init__(self, n_states, n_actions, gamma, m, r_max):
        super().__init__(n_states, n_actions, gamma)
        self.m, self.r_max = m, r_max

    def _estimate(self):
        T, R = super()._estimate()
        unknown = self.counts() < self.m              # (S, A) mask
        s_idx, a_idx = np.nonzero(unknown)
        T[s_idx, a_idx, :] = 0.0
        T[s_idx, a_idx, s_idx] = 1.0                  # paired indexing: s -> s
        R[unknown] = self.r_max
        return T, R


class BayesianModel:
    """Posterior over the model (eqs. 16.5-16.6): a Dirichlet over next states
    for each (s, a), starting from `prior` pseudocounts, and, because the
    rewards here lie in [0, 1], a Beta(1 + rho, 1 + N - rho) belief about
    each mean reward (fractional counts are fine)."""

    def __init__(self, n_states, n_actions, gamma, prior=1.0):
        self.alpha = np.full((n_states, n_actions, n_states), prior)
        self.rho = np.zeros((n_states, n_actions))
        self.n = np.zeros((n_states, n_actions))
        self.gamma = gamma

    def update(self, s, a, r, s2):
        self.alpha[s, a, s2] += 1                     # Bayes: add the count
        self.rho[s, a] += r
        self.n[s, a] += 1

    def model(self):
        """Posterior means: alpha normalized (eq. 16.9) and (1 + rho)/(2 + n)."""
        return (self.alpha / self.alpha.sum(axis=2, keepdims=True),
                (1 + self.rho) / (2 + self.n))

    def sample(self, rng):
        """One MDP drawn from the posterior. A Dirichlet draw is independent
        gamma draws, normalized; Beta draws give the mean rewards."""
        g = rng.gamma(self.alpha)
        T = g / g.sum(axis=2, keepdims=True)
        R = rng.beta(1 + self.rho, 1 + self.n - self.rho)
        return T, R

    def q_values(self, U, s):
        T, R = self.model()
        return R[s] + self.gamma * T[s] @ U


def solve(T, R, gamma, U, tol=1e-6, k_max=10_000):
    """Value iteration from the current U (a warm start). Updates U in
    place; returns the number of Bellman backups (states x sweeps)."""
    for k in range(1, k_max + 1):
        U_next = lookahead(T, R, gamma, U).max(axis=1)
        change = np.max(np.abs(U_next - U))
        U[:] = U_next
        if change < tol:
            break
    return k * len(U)


def backup(model, U, s):
    """One Bellman backup of state s on the model."""
    U[s] = model.q_values(U, s).max()


def full_update(model, U, s, rng=None):
    """Re-solve the whole estimated model after every step (16.2.1)."""
    T, R = model.model()
    return solve(T, R, model.gamma, U)


def randomized_update(m):
    """Back up the state just left and m random states (16.2.2, Dyna-like)."""
    def planner(model, U, s, rng):
        backup(model, U, s)
        for s_rand in rng.integers(len(U), size=m):
            backup(model, U, s_rand)
        return m + 1
    return planner


def prioritized_update(m):
    """Prioritized sweeping (16.2.3): back up the state just left, then up
    to m states in order of priority. When U(s) changes by delta, each
    predecessor s- gets priority max over a- of T(s | s-, a-) * |delta|.
    The queue persists between steps, so unfinished work carries over."""
    pq = {}                                           # state -> priority

    def planner(model, U, s, rng):
        pq[s] = np.inf                                # the state just left first
        done = 0
        while pq and done < m + 1:
            s_now = max(pq, key=pq.get)               # highest priority
            del pq[s_now]
            before = U[s_now]
            backup(model, U, s_now)
            done += 1
            delta = abs(U[s_now] - before)
            if delta == 0:
                continue
            T, _ = model.model()
            # T[:, :, s_now]: probability of reaching s_now from each (s-, a-).
            push = (T[:, :, s_now] * delta).max(axis=1)
            for s_pred in np.nonzero(push > 1e-12)[0]:
                pq[s_pred] = max(pq.get(s_pred, 0.0), push[s_pred])
        return done
    return planner


def posterior_sampling_update(model, U, s, rng):
    """Posterior sampling (16.6): draw an MDP from the belief, solve it,
    and act greedily on its values until the next update."""
    T, R = model.sample(rng)
    return solve(T, R, model.gamma, U)


def greedy_action(q, rng):
    """argmax with ties broken at random."""
    best = np.flatnonzero(np.isclose(q, q.max(), rtol=0, atol=1e-12))
    return rng.choice(best)


def learn(env, model, planner, steps, s0, rng, epsilon=0.0, reset=None,
          behavior=None, world_rng=None, on_step=None):
    """Interact with the true MDP `env` for `steps` steps (algorithm 15.9).

    Each step: choose a (random w.p. epsilon, else greedy on the model's
    Q values from U), observe r = env.R[s, a] and s2 ~ env.T[s, a],
    update the model, then plan. With posterior sampling, U holds the
    values of the last sampled MDP, as in the book.

    reset(s2, rng) may return a new start state (episodic problems) or
    None to continue from s2. behavior(s, rng), if given, picks the
    actions instead: a fixed data-collecting policy. world_rng, if given,
    drives the environment, the behavior and the resets, while rng drives
    the agent; then planners that use randomness still see identical
    experience. on_step(t, model, U) is called after every step.
    Returns (rewards, backups) per step, and the final U.
    """
    world = rng if world_rng is None else world_rng
    n_s, n_a = env.R.shape
    U = np.zeros(n_s)
    rewards, backups = np.zeros(steps), np.zeros(steps, dtype=int)
    s = s0
    for t in range(steps):
        if behavior is not None:
            a = behavior(s, world)
        elif rng.random() < epsilon:
            a = rng.integers(n_a)
        else:
            a = greedy_action(model.q_values(U, s), rng)
        s2 = world.choice(n_s, p=env.T[s, a])
        r = env.R[s, a]
        model.update(s, a, r, s2)
        backups[t] = planner(model, U, s, rng)
        rewards[t] = r
        s = s2
        if reset is not None:
            start = reset(s, world)
            s = s if start is None else start
        if on_step is not None:
            on_step(t, model, U)
    return rewards, backups, U
