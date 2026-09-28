"""Model-free reinforcement learning (ADM Ch. 17).

These methods learn action values Q(s, a) straight from experience
tuples (s, a, r, s2), with no transition or reward model. Tabular Q is an
(S, A) array; approximate Q is theta @ features(s), one weight row per
action.
"""

from collections import deque

import numpy as np

from adm.policy_optimization import scale_gradient

Array = np.ndarray


def incremental_mean(xs, alpha):
    """Running estimates x_hat_m = x_hat_{m-1} + alpha(m) (x_m - x_hat_{m-1})
    (eq. 17.5), starting from 0. alpha(m) = 1/m gives the sample mean."""
    est, out = 0.0, []
    for m, x in enumerate(xs, start=1):
        est += alpha(m) * (x - est)
        out.append(est)
    return np.array(out)


def epsilon_greedy(q, epsilon, rng):
    """A random action w.p. epsilon, else a greedy one (ties at random)."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    return int(rng.choice(np.flatnonzero(q == q.max())))


class QLearning:
    """Q(s, a) <- Q(s, a) + alpha (r + gamma max_a' Q(s', a') - Q(s, a))
    (eq. 17.10): off-policy, it learns the greedy policy's values while
    following an exploring one."""

    def __init__(self, n_states, n_actions, gamma, alpha, q0=0.0):
        self.Q = np.full((n_states, n_actions), float(q0))
        self.gamma, self.alpha = gamma, alpha

    def start(self):
        """Called at the start of each episode (nothing to reset here)."""

    def update(self, s, a, r, s2, a2):
        target = r + self.gamma * self.Q[s2].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


class Sarsa(QLearning):
    """Q(s, a) <- Q(s, a) + alpha (r + gamma Q(s', a') - Q(s, a)) (eq. 17.11),
    with a' the action actually taken next: on-policy, it learns the
    values of the exploring policy itself."""

    def update(self, s, a, r, s2, a2):
        target = r + self.gamma * self.Q[s2, a2]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


class SarsaLambda(QLearning):
    """Sarsa with eligibility traces (eqs. 17.12-17.14). N[s, a] is a
    decaying visit count; each TD error delta updates every pair in
    proportion to it, so reward flows back along the recent path."""

    def __init__(self, n_states, n_actions, gamma, alpha, lam, q0=0.0):
        super().__init__(n_states, n_actions, gamma, alpha, q0)
        self.lam = lam
        self.N = np.zeros_like(self.Q)

    def start(self):
        self.N[:] = 0.0                               # traces don't cross episodes

    def update(self, s, a, r, s2, a2):
        self.N[s, a] += 1
        delta = r + self.gamma * self.Q[s2, a2] - self.Q[s, a]
        self.Q += self.alpha * delta * self.N         # every pair, by its trace
        self.N *= self.gamma * self.lam


def run_tabular(env, learner, steps, s0, rng, epsilon=0.1, reset=None,
                shaping=None, on_step=None):
    """Learn on a DiscreteMDP env for `steps` steps with epsilon-greedy
    actions on learner.Q.

    reset(s2, rng) returns a new start state when an episode ends (and
    None otherwise). shaping(s, s2) is added to the reward the learner
    sees (not to the reward recorded). on_step(t, learner) is called after
    every step. Returns the rewards received.
    """
    n_s = env.R.shape[0]
    rewards = np.zeros(steps)
    s = s0
    learner.start()
    a = epsilon_greedy(learner.Q[s], epsilon, rng)
    for t in range(steps):
        s2 = rng.choice(n_s, p=env.T[s, a])
        r = env.R[s, a]
        a2 = epsilon_greedy(learner.Q[s2], epsilon, rng)   # Sarsa needs a'
        bonus = 0.0 if shaping is None else shaping(s, s2)
        learner.update(s, a, r + bonus, s2, a2)
        rewards[t] = r
        s, a = s2, a2
        start = None if reset is None else reset(s, rng)
        if start is not None:                              # new episode
            s = start
            learner.start()
            a = epsilon_greedy(learner.Q[s], epsilon, rng)
        if on_step is not None:
            on_step(t, learner)
    return rewards


def potential_shaping(beta, gamma):
    """F(s, s') = gamma beta(s') - beta(s) (eq. 17.15): shaping that
    provably leaves the optimal policy unchanged."""
    return lambda s, s2: gamma * beta[s2] - beta[s]


class GradientQLearning:
    """Q-learning with a linear action value function
    Q_theta(s, a) = theta[a] @ features(s) (eq. 17.21), with the step's
    gradient scaled to norm at most 1.

    With replay > 0, experience tuples go into a buffer of that size and,
    every `every` steps, one step is taken along the average gradient of
    a random batch of `batch` tuples (eq. 17.22) instead.
    """

    def __init__(self, features, n_features, n_actions, gamma, alpha,
                 replay=0, batch=32, every=1):
        self.features, self.gamma, self.alpha = features, gamma, alpha
        self.theta = np.zeros((n_actions, n_features))
        self.buffer = deque(maxlen=replay) if replay else None
        self.batch, self.every, self.t = batch, every, 0

    def q(self, f):
        """Q values of all actions, for one feature vector f."""
        return self.theta @ f

    def update(self, f, a, r, f2, done, rng):
        self.t += 1
        if self.buffer is None:
            # One tuple: TD error times grad_theta Q = the features f.
            target = r + (0.0 if done else self.gamma * self.q(f2).max())
            g = np.zeros_like(self.theta)
            g[a] = (target - self.q(f)[a]) * f
        else:
            self.buffer.append((f, a, r, f2, done))
            if len(self.buffer) < self.batch or self.t % self.every:
                return
            pick = rng.integers(len(self.buffer), size=self.batch)
            F, A, R, F2, D = (np.array(x) for x in
                              zip(*(self.buffer[i] for i in pick)))
            # Batched targets: (batch, n_actions) values of the next states.
            targets = R + np.where(D, 0.0,
                                   self.gamma * (F2 @ self.theta.T).max(1))
            td = targets - (F * self.theta[A]).sum(axis=1)
            g = np.zeros_like(self.theta)
            np.add.at(g, A, td[:, None] * F)          # sum rows per action
            g /= self.batch
        self.theta += self.alpha * scale_gradient(g, 1.0)


def run_episode(learner, env_step, s0, rng, epsilon=0.0, max_steps=1000):
    """One episode of gradient Q-learning from state s0.

    env_step(s, a) returns (s2, r, done) for action index a. Actions are
    epsilon-greedy on the current Q. Returns the number of steps taken.
    """
    s, f = s0, learner.features(s0)
    for t in range(1, max_steps + 1):
        a = epsilon_greedy(learner.q(f), epsilon, rng)
        s2, r, done = env_step(s, a)
        f2 = learner.features(s2)
        learner.update(f, a, r, f2, done, rng)
        if done:
            return t
        s, f = s2, f2
    return max_steps
