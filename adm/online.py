"""Online planning (ADM Ch. 9): choose the action for the current state
only, by reasoning about the states reachable from it.

Planners work on a finite MDP `mdp` with arrays T[s, a, s2] = T(s2 | s, a),
R[s, a] and discount mdp.gamma (a DiscreteMDP). States and actions are
indices. Value functions passed in, such as U or U_lo, are functions of a
state index. Sampling planners take a NumPy random generator `rng`.

Lines stay under 80 characters because the Ch 9 page prints this source.
"""

import numpy as np

Array = np.ndarray


def sample_step(mdp, s: int, a: int, rng) -> tuple[int, float]:
    """Generative model: draw s2 ~ T(. | s, a) and return (s2, reward)."""
    s2 = rng.choice(len(mdp.T), p=mdp.T[s, a])
    return int(s2), float(mdp.R[s, a])


def lookahead(mdp, U, s: int, a: int) -> float:
    """R(s, a) + gamma * sum_s2 T(s2 | s, a) U(s2), visiting only the
    next states that can actually happen."""
    p = mdp.T[s, a]
    future = sum(p[s2] * U(s2) for s2 in np.flatnonzero(p))
    return float(mdp.R[s, a] + mdp.gamma * future)


def rollout(mdp, s: int, policy, d: int, rng) -> float:
    """Discounted return of one simulation of `policy` for d steps."""
    ret = 0.0
    for t in range(d):
        s, r = sample_step(mdp, s, policy(s, rng), rng)
        ret += mdp.gamma**t * r
    return ret


def rollout_lookahead(mdp, s: int, policy, d: int, m: int, rng) -> int:
    """Look one step ahead, valuing each next state by the average of m
    rollouts of `policy` (a one-step policy improvement, as in Ch 7)."""
    def U(s2):
        return np.mean([rollout(mdp, s2, policy, d, rng) for _ in range(m)])
    q = [lookahead(mdp, U, s, a) for a in range(mdp.T.shape[1])]
    return int(np.argmax(q))


def forward_search(mdp, s: int, d: int, U) -> tuple[int | None, float]:
    """Best action and its value, searching every action and every
    possible next state to depth d, with U(s) at the leaves."""
    if d <= 0:
        return None, U(s)

    def U_next(s2):                   # value of s2 with d - 1 steps left
        return forward_search(mdp, s2, d - 1, U)[1]
    q = [lookahead(mdp, U_next, s, a) for a in range(mdp.T.shape[1])]
    a = int(np.argmax(q))
    return a, q[a]


def branch_and_bound(mdp, s: int, d: int, U_lo, Q_hi) -> tuple:
    """Forward search that skips actions which can't win: U_lo (a lower
    bound on U) values the leaves, and Q_hi (an upper bound on Q) says how
    good an action could possibly be."""
    if d <= 0:
        return None, U_lo(s)

    def U_next(s2):
        return branch_and_bound(mdp, s2, d - 1, U_lo, Q_hi)[1]
    best_a, best_u = None, -np.inf
    # Try the most promising actions first, so pruning starts early.
    for a in sorted(range(mdp.T.shape[1]), key=lambda a: -Q_hi(s, a)):
        if Q_hi(s, a) < best_u:
            break          # this action and all later ones can't do better
        u = lookahead(mdp, U_next, s, a)
        if u > best_u:
            best_a, best_u = a, u
    return best_a, best_u


def sparse_sampling(mdp, s: int, d: int, m: int, U, rng) -> tuple:
    """Forward search that follows only m sampled next states per action,
    so its cost no longer depends on the number of states."""
    if d <= 0:
        return None, U(s)
    best_a, best_u = None, -np.inf
    for a in range(mdp.T.shape[1]):
        u = 0.0
        for _ in range(m):
            s2, r = sample_step(mdp, s, a, rng)
            u += (r + mdp.gamma * sparse_sampling(mdp, s2, d - 1, m, U,
                                                  rng)[1]) / m
        if u > best_u:
            best_a, best_u = a, u
    return best_a, best_u


class MonteCarloTreeSearch:
    """Grow a search tree with m simulations from the current state, using
    UCB1 to balance trying promising actions against untried ones."""

    def __init__(self, mdp, d: int, m: int, c: float, U, rng):
        self.mdp, self.d, self.m, self.c, self.U = mdp, d, m, c, U
        self.rng = rng
        self.N = {}    # N[s][a]: how often action a was tried in state s
        self.Q = {}    # Q[s][a]: average return seen after doing so

    def __call__(self, s: int) -> int:
        for _ in range(self.m):
            self.simulate(s, self.d)
        return int(np.argmax(self.Q[s]))

    def simulate(self, s: int, d: int) -> float:
        if d <= 0:
            return self.U(s)
        n_actions = self.mdp.T.shape[1]
        if s not in self.N:           # a new leaf: add it, estimate its value
            self.N[s], self.Q[s] = np.zeros(n_actions), np.zeros(n_actions)
            return self.U(s)
        a = self.explore(s)
        s2, r = sample_step(self.mdp, s, a, self.rng)
        q = r + self.mdp.gamma * self.simulate(s2, d - 1)
        self.N[s][a] += 1
        self.Q[s][a] += (q - self.Q[s][a]) / self.N[s][a]  # running mean
        return q

    def explore(self, s: int) -> int:
        """UCB1: Q(s, a) + c sqrt(log N(s) / N(s, a)); untried actions
        get an infinite bonus, so each is tried once first."""
        N, Q = self.N[s], self.Q[s]
        bonus = np.full(len(N), np.inf)
        tried = N > 0
        if tried.any():
            bonus[tried] = np.sqrt(np.log(N.sum()) / N[tried])
        return int(np.argmax(Q + self.c * bonus))


def greedy(mdp, U: Array, s: int) -> tuple[int, float]:
    """Best action at s by one-step lookahead on a value table U."""
    q = mdp.R[s] + mdp.gamma * mdp.T[s] @ U     # one entry per action
    a = int(np.argmax(q))
    return a, float(q[a])


def heuristic_search(mdp, s: int, U_hi: Array, d: int, m: int, rng):
    """Run m greedy simulations of depth d from s, updating U by one-step
    lookahead along the way (real-time dynamic programming). Starting from
    an upper bound U_hi makes unexplored states look attractive."""
    U = np.array(U_hi, dtype=float)
    for _ in range(m):
        state = s
        for _ in range(d):
            a, U[state] = greedy(mdp, U, state)
            state, _ = sample_step(mdp, state, a, rng)
    return greedy(mdp, U, s)[0], U


def labeled_heuristic_search(mdp, s: int, U_hi: Array, d: int,
                             delta: float, rng, max_sims: int = 10_000):
    """Heuristic search that stops once s is 'solved': every state the
    greedy policy can reach from s has a residual below delta."""
    U, solved, sims = np.array(U_hi, dtype=float), set(), 0
    while s not in solved and sims < max_sims:
        visited, state = [], s
        for _ in range(d):           # a greedy simulation, as before
            if state in solved:
                break
            visited.append(state)
            a, U[state] = greedy(mdp, U, state)
            state, _ = sample_step(mdp, state, a, rng)
        # Try to label the visited states, last first; stop at the first
        # one whose greedy envelope still has a large residual.
        while visited and _label(mdp, U, solved, visited.pop(), delta):
            pass
        sims += 1
    return greedy(mdp, U, s)[0], U, sims


def _label(mdp, U: Array, solved: set, s: int, delta: float) -> bool:
    """Mark s and its greedy envelope solved if all residuals are below
    delta; otherwise update the envelope's values. True if solved."""
    if s in solved:
        return True
    envelope, to_expand, converged = [], [s], True
    while to_expand:                 # states reachable under greedy actions
        state = to_expand.pop()
        envelope.append(state)
        a, u = greedy(mdp, U, state)
        if abs(U[state] - u) > delta:
            converged = False
        else:                        # only expand states that look settled
            for s2 in np.flatnonzero(mdp.T[state, a]):
                if s2 not in solved and s2 not in envelope + to_expand:
                    to_expand.append(int(s2))
    if converged:
        solved.update(envelope)
    else:
        for state in reversed(envelope):
            U[state] = greedy(mdp, U, state)[1]
    return converged


def deterministic_forward_search(step, is_terminal, actions, U, s,
                                 d: int, reward: float = -1.0):
    """Forward search for a deterministic problem, vectorized: try every
    sequence of d actions at once. With no chance, the best first action of
    the best sequence is the forward-search answer (gamma = 1)."""
    n = len(actions)
    seqs = np.array(np.meshgrid(*[actions] * d, indexing="ij"))
    seqs = seqs.reshape(d, -1).T          # one row per action sequence
    states = np.repeat(np.asarray(s, float)[None], len(seqs), axis=0)
    ret, done = np.zeros(len(seqs)), np.zeros(len(seqs), bool)
    for t in range(d):
        ret[~done] += reward              # pay until the goal is reached
        states[~done] = step(states[~done], seqs[~done, t])
        done |= is_terminal(states)
    ret[~done] += U(states[~done])        # value estimate at the leaves
    best = int(np.argmax(ret))
    return seqs[best, 0], ret[best], n**d


def double_integrator(s0: Array, a: Array) -> tuple[Array, Array]:
    """Positions and velocities after each action for a point that
    accelerates in 2-D: s0 = [x, y, vx, vy], a[t] = [ax, ay].
    Velocity changes first, then position moves by the new velocity."""
    v = s0[2:] + np.cumsum(a, axis=0)
    p = s0[:2] + np.cumsum(v, axis=0)
    return p, v


def mpc_plan(s0: Array, goal: Array, centers: Array, radius: float,
             a_max: float, weight: float = 50.0, warm: Array | None = None):
    """Open-loop plan for the double integrator: minimize effort
    sum ||a_t||^2 plus weight * (distance from the goal state at the end)^2,
    keeping each planned position outside a circle. centers[t] is where
    the planner expects the obstacle's center at step t + 1, so the plan has
    len(centers) steps. Returns the (d, 2) actions."""
    from scipy.optimize import minimize
    d = len(centers)

    def cost(x):
        a = x.reshape(d, 2)
        p, v = double_integrator(s0, a)
        miss = np.sum((p[-1] - goal[:2])**2) + np.sum((v[-1] - goal[2:])**2)
        return np.sum(a**2) + weight * miss

    def clearance(x):                # must be >= 0 at every step
        p, _ = double_integrator(s0, x.reshape(d, 2))
        return np.sum((p - centers)**2, axis=1) - radius**2
    x0 = np.zeros(2 * d) if warm is None else warm.ravel()
    result = minimize(cost, x0, method="SLSQP",
                      bounds=[(-a_max, a_max)] * (2 * d),
                      constraints=[{"type": "ineq", "fun": clearance}],
                      options={"maxiter": 300})
    return result.x.reshape(d, 2)
