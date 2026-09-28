"""Markov games (ADM Ch. 25): evaluation, responses, equilibria, learning.

A two-agent game stores T[s, a1, a2, s2] and R[i, s, a1, a2]
(adm.problems.markov_games). A policy for agent i is a matrix
pi[s, a]: the probability of action a in state s.
"""

import numpy as np
from scipy.optimize import minimize

from adm import games
from adm.mdp import policy_iteration
from adm.problems.simple_games import SimpleGame

Array = np.ndarray


def joint_matrices(game, policies):
    """eqs. 25.1-25.2: the rewards R^i(s, pi(s)) and transitions
    T(s2 | s, pi(s)) when both agents follow their policies."""
    p1, p2 = policies
    # w[s, a1, a2]: probability of the joint action in state s.
    w = p1[:, :, None] * p2[:, None, :]
    R = np.einsum("sab,isab->is", w, game.R)
    T = np.einsum("sab,sabt->st", w, game.T)
    return R, T


def policy_evaluation(game, policies) -> Array:
    """eq. 25.3 for both agents by one linear solve: U[i, s]."""
    R, T = joint_matrices(game, policies)
    n = len(T)
    return np.linalg.solve(np.eye(n) - game.gamma * T, R.T).T


def response_mdp(game, policies, i):
    """eqs. 25.4-25.5: with the other agent's policy fixed, agent i faces an
    MDP with T'[s, ai, s2] and R'[s, ai]."""
    other = policies[1 - i]
    if i == 0:          # sum out agent 2's action a2
        T = np.einsum("sb,sabt->sat", other, game.T)
        R = np.einsum("sb,sab->sa", other, game.R[0])
    else:               # sum out agent 1's action a1
        T = np.einsum("sa,sabt->sbt", other, game.T)
        R = np.einsum("sa,sab->sb", other, game.R[1])
    return T, R


def best_response(game, policies, i):
    """Agent i's optimal policy against the other's fixed policy, by exact
    policy iteration on the response MDP. Returns (policy matrix, values)."""
    T, R = response_mdp(game, policies, i)
    pi, U = policy_iteration(T, R, game.gamma, np.zeros(len(R), int))[:2]
    return np.eye(R.shape[1])[pi], U


def softmax_response(game, policies, i, lam) -> Array:
    """eq. 25.6: pi^i(a | s) proportional to exp(lam Q(s, a)), with Q from
    the response MDP's optimal values."""
    T, R = response_mdp(game, policies, i)
    _, U = best_response(game, policies, i)
    Q = lam * (R + game.gamma * T @ U)
    P = np.exp(Q - Q.max(axis=1, keepdims=True))
    return P / P.sum(axis=1, keepdims=True)


def exploitability(game, policies) -> float:
    """The most either agent could gain, in any state, by switching to its
    best response: 0 exactly at a (stationary) Nash equilibrium."""
    U = policy_evaluation(game, policies)
    return max((best_response(game, policies, i)[1] - U[i]).max()
               for i in range(2))


def nash_equilibrium(game, rng, starts=10):
    """eq. 25.7 by SLSQP: minimize sum_{i,s} U_i(s) - Q_i(s, pi(s)) subject
    to U_i(s) >= Q_i(s, a_i, pi^-i(s)) for every agent, state and action.
    Returns the best run's policies and its exploitability."""
    nS, n1, n2 = game.T.shape[:3]
    sizes = (nS * n1, nS * n2, 2 * nS)

    def unpack(z):
        p1 = z[:sizes[0]].reshape(nS, n1)
        p2 = z[sizes[0]:sizes[0] + sizes[1]].reshape(nS, n2)
        return p1, p2, z[-sizes[2]:].reshape(2, nS)

    def action_q(p1, p2, U):
        """Q^i(s, a_i, pi^-i(s)) for each agent, using the value guess U."""
        # q[i, s, a1, a2] = R^i(s, a1, a2)
        #                   + gamma sum_s2 T(s2 | s, a1, a2) U_i(s2)
        q = game.R + game.gamma * np.einsum("sabt,it->isab", game.T, U)
        q1 = np.einsum("sab,sb->sa", q[0], p2)   # average out agent 2
        q2 = np.einsum("sab,sa->sb", q[1], p1)   # average out agent 1
        return q1, q2

    def objective(z):
        p1, p2, U = unpack(z)
        q1, q2 = action_q(p1, p2, U)
        return (U[0] - (q1 * p1).sum(1)).sum() + (U[1] - (q2 * p2).sum(1)).sum()

    def gaps(z):                          # >= 0: no action beats U_i(s)
        p1, p2, U = unpack(z)
        q1, q2 = action_q(p1, p2, U)
        return np.concatenate([(U[0][:, None] - q1).ravel(),
                               (U[1][:, None] - q2).ravel()])

    def sums(z):                          # every policy row sums to 1
        p1, p2, _ = unpack(z)
        return np.concatenate([p1.sum(1) - 1, p2.sum(1) - 1])

    cons = [{"type": "ineq", "fun": gaps}, {"type": "eq", "fun": sums}]
    bounds = [(0, 1)] * (sizes[0] + sizes[1]) + [(None, None)] * sizes[2]
    best = None
    for _ in range(starts):
        p1 = rng.dirichlet(np.ones(n1), nS)
        p2 = rng.dirichlet(np.ones(n2), nS)
        U0 = policy_evaluation(game, [p1, p2])
        z0 = np.concatenate([p1.ravel(), p2.ravel(), U0.ravel()])
        res = minimize(objective, z0, method="SLSQP", bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-12})
        p1, p2, _ = unpack(res.x)
        pols = [np.clip(p, 0, None) for p in (p1, p2)]
        pols = [p / p.sum(1, keepdims=True) for p in pols]
        e = exploitability(game, pols)
        if best is None or e < best[1]:
            best = (pols, e)
    return best


def iterated_best_response(game, k_max, policies=None) -> list:
    """Both agents best-respond at once to each other's previous policies."""
    if policies is None:
        policies = [np.full((game.T.shape[0], n), 1 / n)
                    for n in game.T.shape[1:3]]
    history = [policies]
    for _ in range(k_max):
        policies = [best_response(game, policies, i)[0] for i in range(2)]
        history.append(policies)
    return history


def step(game, s, a, rng) -> int:
    """Sample the next state after joint action a in state s."""
    return rng.choice(game.T.shape[-1], p=game.T[(s,) + tuple(a)])


def fictitious_play(game, steps, rng, s0=0, reset=None, counts=1.0):
    """§25.4: each agent keeps counts N(j, a_j, s) of both agents' actions
    (eq. 25.9) and joint-action values Q_i(s, a) from the known model. It
    acts greedily on Q_i against the others' frequencies in s (eq. 25.11),
    then refreshes Q_i(s, .) at the state just visited. `reset` restarts
    every so many steps at s0. Returns the visited states, joint actions and
    each agent's frequency model after every step."""
    nS, n1, n2 = game.T.shape[:3]
    N = [np.full((nS, n1), counts), np.full((nS, n2), counts)]
    Q = [game.R[0].copy(), game.R[1].copy()]   # Q[i][s, a1, a2]
    s, trace = s0, []
    for t in range(steps):
        if reset and t % reset == 0:
            s = s0
        freq = [N[0][s] / N[0][s].sum(), N[1][s] / N[1][s].sum()]
        a1 = int(np.argmax(Q[0][s] @ freq[1]))       # sum over agent 2's action
        a2 = int(np.argmax(freq[0] @ Q[1][s]))       # sum over agent 1's action
        N[0][s, a1] += 1
        N[1][s, a2] += 1
        # Refresh Q_i(s, .): reward plus the discounted value of the next
        # state, where both agents play their frequencies there.
        pol = [Ni / Ni.sum(1, keepdims=True) for Ni in N]
        w = pol[0][:, :, None] * pol[1][:, None, :]
        for i in range(2):
            U = (Q[i] * w).sum(axis=(1, 2))           # U_i(s2) for every s2
            Q[i][s] = game.R[i, s] + game.gamma * game.T[s] @ U
        trace.append((s, a1, a2))
        s = step(game, s, (a1, a2), rng)
    return {"trace": trace, "N": N}


def gradient_ascent_play(game, steps, rng, s0=0):
    """§25.5: each agent learns Q_i(s, a1, a2) by Q-learning, treating the
    other's last action as its policy (eq. 25.15), and moves its policy at
    the visited state along Q_i(s, ., a^-i) (eqs. 25.13-25.14), projected
    onto the simplex. Step sizes and exploration are 1 / sqrt(t).
    Returns the final policies."""
    nS, n1, n2 = game.T.shape[:3]
    Q = np.zeros((2, nS, n1, n2))
    pol = [np.full((nS, n1), 1 / n1), np.full((nS, n2), 1 / n2)]
    s = s0
    for t in range(1, steps + 1):
        rate = 1 / np.sqrt(t)
        # Act: epsilon-greedy mixing with the uniform policy, epsilon = rate.
        a = [rng.choice(n, p=rate / n + (1 - rate) * p[s])
             for n, p in zip((n1, n2), pol)]
        s2 = step(game, s, a, rng)
        for i in range(2):
            # The other agent's action is held at what it just did.
            q_next = Q[i, s2, :, a[1]] if i == 0 else Q[i, s2, a[0], :]
            idx = (i, s) + tuple(a)
            Q[idx] += rate * (game.R[idx] + game.gamma * q_next.max() - Q[idx])
            grad = Q[i, s, :, a[1]] if i == 0 else Q[i, s, a[0], :]
            pol[i][s] = games.project_simplex(pol[i][s] + rate * grad)
        s = s2
    return pol


def nash_q_learning(game, steps, rng, s0=0, eps=None):
    """§25.6 (J. Hu and M. P. Wellman, 2003): learn joint-action values
    Q[i, s, a1, a2] for both agents. After each transition, solve the stage
    game Q[:, s2] for a Nash equilibrium (support enumeration; the first
    one found) and move Q toward reward + gamma times its value, with
    rate 1 / sqrt(N(s, a)). Acts on the stage equilibrium at the current
    state, exploring (a uniform action) with probability eps, or by default
    1 / sum_a N(s, a) as in the book."""
    nS, n1, n2 = game.T.shape[:3]
    Q = np.zeros((2, nS, n1, n2))
    N = np.ones((nS, n1, n2))
    s = s0

    def stage(s):
        g = SimpleGame(Q[:, s], (tuple(range(n1)), tuple(range(n2))))
        eqs = games.support_enumeration(g)
        if not eqs:                         # degenerate stage game: fall back
            return games.nash_equilibrium(g, rng, starts=3)[0]
        return list(eqs[0])

    for _ in range(steps):
        pols = stage(s)
        e = 1 / N[s].sum() if eps is None else eps
        a = [rng.integers(n) if rng.random() < e else rng.choice(n, p=p)
             for n, p in zip((n1, n2), pols)]
        s2 = step(game, s, a, rng)
        nxt = stage(s2)
        N[(s,) + tuple(a)] += 1
        alpha = 1 / np.sqrt(N[(s,) + tuple(a)])
        for i in range(2):
            u2 = nxt[0] @ Q[i, s2] @ nxt[1]           # stage value at s2
            target = game.R[(i, s) + tuple(a)] + game.gamma * u2
            Q[(i, s) + tuple(a)] += alpha * (target - Q[(i, s) + tuple(a)])
        s = s2
    policies = [np.zeros((nS, n1)), np.zeros((nS, n2))]
    for s in range(nS):
        p1, p2 = stage(s)
        policies[0][s], policies[1][s] = p1, p2
    return Q, policies, N                    # N - 1 = visits of (s, a1, a2)
