"""Simple games (ADM Ch. 24): responses, equilibria, learning by play.

A game stores R[i, a1, ..., ak] (adm.problems.simple_games). A joint
policy is a list of probability vectors: policies[i][a] = pi^i(a).
"""

from itertools import combinations

import numpy as np
from scipy.optimize import linprog, minimize

Array = np.ndarray


def action_values(game, policies, i) -> Array:
    """U^i(a_i, pi^-i) for every action a_i of agent i.

    R[i] has one axis per agent. Contract every other agent's axis with
    its policy, from the last axis down, so the axis numbers stay valid.
    """
    Ri = game.R[i]
    for j in reversed(range(len(policies))):
        if j != i:
            Ri = np.tensordot(Ri, policies[j], axes=([j], [0]))
    return Ri


def utility(game, policies, i) -> float:
    """eq. 24.1: agent i's expected reward under independent policies."""
    return float(action_values(game, policies, i) @ policies[i])


def best_response(game, policies, i) -> Array:
    """eq. 24.3: a deterministic best response (ties: lowest index)."""
    n = game.R.shape[i + 1]
    return np.eye(n)[np.argmax(action_values(game, policies, i))]


def softmax_response(game, policies, i, lam) -> Array:
    """eq. 24.4: pi^i(a) proportional to exp(lam U^i(a, pi^-i))."""
    u = lam * action_values(game, policies, i)
    p = np.exp(u - u.max())           # subtract the max: exp can't overflow
    return p / p.sum()


def regret(game, policies) -> float:
    """The most any one agent could gain by switching alone: 0 exactly at
    a Nash equilibrium."""
    return max(action_values(game, policies, i).max()
               - utility(game, policies, i) for i in range(len(policies)))


def dominant_actions(game, i) -> list:
    """Actions of agent i that are a best response to every pure joint
    action of the others (so to every mixed one too)."""
    # Rows: agent i's actions; columns: every joint action of the others.
    Ri = np.moveaxis(game.R[i], i, 0).reshape(game.R.shape[i + 1], -1)
    return [a for a in range(len(Ri)) if np.all(Ri[a] >= Ri.max(axis=0))]


def dominant_strategy_equilibrium(game):
    """A joint action of dominant actions, or None if some agent has none."""
    doms = [dominant_actions(game, i) for i in range(game.R.shape[0])]
    return tuple(d[0] for d in doms) if all(doms) else None


def nash_equilibrium(game, rng, starts=10):
    """eq. 24.5 by SLSQP from random starts: minimize sum_i (U_i - U^i(pi))
    subject to U_i >= U^i(a_i, pi^-i) for every action. Returns the best
    run's policies and its regret (0 at an equilibrium)."""
    sizes = game.R.shape[1:]
    cuts = np.cumsum(sizes)[:-1]
    k = len(sizes)

    def unpack(z):                     # z = [pi^1, ..., pi^k, U_1..U_k]
        return np.split(z[:-k], cuts), z[-k:]

    def objective(z):
        pis, U = unpack(z)
        return sum(U[i] - utility(game, pis, i) for i in range(k))

    def gaps(z):                       # >= 0 when no action beats U_i
        pis, U = unpack(z)
        return np.concatenate([U[i] - action_values(game, pis, i)
                               for i in range(k)])

    cons = [{"type": "ineq", "fun": gaps}]
    cons += [{"type": "eq", "fun": lambda z, i=i: unpack(z)[0][i].sum() - 1}
             for i in range(k)]
    bounds = [(0, 1)] * sum(sizes) + [(None, None)] * k
    best = None
    for _ in range(starts):
        pis0 = [rng.dirichlet(np.ones(n)) for n in sizes]
        U0 = [action_values(game, pis0, i).max() for i in range(k)]
        z0 = np.concatenate(pis0 + [np.array(U0)])
        res = minimize(objective, z0, method="SLSQP", bounds=bounds,
                       constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12})
        pis = [np.clip(p, 0, None) for p in unpack(res.x)[0]]
        pis = [p / p.sum() for p in pis]
        r = regret(game, pis)
        if best is None or r < best[1]:
            best = (pis, r)
    return best


def support_enumeration(game, tol=1e-9) -> list:
    """Every Nash equilibrium of a nondegenerate two-player game, exactly.

    For each pair of equal-size supports, solve for the mixes that make
    the other agent indifferent across its support; keep the solution if
    the mixes are valid and no outside action does better. Exponential in
    the number of actions: for validating small games only.
    """
    A, B = game.R[0], game.R[1]        # rows = agent 1's action
    n1, n2 = A.shape
    found = []

    def indifferent(M, rows, cols):
        """Mix x over `rows` making x @ M[rows][:, cols] constant (v)."""
        s = len(rows)
        lhs = np.zeros((s + 1, s + 1))
        # One equation per column c: sum_r x_r M[r, c] - v = 0.
        lhs[:s, :s] = M[np.ix_(rows, cols)].T
        lhs[:s, s] = -1
        lhs[s, :s] = 1                 # and the mix sums to 1
        rhs = np.zeros(s + 1)
        rhs[s] = 1
        try:
            return np.linalg.solve(lhs, rhs)[:s]
        except np.linalg.LinAlgError:
            return None

    for s in range(1, min(n1, n2) + 1):
        for S1 in combinations(range(n1), s):
            for S2 in combinations(range(n2), s):
                # Agent 1 mixes to make agent 2 indifferent, and vice versa.
                xs = indifferent(B, list(S1), list(S2))
                ys = indifferent(A.T, list(S2), list(S1))
                if xs is None or ys is None or min(xs.min(), ys.min()) < -tol:
                    continue
                x, y = np.zeros(n1), np.zeros(n2)
                x[list(S1)], y[list(S2)] = xs, ys
                new = not any(np.allclose(x, fx) and np.allclose(y, fy)
                              for fx, fy in found)
                if regret(game, [x, y]) <= 1e-7 and new:
                    found.append((x, y))
    return found


def correlated_equilibrium(game, objective="utilitarian", agent=0) -> Array:
    """eq. 24.8 by linear programming; returns pi[a1, ..., ak], a
    distribution over joint actions. Objectives (Table 24.1):
    "utilitarian" (total reward), "egalitarian" (the worst-off agent),
    "dictatorial" (agent `agent`), or a weight vector w (maximize
    sum_i w_i U^i: used to trace the payoffs equilibria can reach)."""
    k, shape = game.R.shape[0], game.R.shape[1:]
    joints = list(np.ndindex(*shape))
    n = len(joints)
    rows = []
    # For each agent i, recommendation a_i and alternative b: the gain
    # from switching to b, over the joint actions that recommend a_i, <= 0.
    for i in range(k):
        for ai in range(shape[i]):
            for b in range(shape[i]):
                if b == ai:
                    continue
                row = np.zeros(n)
                for idx, a in enumerate(joints):
                    if a[i] == ai:
                        alt = a[:i] + (b,) + a[i + 1:]
                        row[idx] = game.R[(i,) + alt] - game.R[(i,) + a]
                rows.append(row)
    rows = np.array(rows)
    # U[i, joint]: agent i's reward for each joint action.
    U = np.array([[game.R[(i,) + a] for a in joints] for i in range(k)])
    if objective == "egalitarian":
        # Variables (pi, t): maximize t with t <= every agent's reward.
        c = np.concatenate([np.zeros(n), [-1.0]])
        A_ub = np.vstack([np.hstack([rows, np.zeros((len(rows), 1))]),
                          np.hstack([-U, np.ones((k, 1))])])
        b_ub = np.zeros(len(rows) + k)
        A_eq = np.concatenate([np.ones(n), [0.0]])[None, :]
        bounds = [(0, None)] * n + [(None, None)]
    else:
        if isinstance(objective, str):
            w = np.ones(k) if objective == "utilitarian" else np.eye(k)[agent]
        else:
            w = np.asarray(objective, dtype=float)
        c = -(w @ U)                   # linprog minimizes
        A_ub, b_ub = rows, np.zeros(len(rows))
        A_eq, bounds = np.ones((1, n)), [(0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=bounds, method="highs")
    return np.clip(res.x[:n], 0, None).reshape(shape)


def correlated_utilities(game, pi) -> Array:
    """Each agent's expected reward under a joint-action distribution."""
    return np.array([(game.R[i] * pi).sum() for i in range(game.R.shape[0])])


def uniform_policies(game) -> list:
    return [np.full(n, 1 / n) for n in game.R.shape[1:]]


def iterated_best_response(game, k_max, policies=None) -> list:
    """§24.6: all agents best-respond at once to the others' previous
    policies. Returns the joint policy after each round (0 = start)."""
    policies = uniform_policies(game) if policies is None else policies
    history = [policies]
    for _ in range(k_max):
        policies = [best_response(game, policies, i)
                    for i in range(len(policies))]
        history.append(policies)
    return history


def hierarchical_softmax(game, lam, k, policies=None) -> list:
    """§24.7: level 0 plays uniformly; level k softmax-responds (precision
    lam) to everyone else playing level k - 1."""
    policies = uniform_policies(game) if policies is None else policies
    for _ in range(k):
        policies = [softmax_response(game, policies, i, lam)
                    for i in range(len(policies))]
    return policies


def fictitious_play(game, k_max, counts=None) -> dict:
    """§24.8: each agent counts the others' actions (eq. 24.9) and best-
    responds to those frequencies (ties: lowest index). counts[i][j] is
    agent i's count vector for agent j (default: ones). Returns each
    round's actions and each agent's model of the others."""
    k, sizes = game.R.shape[0], game.R.shape[1:]
    if counts is None:
        N = [[np.ones(sizes[j]) for j in range(k)] for _ in range(k)]
    else:
        N = [[np.array(c, dtype=float) for c in row] for row in counts]
    actions, models = [], []
    for _ in range(k_max):
        model = [[N[i][j] / N[i][j].sum() for j in range(k)]
                 for i in range(k)]
        a = [int(np.argmax(action_values(game, model[i], i)))
             for i in range(k)]
        for i in range(k):             # everyone sees the joint action
            for j in range(k):
                N[i][j][a[j]] += 1
        actions.append(a)
        models.append(model)
    return {"actions": np.array(actions), "models": models}


def project_simplex(y) -> Array:
    """Closest probability vector to y in Euclidean distance (Ch. 23)."""
    mu = np.sort(y)[::-1]
    cs = np.cumsum(mu)
    rho = np.nonzero(mu - (cs - 1) / np.arange(1, len(y) + 1) > 0)[0][-1]
    return np.maximum(y - (cs[rho] - 1) / (rho + 1), 0.0)


def gradient_ascent_play(game, k_max, rng, policies=None, rate=1.0):
    """§24.9: each agent samples an action, then steps its policy along
    R^i(., a^-i), the gradient if the others repeat their last action
    (eq. 24.12), with step rate / sqrt(t), and projects back onto the
    simplex. Returns the joint policy after each round."""
    if policies is None:
        policies = uniform_policies(game)
    policies = [np.array(p, dtype=float) for p in policies]
    history = [policies]
    for t in range(1, k_max + 1):
        a = [rng.choice(len(p), p=p) for p in policies]
        new = []
        for i, p in enumerate(policies):
            # Others as their last actions (one-hot); agent i as itself.
            others = [np.eye(len(q))[a[j]] for j, q in enumerate(policies)]
            others[i] = p
            step = rate / np.sqrt(t) * action_values(game, others, i)
            new.append(project_simplex(p + step))
        policies = new
        history.append(policies)
    return history
