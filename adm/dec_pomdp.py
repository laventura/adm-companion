"""Dec-POMDPs (ADM Ch. 27): a POMG whose agents share one reward.

Games are stored as POMGs (adm.problems.multicaregiver) with identical
reward rows R[0] == R[1]; plans and their evaluation come from adm.pomg.
"""

from itertools import product

import numpy as np
from scipy.optimize import minimize

from adm import pomg

Array = np.ndarray


def shared(game):
    """A POMG with both reward rows set to the team's total reward."""
    total = game.R.sum(axis=0)
    return game.__class__(game.T, game.O, np.stack([total, total]), game.gamma,
                          game.states, game.actions, game.observations,
                          game.name + " (team)")


def best_joint_plan(game, b, plans, U=None):
    """The joint plan with the highest shared value from b, among
    plans[0] x plans[1], by checking every pair. Returns (p1, p2, value)."""
    U = pomg.plan_evaluator(game) if U is None else U
    best = max(((p1, p2, pomg.utility(U, b, p1, p2)[0])
                for p1, p2 in product(*plans)), key=lambda t: t[2])
    return best


def exact_best_plans(game, b, horizon, U=None):
    """Brute force over every joint plan of the given horizon, for two
    agents with two observations each. A plan is (a, (p_obs0, p_obs1)), so
    a joint plan's value is its first reward plus four lookups in the table
    W of (horizon - 1)-step joint values, one per joint observation: all
    |A|^2 |P|^4 of them can be scored at once with NumPy broadcasting."""
    U = pomg.plan_evaluator(game) if U is None else U
    nS, nA = game.T.shape[0], game.T.shape[1]
    if horizon == 1:                   # one step: the best joint action
        v, a1, a2 = max((b @ game.R[0, :, a1, a2], a1, a2)
                        for a1, a2 in product(range(nA), repeat=2))
        return (a1,), (a2,), float(v)
    subs = pomg.all_plans(nA, 2, horizon - 1)
    n = len(subs)
    # W[k1, k2, s2]: shared value of subplans k1 and k2 from next state s2.
    W = np.array([[[U(p, q, s)[0] for s in range(nS)] for q in subs]
                  for p in subs])
    best = (-np.inf, None)
    for a1, a2 in product(range(nA), repeat=2):
        first = b @ game.R[0, :, a1, a2]
        # w[s2, o1, o2]: probability of next state s2 and joint observation.
        w = np.einsum("s,st,tuv->tuv", b, game.T[:, a1, a2], game.O[a1, a2])
        # M[o1][o2][k1, k2] = gamma sum_s2 w W: the value of subplan pair
        # (k1 after o1, k2 after o2), weighted by that observation pair.
        M = game.gamma * np.einsum("tuv,ijt->uvij", w, W)
        # Axes: agent 1's subplans after obs 0 and 1, agent 2's after 0, 1.
        V = (first + M[0, 0][:, None, :, None] + M[0, 1][:, None, None, :]
             + M[1, 0][None, :, :, None] + M[1, 1][None, :, None, :])
        k = np.unravel_index(np.argmax(V), V.shape)
        if V[k] > best[0]:
            p1 = (a1, (subs[k[0]], subs[k[1]]))
            p2 = (a2, (subs[k[2]], subs[k[3]]))
            best = (float(V[k]), (p1, p2))
    return best[1][0], best[1][1], best[0]


def optimal_plans(game, b, horizon, rng, U=None):
    """§27.3: dynamic programming with pruning (as for POMGs), then the best
    surviving joint plan. Pruning only drops plans that are never a best
    response, so the optimal joint plan survives."""
    U = pomg.plan_evaluator(game) if U is None else U
    plans, counts = pomg.dynamic_programming(game, horizon, rng, U)
    return best_joint_plan(game, b, plans, U), counts


def iterated_best_response(game, b, plans, rng, k_max=50, U=None):
    """§27.4 (JESP): start from random plans; in random agent order, switch
    each agent to its best plan against the other's (ties keep the current
    plan) until nobody changes. Returns the joint plan and its value."""
    U = pomg.plan_evaluator(game) if U is None else U
    joint = [plans[0][rng.integers(len(plans[0]))],
             plans[1][rng.integers(len(plans[1]))]]
    value = lambda p1, p2: pomg.utility(U, b, p1, p2)[0]
    for _ in range(k_max):
        changed = False
        for i in rng.permutation(2):
            def v(p):
                return value(p, joint[1]) if i == 0 else value(joint[0], p)
            best = max(plans[i], key=v)
            if v(best) > v(joint[i]) + 1e-9:        # ties keep the current plan
                joint[i], changed = best, True
        if not changed:
            break
    return joint[0], joint[1], value(*joint)


def explore_belief(game, b, steps, rng) -> Array:
    """Belief after `steps` random joint actions and sampled joint
    observations from b (the heuristic of memory-bounded search)."""
    nS, n1, n2 = game.T.shape[:3]
    s = rng.choice(nS, p=b)
    for _ in range(steps):
        a1, a2 = rng.integers(n1), rng.integers(n2)
        s2 = rng.choice(nS, p=game.T[s, a1, a2])
        flat = game.O[a1, a2, s2].ravel()
        o1, o2 = divmod(rng.choice(len(flat), p=flat), game.O.shape[4])
        # Filter with the joint action and joint observation (Ch. 19).
        b = game.O[a1, a2, :, o1, o2] * (b @ game.T[:, a1, a2, :])
        b, s = b / b.sum(), s2
    return b


def memory_bounded_search(game, b, horizon, max_plans, rng, U=None):
    """§27.5 (MBDP): keep at most max_plans plans per agent. At each step,
    expand every kept plan by one step; then, max_plans times, sample a
    belief that many steps from the end and keep the best joint plan there.
    Returns the best kept joint plan from b and its value."""
    U = pomg.plan_evaluator(game) if U is None else U
    nA, nO = game.T.shape[1:3], game.O.shape[3:]
    kept = [[(a,) for a in range(nA[i])] for i in range(2)]
    for t in range(1, horizon):
        grown = [pomg.expand(kept[i], nA[i], nO[i]) for i in range(2)]
        kept = [[], []]
        for _ in range(max_plans):
            bt = explore_belief(game, b, horizon - t - 1, rng)
            p1, p2, _ = best_joint_plan(game, bt, grown, U)
            for i, p in enumerate((p1, p2)):
                if p not in kept[i]:
                    kept[i].append(p)
    return best_joint_plan(game, b, kept, U)


def centralized_value(game, b, horizon) -> float:
    """The team's optimal value if the agents could share observations
    (an MPOMDP): forward search over beliefs, joint actions and joint
    observations. An upper bound on any Dec-POMDP policy's value."""
    nS, n1, n2 = game.T.shape[:3]
    R = game.R[0]

    def value(b, h):
        best = -np.inf
        for a1, a2 in product(range(n1), range(n2)):
            q = b @ R[:, a1, a2]
            if h > 1:
                pred = b @ game.T[:, a1, a2, :]          # predicted next state
                for o1, o2 in product(*map(range, game.O.shape[3:])):
                    joint = game.O[a1, a2, :, o1, o2] * pred
                    p = joint.sum()
                    if p > 0:
                        q += game.gamma * p * value(joint / p, h - 1)
            best = max(best, q)
        return best

    return value(np.asarray(b, dtype=float), horizon)


def optimize_controllers(game, b, n_nodes, rng, starts=5):
    """§27.6: the best pair of n-node stochastic controllers for the shared
    reward, by SLSQP over psi and eta (values from the linear solve of
    eq. 26.3, starting both controllers at node 0). Returns the value."""
    nA, nO = game.T.shape[1], game.O.shape[3]
    size = n_nodes * nA + n_nodes * nA * nO * n_nodes

    def unpack(z):
        psi = z[:n_nodes * nA].reshape(n_nodes, nA)
        eta = z[n_nodes * nA:].reshape(n_nodes, nA, nO, n_nodes)
        return psi, eta

    def negative_value(z):
        c1, c2 = unpack(z[:size]), unpack(z[size:])
        return -pomg.evaluate_controllers(game, c1, c2)[0, 0, 0] @ b

    # Every psi row (over actions) and eta row (over next nodes) sums to 1.
    rows = [(k * nA, nA) for k in range(n_nodes)]
    rows += [(n_nodes * nA + k * n_nodes, n_nodes)
             for k in range(n_nodes * nA * nO)]
    def row_sum(o, w):                 # constraint: z[o:o+w] sums to 1
        return {"type": "eq", "fun": lambda z: z[o:o + w].sum() - 1}
    cons = [row_sum(off + a, w) for off in (0, size) for a, w in rows]
    best = -np.inf
    for _ in range(starts):
        z0 = np.concatenate([rng.dirichlet(np.ones(w)) for _ in range(2)
                             for _, w in rows])
        res = minimize(negative_value, z0, method="SLSQP",
                       bounds=[(0, 1)] * (2 * size), constraints=cons,
                       options={"maxiter": 300})
        best = max(best, -res.fun)
    return best
