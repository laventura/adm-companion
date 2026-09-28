"""Aircraft collision avoidance (ADM Appendix F.6), from its description.

Our aircraft meets an intruder head on. The state is our altitude h
relative to the intruder (m), our vertical rate hdot (m/s), the previous
advisory, and the time to potential collision t_col (s). Each second the
policy picks an advisory: none, descend at 5 m/s, or climb at 5 m/s. Then
    h    <- h + hdot
    hdot <- hdot + hddot + nu,   nu in {-2, 0, 2} w.p. {1/4, 1/2, 1/4}
where hddot pushes hdot toward the advised rate, at most 1 m/s^2
(hddot = 0 with no advisory). At t_col = 0 there is a collision if
|h| < 50 m. Costs: 1 for a collision, 0.01 for each change of advisory.

Starting from integer h and hdot, every later h and hdot is an integer
too, so the problem is an exact finite MDP on an integer lattice. This
module solves and evaluates it on that lattice by backward induction
(no interpolation), which gives exact references for every sampling
method on the Ch 14 page. Mirrors julia/src/problems/collision_avoidance.jl.
"""

from dataclasses import dataclass

import numpy as np

ADVISORIES = ("none", "descend", "climb")
RATES = (0, -5, 5)             # advised vertical rate (m/s); unused for "none"
T_START = 40                   # t_col at the start of an episode (s)
H_START, HDOT_START = 200, 10  # start: h, hdot uniform on the integers in +-these
COLLISION_H = 50               # collision if |h| < 50 m at t_col = 0


@dataclass(frozen=True)
class Model:
    """One version of the dynamics and costs. The planning model is the
    default; robustness checks evaluate policies on other versions."""
    noise: int = 2               # nu is -noise, 0 or +noise (m/s^2)
    p_noise: float = 0.5         # probability that nu != 0 (split evenly)
    accel_limit: int = 1         # largest |hddot| the pilot uses (m/s^2)
    compliance: float = 1.0      # probability the pilot follows an advisory
    collision_cost: float = 1.0
    change_cost: float = 0.01
    h_max: int = 500             # lattice bounds; states beyond are clamped
    hdot_max: int = 30

    @property
    def h(self):
        return np.arange(-self.h_max, self.h_max + 1)

    @property
    def hdot(self):
        return np.arange(-self.hdot_max, self.hdot_max + 1)

    def noise_outcomes(self):
        """The (nu, probability) pairs of the vertical-rate noise."""
        p = self.p_noise / 2
        return ((-self.noise, p), (0, 1 - self.p_noise), (self.noise, p))


def accel(model, hdot, a):
    """hddot for advisory a: toward the advised rate, capped at the limit.
    For integer hdot this is clip(rate - hdot) (F.9 with the step reaching
    the rate when it is within the limit)."""
    if a == 0:
        return np.zeros_like(hdot)
    return np.clip(RATES[a] - hdot, -model.accel_limit, model.accel_limit)


def outcomes(model, a):
    """All next states of every lattice state under advisory a.

    Returns a list of (i_h, i_hdot, p): lattice indices of the next h and
    hdot, each of shape (n_h, n_hdot), and the probability of that branch.
    Branches: the pilot complies or not, times the three noise values.
    """
    H, D = np.meshgrid(model.h, model.hdot, indexing="ij")  # (n_h, n_hdot)
    h2 = np.clip(H + D, -model.h_max, model.h_max)          # uses the old hdot
    out = []
    for comply, pc in ((True, model.compliance), (False, 1 - model.compliance)):
        if pc == 0:
            continue
        # A pilot who ignores the advisory keeps hddot = 0, as with "none".
        acc = accel(model, D, a) if comply else np.zeros_like(D)
        for nu, pn in model.noise_outcomes():
            d2 = np.clip(D + acc + nu, -model.hdot_max, model.hdot_max)
            # Shift values to 0-based lattice indices.
            out.append((h2 + model.h_max, d2 + model.hdot_max, pc * pn))
    return out


def collided(model):
    """1.0 where |h| < 50 m, as a (n_h, n_hdot) array; the t_col = 0 value."""
    H, _ = np.meshgrid(model.h, model.hdot, indexing="ij")
    return (np.abs(H) < COLLISION_H).astype(float)


def expect(model, U):
    """E[U(next state)] for each lattice state and each advisory.

    U has shape (3, n_h, n_hdot): the value at t_col - 1 indexed by the
    advisory just taken (it becomes a_prev). Returns (3, n_h, n_hdot).
    """
    EU = np.zeros_like(U)
    for a in range(3):
        for ih, idd, p in outcomes(model, a):
            EU[a] += p * U[a][ih, idd]     # paired fancy indexing: a gather
    return EU


def optimal_policy(model=Model(), models=None):
    """Backward induction for the minimum expected cost.

    Returns (policy, cost): policy[t, a_prev, i_h, i_hdot] is the advisory
    at t_col = t (row 0 is unused) and cost[a_prev, i_h, i_hdot] is the
    optimal expected cost at t_col = T_START. With `models`, a list of
    Models, it is robust dynamic programming: each advisory is scored by
    its worst model, then the best advisory is chosen.
    """
    models = models or [model]
    change = model.change_cost * (1 - np.eye(3))  # change[a_prev, a]
    U = np.repeat(model.collision_cost * collided(model)[None], 3, axis=0)
    policy = np.zeros((T_START + 1,) + U.shape, dtype=np.int8)
    for t in range(1, T_START + 1):
        EU = np.max([expect(m, U) for m in models], axis=0)  # worst model
        # Q[a_prev, a, ...]: cost of switching to a, then continuing.
        Q = change[:, :, None, None] + EU[None]
        policy[t] = Q.argmin(axis=1)
        U = Q.min(axis=1)
    return policy, U


def simple_policy(model=Model(), h_thresh=50, t_thresh=30):
    """The threshold policy: when |h| < h_thresh and t_col < t_thresh,
    climb if we are above the intruder (h > 0), else descend. As a table
    shaped like optimal_policy's output; it ignores hdot and a_prev."""
    H, _ = np.meshgrid(model.h, model.hdot, indexing="ij")
    act = np.where(np.abs(H) < h_thresh, np.where(H > 0, 2, 1), 0)
    policy = np.zeros((T_START + 1, 3) + H.shape, dtype=np.int8)
    for t in range(1, T_START + 1):
        if t < t_thresh:
            policy[t] = act
    return policy


def sample_starts(n, rng):
    """n start states: integer h and hdot, uniform on +-H_START, +-HDOT_START."""
    return (rng.integers(-H_START, H_START + 1, n),
            rng.integers(-HDOT_START, HDOT_START + 1, n))


def _index(model, h, hdot):
    """Lattice indices of (h, hdot), clamped to the lattice."""
    return (np.clip(h, -model.h_max, model.h_max) + model.h_max,
            np.clip(hdot, -model.hdot_max, model.hdot_max) + model.hdot_max)


def simulate(model, policy, h, hdot, rng, tilt=0.0, guide=None, eps=1e-6,
             record=False):
    """Roll out len(h) episodes side by side from integer starts h, hdot.

    By default the noise is drawn from the model. Two proposals bias it
    toward collisions, for importance sampling:
      tilt > 0: P'(nu) is proportional to P(nu) exp(tilt * s), where
        s = +1 for a push toward the intruder (it reduces |h|), -1 for a
        push away and 0 for nu = 0;
      guide: risk-to-go tables from evaluate(..., history=True). P'(nu)
        is proportional to P(nu) (guide[t-1](next state) + eps): eq. 14.13
        applied one step at a time. eps keeps every branch possible.
    The returned log weights, sum log P(nu) / P'(nu), undo the bias
    (eq. 14.12); without a proposal they are all 0.

    Returns (h_final, changes, log_w), plus the (n, T_START + 1) altitude
    paths when record is True. The dynamics are exact (no clamping); only
    the policy lookup clamps to the lattice.
    """
    h, hdot = np.array(h), np.array(hdot)
    n = len(h)
    a_prev = np.zeros(n, dtype=int)
    changes, log_w = np.zeros(n), np.zeros(n)
    path = [h.copy()]
    nus, p_nu = zip(*model.noise_outcomes())
    for t in range(T_START, 0, -1):
        a = policy[t][(a_prev,) + _index(model, h, hdot)]
        changes += a != a_prev
        comply = rng.random(n) < model.compliance
        acc = np.zeros(n, dtype=int)
        for k in (1, 2):                  # advised rates; "none" keeps 0
            m = comply & (a == k)
            acc[m] = accel(model, hdot[m], k)
        h2 = h + hdot                     # the next altitude uses the old rate
        # Proposal weight of each noise value, shape (3, n), unnormalized.
        w = np.array([np.full(n, p) for p in p_nu])
        if tilt:
            toward = -np.sign(h)          # +1 below the intruder, -1 above
            w *= np.exp(tilt * np.sign(nus)[:, None] * toward)
        if guide is not None:
            for k, nu in enumerate(nus):
                # guide[t - 1][a_prev', i_h', i_hdot'] with a_prev' = a.
                w[k] *= guide[t - 1][(a,) + _index(model, h2, hdot + acc + nu)] + eps
        q = w / w.sum(axis=0)             # proposal probabilities P'(nu)
        # Inverse-CDF draw of the branch: count the cumulative
        # probabilities below u (the min guards against rounding).
        k = np.minimum((rng.random(n) > np.cumsum(q, axis=0)).sum(axis=0), 2)
        log_w += np.log(np.take(p_nu, k)) - np.log(q[k, np.arange(n)])
        h, hdot, a_prev = h2, hdot + acc + np.take(nus, k), a
        path.append(h.copy())
    out = (h, changes, log_w)
    return out + (np.array(path).T,) if record else out


def evaluate(model, policy, step_metric=None, final_metric=None,
             history=False):
    """Exact policy evaluation of an additive metric (eq. 14.3) by
    backward induction.

    step_metric(a_prev, a) scores each step (default: 0) and
    final_metric is a (n_h, n_hdot) array scored at t_col = 0 (default:
    the collision indicator). Returns V[a_prev, i_h, i_hdot] at
    t_col = T_START: the expected total from each start state. With
    history=True, returns the list of V at t_col = 0, 1, ..., T_START.
    """
    final = collided(model) if final_metric is None else final_metric
    V = np.repeat(final[None], 3, axis=0)
    Vs = [V]
    a_prev = np.arange(3)[:, None, None]
    i = np.arange(V.shape[1])[:, None]            # row index, broadcasts
    j = np.arange(V.shape[2])[None, :]            # column index
    for t in range(1, T_START + 1):
        EV = expect(model, V)                     # (a, n_h, n_hdot)
        a = policy[t]                             # (a_prev, n_h, n_hdot)
        V = EV[a, i, j]                           # V[p, i, j] = EV[a[p, i, j], i, j]
        if step_metric is not None:
            V = V + step_metric(a_prev, a)
        Vs.append(V)
    return Vs if history else V


def adversary(model, policy, lam=None):
    """Solve the adversarial problem for a fixed policy (ADM §14.5).

    The adversary picks each transition's outcome (compliance and noise
    branch). Its step reward is our cost plus lam * log T(branch), and
    it also collects our collision cost at t_col = 0 (eq. 14.15). With
    lam=None it looks for the most likely failure instead (eq. 14.16):
    step reward log T(branch), 0 for ending in a collision, -inf for not.

    Returns (W, choice): W[a_prev, i_h, i_hdot] is the adversary's best
    total from each state at t_col = T_START, and choice[t] (same shape)
    is the index of its best branch at t_col = t, in the order of
    outcomes(). Deterministic search, so dynamic programming is exact.
    """
    hit = collided(model)
    if lam is None:
        W = np.where(hit > 0, 0.0, -np.inf)
        W = np.repeat(W[None], 3, axis=0)
    else:
        W = np.repeat(model.collision_cost * hit[None], 3, axis=0)
    a_prev = np.arange(3)[:, None, None]
    i = np.arange(W.shape[1])[:, None]
    j = np.arange(W.shape[2])[None, :]
    choice = [None]
    for t in range(1, T_START + 1):
        # best[a, k]: value of branch k after advisory a, for every state.
        best = np.array([[(np.log(p) if lam is None else lam * np.log(p)) + W[a][ih, idd]
                          for ih, idd, p in outcomes(model, a)] for a in range(3)])
        a = policy[t]                                 # (a_prev, n_h, n_hdot)
        per_branch = best[a, :, i, j]                 # (a_prev, n_h, n_hdot, k)
        choice.append(per_branch.argmax(axis=-1))
        W = per_branch.max(axis=-1)
        if lam is not None:
            W = W + model.change_cost * (a != a_prev)
    return W, choice


def replay(model, policy, choice, h, hdot):
    """Follow the adversary's choices from one start (h, hdot).
    Returns a list of (t_col, h, hdot, advisory, nu) per step, then the
    final (0, h, hdot, None, None)."""
    a_prev, steps = 0, []
    for t in range(T_START, 0, -1):
        ih, idd = _index(model, h, hdot)
        a = int(policy[t][a_prev, ih, idd])
        k = int(choice[t][a_prev, ih, idd])
        # Recover the branch: compliance outer loop, noise inner loop.
        branches = [(c, nu) for c, pc in ((True, model.compliance),
                                          (False, 1 - model.compliance)) if pc > 0
                    for nu, _ in model.noise_outcomes()]
        comply, nu = branches[k]
        acc = int(accel(model, np.array(hdot), a)) if comply else 0
        steps.append((t, h, hdot, a, nu))
        h, hdot, a_prev = h + hdot, hdot + acc + nu, a
    steps.append((0, h, hdot, None, None))
    return steps


def start_average(model, V):
    """Average of V[a_prev = none] over the start distribution (eq. 14.4)."""
    ih = np.arange(-H_START, H_START + 1) + model.h_max
    idd = np.arange(-HDOT_START, HDOT_START + 1) + model.hdot_max
    return V[0][np.ix_(ih, idd)].mean()
