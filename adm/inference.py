"""Inference in Bayesian networks and Gaussian models (ADM Ch. 3).

A factor phi is a table over some variables: `variables` names its axes in
order, and `table` holds one number per joint value of those variables.
CPTs, joint distributions, and intermediate results are all factors.
Lines stay under 80 characters because the Ch 3 page prints this source.
"""

from dataclasses import dataclass
from functools import reduce
from math import prod

import numpy as np

from adm.bayes_net import BayesNet


@dataclass(frozen=True)
class Factor:
    variables: tuple
    table: np.ndarray


def cpt_factor(bn: BayesNet, x: str) -> Factor:
    """X's CPT as a factor over (parents..., X)."""
    return Factor(tuple(bn.parents[x]) + (x,), bn.cpt[x])


def factor_product(f: Factor, g: Factor) -> Factor:
    """phi(x, y, z) = f(x, y) * g(y, z), matching entries on shared variables."""
    variables = f.variables + tuple(v for v in g.variables if v not in f.variables)

    def expand(h: Factor) -> np.ndarray:
        # Reorder h's axes to follow `variables`, then insert a size-1 axis
        # for every variable h doesn't mention. NumPy broadcasting then lines
        # up shared variables and repeats h along the ones it lacks.
        order = [h.variables.index(v) for v in variables if v in h.variables]
        shape = [h.table.shape[h.variables.index(v)] if v in h.variables else 1
                 for v in variables]
        return np.transpose(h.table, order).reshape(shape)

    return Factor(variables, expand(f) * expand(g))


def sum_out(f: Factor, x: str) -> Factor:
    """Marginalize x away: add the table up along x's axis."""
    i = f.variables.index(x)
    return Factor(f.variables[:i] + f.variables[i + 1:], f.table.sum(axis=i))


def condition(f: Factor, evidence: dict) -> Factor:
    """Keep only entries that agree with the evidence, and drop those axes."""
    # An integer index picks one value (and removes the axis); slice(None)
    # keeps the whole axis, like `:` does.
    index = tuple(evidence.get(v, slice(None)) for v in f.variables)
    kept = tuple(v for v in f.variables if v not in evidence)
    return Factor(kept, np.asarray(f.table[index]))


def normalize(f: Factor, order=None) -> Factor:
    """Scale the table to sum to 1, optionally reordering its axes."""
    order = tuple(order or f.variables)
    table = np.transpose(f.table, [f.variables.index(v) for v in order])
    return Factor(order, table / table.sum())


def exact_inference(bn: BayesNet, query, evidence: dict) -> Factor:
    """P(query | evidence) from the full joint: exact but exponential."""
    phi = reduce(factor_product, (cpt_factor(bn, x) for x in bn.parents))
    phi = condition(phi, evidence)
    for v in phi.variables:
        if v not in query:          # every hidden variable gets summed out
            phi = sum_out(phi, v)
    return normalize(phi, query)


def variable_elimination(bn: BayesNet, query, evidence: dict, order) -> Factor:
    """P(query | evidence), summing out hidden variables one at a time.

    `order` lists the hidden variables in the order to eliminate them.
    """
    factors = [condition(cpt_factor(bn, x), evidence) for x in bn.parents]
    for h in order:
        # Only the factors that mention h are multiplied before h is summed
        # out; every other factor is left alone. That is the whole saving
        # over building the full joint.
        involved = [f for f in factors if h in f.variables]
        factors = [f for f in factors if h not in f.variables]
        if involved:
            factors.append(sum_out(reduce(factor_product, involved), h))
    return normalize(reduce(factor_product, factors), query)


def topological_order(parents: dict) -> list:
    """Variables ordered so that every parent comes before its children."""
    order = []
    while len(order) < len(parents):
        for x, pa in parents.items():
            if x not in order and all(p in order for p in pa):
                order.append(x)
    return order


def _cpt_row(bn: BayesNet, sample: dict, x: str) -> np.ndarray:
    """Distribution of x given the parents' values in `sample`."""
    return bn.cpt[x][tuple(sample[q] for q in bn.parents[x])]


def direct_sample(bn: BayesNet, rng, order=None) -> dict:
    """One joint sample: draw each variable given its already-drawn parents."""
    sample = {}
    for x in order or topological_order(bn.parents):
        row = _cpt_row(bn, sample, x)
        sample[x] = rng.choice(len(row), p=row)
    return sample


def rejection_sampling(bn, query, evidence, n, rng) -> Factor:
    """Draw n direct samples; keep those that match the evidence."""
    order = topological_order(bn.parents)
    counts = np.zeros([bn.card[q] for q in query])
    for _ in range(n):
        s = direct_sample(bn, rng, order)
        if all(s[v] == value for v, value in evidence.items()):
            counts[tuple(s[q] for q in query)] += 1
    if counts.sum() == 0:
        raise ValueError("no sample matched the evidence; draw more samples")
    return Factor(tuple(query), counts / counts.sum())


def likelihood_weighted_sampling(bn, query, evidence, n, rng) -> Factor:
    """Fix evidence variables instead of sampling them, and weight each
    sample by how likely that evidence was given its sampled parents."""
    order = topological_order(bn.parents)
    weights = np.zeros([bn.card[q] for q in query])
    for _ in range(n):
        s, w = {}, 1.0
        for x in order:
            row = _cpt_row(bn, s, x)
            if x in evidence:
                s[x] = evidence[x]
                w *= row[s[x]]       # P(observed value | sampled parents)
            else:
                s[x] = rng.choice(len(row), p=row)
        weights[tuple(s[q] for q in query)] += w
    return Factor(tuple(query), weights / weights.sum())


def blanket_distribution(bn: BayesNet, sample: dict, x: str) -> np.ndarray:
    """P(x | all other variables), which depends only on x's Markov blanket:
    P(x | pa(x)) times P(c | pa(c)) for each child c of x."""
    children = [c for c, pa in bn.parents.items() if x in pa]
    p = np.empty(bn.card[x])
    for value in range(bn.card[x]):
        s = {**sample, x: value}      # try x = value, keep everything else
        p[value] = prod(_cpt_row(bn, s, y)[s[y]] for y in [x, *children])
    return p / p.sum()


def gibbs_sampling(bn, query, evidence, n, rng, burn_in=200, thin=1):
    """Markov chain: resample one hidden variable at a time from its
    distribution given everything else. Returns P(query | evidence)."""
    # Start anywhere consistent with the evidence.
    sample = {**direct_sample(bn, rng), **evidence}
    hidden = [x for x in bn.parents if x not in evidence]
    counts = np.zeros([bn.card[q] for q in query])
    for k in range(burn_in + n * thin):
        for x in hidden:
            p = blanket_distribution(bn, sample, x)
            sample[x] = rng.choice(len(p), p=p)
        # Skip the first `burn_in` sweeps (the chain hasn't forgotten its
        # start yet), then keep one sweep in every `thin`.
        if k >= burn_in and (k - burn_in) % thin == 0:
            counts[tuple(sample[q] for q in query)] += 1
    return Factor(tuple(query), counts / counts.sum())


def naive_bayes_posterior(prior, likelihoods, observations) -> np.ndarray:
    """P(c | o_1..o_n), proportional to P(c) * prod_i P(o_i | c).

    likelihoods[i][c, o] = P(feature i = o | class c).
    """
    p = np.array(prior, dtype=float)
    for L, o in zip(likelihoods, observations):
        p = p * L[:, o]
    return p / p.sum()


def gaussian_condition(mu, Sigma, a, b, values):
    """Mean and covariance of x_a given x_b = values, for a joint Gaussian.

    a and b are lists of indices into mu and Sigma.
    """
    S_ab, S_bb = Sigma[np.ix_(a, b)], Sigma[np.ix_(b, b)]
    # K = S_ab S_bb^-1: how far the query moves per unit of surprise in
    # the observed variables. solve() avoids forming the inverse.
    K = np.linalg.solve(S_bb, S_ab.T).T
    mu_a = mu[a] + K @ (np.asarray(values) - mu[b])
    Sigma_a = Sigma[np.ix_(a, a)] - K @ S_ab.T
    return mu_a, Sigma_a
