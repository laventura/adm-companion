"""Learning Bayesian-network parameters from data (ADM Ch. 4).

A dataset is a list of records, one per observation: dicts that map every
variable name to its value (0, 1, ...), or to None if it wasn't recorded.
Lines stay under 80 characters because the Ch 4 page prints this source.
"""

from itertools import product

import numpy as np

from adm.bayes_net import BayesNet, probability
from adm.inference import direct_sample, topological_order


def sample_dataset(bn: BayesNet, m: int, rng) -> list:
    """m complete records drawn from the network by direct sampling (Ch 3)."""
    order = topological_order(bn.parents)
    return [direct_sample(bn, rng, order) for _ in range(m)]


def count_table(card: dict, parents: dict, data: list) -> dict:
    """m_ijk: how often variable i took value k with its parents in
    configuration j. counts[x] has the same shape as x's CPT.

    A record is skipped for x if x or any of its parents is missing.
    """
    counts = {x: np.zeros([card[q] for q in pa] + [card[x]])
              for x, pa in parents.items()}
    for record in data:
        for x, pa in parents.items():
            family = [record[q] for q in pa] + [record[x]]
            if None not in family:
                # The family's values index one cell: parents..., then x.
                counts[x][tuple(family)] += 1
    return counts


def normalize_rows(counts: np.ndarray) -> np.ndarray:
    """Turn counts into a CPT: divide each row (last axis) by its total.

    A parent configuration with no data has no estimate (0/0), so its row
    falls back to uniform.
    """
    totals = counts.sum(axis=-1, keepdims=True)
    uniform = np.full_like(counts, 1 / counts.shape[-1])
    safe = np.where(totals > 0, totals, 1)          # avoid dividing by 0
    return np.where(totals > 0, counts / safe, uniform)


def ml_estimate(card: dict, parents: dict, data: list) -> BayesNet:
    """Maximum likelihood: each CPT row is the observed frequencies,
    theta_ijk = m_ijk / sum_k m_ijk."""
    counts = count_table(card, parents, data)
    cpt = {x: normalize_rows(m) for x, m in counts.items()}
    return BayesNet(dict(card), dict(parents), cpt)


def bayes_estimate(card: dict, parents: dict, data: list,
                   alpha: float = 1.0) -> BayesNet:
    """Posterior mean under a Dirichlet prior that adds `alpha` pseudocounts
    to every cell: (m_ijk + alpha) / sum_k (m_ijk + alpha).

    alpha = 1 is the uniform prior; a larger alpha pulls estimates harder
    toward uniform, which matters most for rarely seen parent values.
    """
    counts = count_table(card, parents, data)
    cpt = {x: normalize_rows(m + alpha) for x, m in counts.items()}
    return BayesNet(dict(card), dict(parents), cpt)


def beta_update(alpha: float, beta: float, successes: int, failures: int):
    """Beta(alpha, beta) prior + data -> Beta posterior. Each observation
    adds one to the matching parameter, which is why the prior's alpha and
    beta act like counts seen before the data (pseudocounts)."""
    return alpha + successes, beta + failures


def gaussian_ml(x) -> tuple:
    """ML mean and variance of 1-D data. The ML variance divides by m,
    not m - 1, so it is slightly smaller than the "sample variance"."""
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    return mu, np.mean((x - mu) ** 2)


def kernel_density(samples, bandwidth: float):
    """p(x) = (1/m) sum_i N(x | x_i, h^2): a Gaussian bump of width h on
    every data point. Returns a function of x (a number or an array)."""
    samples = np.asarray(samples, dtype=float)

    def p(x):
        # The extra axis (None) pairs every x with every sample, so z has
        # shape (..., m); averaging over the last axis sums the m bumps.
        z = (np.asarray(x, dtype=float)[..., None] - samples) / bandwidth
        bumps = np.exp(-0.5 * z**2) / (bandwidth * np.sqrt(2 * np.pi))
        return bumps.mean(axis=-1)

    return p


def impute_mode(data: list) -> list:
    """Fill each missing value with that variable's most common observed
    value. Simple, but it biases the data toward the common case."""
    names = data[0].keys()
    mode = {}
    for x in names:
        seen = [r[x] for r in data if r[x] is not None]
        mode[x] = max(set(seen), key=seen.count)
    return [{x: mode[x] if r[x] is None else r[x] for x in names}
            for r in data]


def _completions(bn: BayesNet, record: dict):
    """Every way to fill in a record's missing values, with each filled-in
    record's probability under the network (chain rule)."""
    missing = [x for x, v in record.items() if v is None]
    for values in product(*(range(bn.card[x]) for x in missing)):
        full = {**record, **dict(zip(missing, values))}
        yield full, probability(bn, full)


def log_likelihood(bn: BayesNet, data: list) -> float:
    """l(theta) = sum over records of log P(observed values), where the
    missing values are summed out."""
    return sum(np.log(sum(p for _, p in _completions(bn, r))) for r in data)


def expected_counts(bn: BayesNet, data: list) -> dict:
    """E-step: counts in which each missing value is split across its
    possible values, in proportion to their probability given the rest of
    the record under the current network."""
    counts = {x: np.zeros_like(t) for x, t in bn.cpt.items()}
    for record in data:
        fills = list(_completions(bn, record))
        total = sum(p for _, p in fills)
        for full, p in fills:
            weight = p / total           # P(this completion | observed values)
            for x, pa in bn.parents.items():
                counts[x][tuple(full[q] for q in pa) + (full[x],)] += weight
    return counts


def expectation_maximization(bn0: BayesNet, data: list, iterations: int = 30):
    """Alternate E (expected counts under the current network) and M
    (re-estimate every CPT from those counts, as in maximum likelihood).

    Returns the final network and the log-likelihood before and after each
    iteration; each iteration can only raise it or leave it unchanged.
    """
    bn, history = bn0, [log_likelihood(bn0, data)]
    for _ in range(iterations):
        counts = expected_counts(bn, data)
        cpt = {x: normalize_rows(m) for x, m in counts.items()}
        bn = BayesNet(bn.card, bn.parents, cpt)
        history.append(log_likelihood(bn, data))
    return bn, history
