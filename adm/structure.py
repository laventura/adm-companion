"""Learning the structure of a Bayesian network (ADM Ch. 5).

A graph is a dict: parents[x] is a tuple of x's parents. Data is a list of
complete records, as in Ch 4. Lines stay under 80 characters because the
Ch 5 page prints this source.
"""

from itertools import combinations, product

import numpy as np
from scipy.special import gammaln

from adm.learning import count_table


def score_component(counts: np.ndarray, alpha) -> float:
    """One variable's term of the Bayesian score, from its counts m_ijk.

    counts has shape (parent values..., own values); alpha is a pseudocount
    per cell (a number, or an array of the same shape).
    """
    r = counts.shape[-1]
    m = counts.reshape(-1, r)              # one row per parent combination j
    a = np.broadcast_to(np.asarray(alpha, dtype=float), counts.shape)
    a = a.reshape(-1, r)
    # gammaln(x) = ln Gamma(x). Row j contributes
    #   ln G(a_j0) - ln G(a_j0 + m_j0)
    #   + sum_k [ln G(a_jk + m_jk) - ln G(a_jk)],
    # where a_j0 and m_j0 are the row totals.
    return float(np.sum(gammaln(a.sum(1)) - gammaln(a.sum(1) + m.sum(1)))
                 + np.sum(gammaln(a + m) - gammaln(a)))


def bayesian_score(card: dict, parents: dict, data: list,
                   prior: str = "uniform", kappa: float = 1.0) -> float:
    """ln P(data | graph): the Bayesian score with a uniform graph prior.

    prior="uniform" puts one pseudocount in every cell. prior="bdeu" spreads
    `kappa` pseudocounts evenly over each variable's table, which gives
    Markov-equivalent graphs exactly the same score.
    """
    total = 0.0
    for x, m in count_table(card, parents, data).items():
        if prior == "bdeu":
            alpha = kappa / m.size    # m.size = parent combos x values
        else:
            alpha = 1.0
        total += score_component(m, alpha)
    return total


def family_scores(card: dict, data: list, prior: str = "uniform",
                  kappa: float = 1.0) -> dict:
    """The score term of every (variable, parent set) pair.

    The Bayesian score is a sum of one term per variable, and each term
    depends only on that variable and its parents. So a graph's score is
    sum(table[x, parents[x]] for x), and whole searches reuse the table.
    Parent sets are stored as sorted tuples.
    """
    table = {}
    nodes = list(card)
    for x in nodes:
        others = [v for v in nodes if v != x]
        for k in range(len(others) + 1):
            for pa in combinations(others, k):
                # Score just the family x <- pa: a one-variable network.
                family = {x: pa, **{p: () for p in pa}}
                m = count_table(card, family, data)[x]
                alpha = kappa / m.size if prior == "bdeu" else 1.0
                table[x, tuple(sorted(pa))] = score_component(m, alpha)
    return table


def graph_score(table: dict, parents: dict) -> float:
    """A graph's Bayesian score, summed from precomputed family terms."""
    return sum(table[x, tuple(sorted(pa))] for x, pa in parents.items())


def is_acyclic(parents: dict) -> bool:
    """True if the graph has no directed cycle (repeatedly remove sources)."""
    remaining = dict(parents)
    while remaining:
        # A node whose parents have all been removed can't be on a cycle.
        free = [x for x, pa in remaining.items()
                if not any(p in remaining for p in pa)]
        if not free:
            return False                    # every node waits on another
        for x in free:
            del remaining[x]
    return True


def k2_search(score, order: list, max_parents: int = 2) -> dict:
    """K2: for each variable in order, greedily add the earlier variable
    that raises the score most, until no addition helps.

    `score` maps a graph to its score. Parents come only from earlier in
    the order, so no cycle can form.
    """
    parents = {x: () for x in order}
    for i, x in enumerate(order):
        best = score(parents)
        while len(parents[x]) < max_parents:
            options = [p for p in order[:i] if p not in parents[x]]
            scored = [(score({**parents, x: parents[x] + (p,)}), p)
                      for p in options]
            if not scored or max(scored)[0] <= best:
                break                       # no candidate improves the score
            best, p = max(scored)
            parents[x] = parents[x] + (p,)
    return parents


def neighbors(parents: dict):
    """Graphs one edit away: add, remove, or reverse a single edge."""
    nodes = list(parents)
    for a, b in combinations(nodes, 2):
        for u, v in ((a, b), (b, a)):
            if u in parents[v]:             # edge u -> v exists
                removed = tuple(p for p in parents[v] if p != u)
                yield {**parents, v: removed}                          # remove
                yield {**parents, v: removed, u: parents[u] + (v,)}   # reverse
            elif v not in parents[u]:       # no edge either way
                yield {**parents, v: parents[v] + (u,)}               # add


def local_search(score, start: dict, max_steps: int = 50):
    """Hill climbing: move to the best-scoring acyclic neighbor while that
    improves the score. Returns the final graph and the score after each
    step. `score` maps a graph to its score.
    """
    graph, scores = start, [score(start)]
    for _ in range(max_steps):
        candidates = [(score(g), i, g)
                      for i, g in enumerate(neighbors(graph)) if is_acyclic(g)]
        best_score, _, best = max(candidates)   # i breaks ties, never g
        if best_score <= scores[-1]:
            break                              # a local maximum
        graph = best
        scores.append(best_score)
    return graph, scores


def skeleton(parents: dict) -> set:
    """The graph's edges with directions dropped."""
    return {frozenset((p, x)) for x, pa in parents.items() for p in pa}


def v_structures(parents: dict) -> set:
    """Triples a -> c <- b where a and b are not adjacent."""
    edges = skeleton(parents)
    return {(frozenset((a, b)), c) for c, pa in parents.items()
            for a, b in combinations(pa, 2) if frozenset((a, b)) not in edges}


def markov_equivalent(g1: dict, g2: dict) -> bool:
    """Same skeleton and same v-structures: the graphs encode exactly the
    same conditional independences, so no data can tell them apart."""
    return (skeleton(g1) == skeleton(g2)
            and v_structures(g1) == v_structures(g2))


def all_dags(nodes: list):
    """Every directed acyclic graph over the nodes (only for tiny n: there
    are 29,281 on five nodes)."""
    pairs = list(combinations(nodes, 2))
    # Each pair is unconnected (0), a -> b (1), or b -> a (2).
    for choice in product(range(3), repeat=len(pairs)):
        parents = {x: () for x in nodes}
        for (a, b), c in zip(pairs, choice):
            if c == 1:
                parents[b] += (a,)
            elif c == 2:
                parents[a] += (b,)
        if is_acyclic(parents):
            yield parents
