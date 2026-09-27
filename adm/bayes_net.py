"""Discrete Bayesian networks (ADM Ch. 2): the chain rule and d-separation.

A network is three dicts keyed by variable name:
  card[X]    number of values X can take (values are 0, 1, ..., card[X] - 1)
  parents[X] tuple of X's parents
  cpt[X]     array of shape (card of each parent..., card[X]): P(X | parents)
Lines stay under 80 characters because the Ch 2 page prints this source.
"""

from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np


@dataclass(frozen=True)
class BayesNet:
    card: dict
    parents: dict
    cpt: dict


def probability(bn: BayesNet, assignment: dict) -> float:
    """P(assignment) by the chain rule: one CPT entry per variable."""
    p = 1.0
    for x, pa in bn.parents.items():
        # CPT axes are (parents..., x), so index with the parents' values
        # first and x's own value last.
        index = tuple(assignment[q] for q in pa) + (assignment[x],)
        p *= bn.cpt[x][index]
    return p


def joint_table(bn: BayesNet) -> np.ndarray:
    """The full joint distribution as an array, axes in `bn.parents` order."""
    names = list(bn.parents)
    table = np.zeros([bn.card[x] for x in names])
    for values in product(*(range(bn.card[x]) for x in names)):
        table[values] = probability(bn, dict(zip(names, values)))
    return table


def num_parameters(bn: BayesNet) -> int:
    """Independent numbers in all the CPTs (each row loses one: it sums to 1)."""
    return sum(
        prod(bn.card[q] for q in pa) * (bn.card[x] - 1)
        for x, pa in bn.parents.items()
    )


def active_trail(parents: dict, x: str, y: str, given=()) -> list | None:
    """An active trail from x to y given the observed set, or None if none.

    Searches (node, direction) pairs: "up" means we arrived from a child,
    "down" means we arrived from a parent. A trail through a node is blocked
    if the node is observed, except at a v-structure (collider), which is
    active only if the collider or one of its descendants is observed.
    """
    given = set(given)
    children = {v: [c for c in parents if v in parents[c]] for v in parents}
    # A collider is active if it or a descendant is observed, i.e. if it is
    # an ancestor of (or in) the observed set.
    active_colliders, frontier = set(), list(given)
    while frontier:
        v = frontier.pop()
        if v not in active_colliders:
            active_colliders.add(v)
            frontier.extend(parents[v])
    came_from = {(x, "up"): None}
    queue = [(x, "up")]
    while queue:
        node, direction = state = queue.pop(0)
        if node == y and node not in given:
            trail = []
            while state is not None:
                trail.append(state[0])
                state = came_from[state]
            return trail[::-1]
        moves = []
        if direction == "up" and node not in given:
            moves += [(p, "up") for p in parents[node]]
            moves += [(c, "down") for c in children[node]]
        elif direction == "down":
            if node not in given:
                moves += [(c, "down") for c in children[node]]
            if node in active_colliders:
                moves += [(p, "up") for p in parents[node]]
        for move in moves:
            if move not in came_from:
                came_from[move] = state
                queue.append(move)
    return None


def d_separated(parents: dict, x: str, y: str, given=()) -> bool:
    """True if x and y are conditionally independent given `given`."""
    return active_trail(parents, x, y, given) is None
