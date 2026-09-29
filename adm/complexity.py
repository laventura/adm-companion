"""Appendix C: brute-force satisfiability and counts of objects that grow
exponentially (conditional plans, joint actions).

A formula is a list of clauses; a clause is a tuple of nonzero integers,
where +i means variable i is true and -i means it is false (1-based, as in
the DIMACS format). Example: (1, -3, 4) is x1 or not x3 or x4.
"""

from itertools import product


def satisfies(clauses, x) -> bool:
    """Check an assignment in O(total literals): every clause needs one
    true literal. x[i - 1] is the value of variable i."""
    return all(any(x[abs(l) - 1] == (l > 0) for l in c) for c in clauses)


def brute_force_sat(clauses, n):
    """Try all 2^n assignments, in the order (F, ..., F), (F, ..., T), ...
    Returns (first satisfying assignment or None, assignments checked)."""
    for k, x in enumerate(product((False, True), repeat=n), start=1):
        if satisfies(clauses, x):
            return x, k
    return None, 2**n


def count_satisfying(clauses, n) -> int:
    return sum(satisfies(clauses, x)
               for x in product((False, True), repeat=n))


def conditional_plan_count(n_actions, n_obs, h) -> int:
    """Number of depth-h conditional plans (Ch. 20): one action per node
    of a tree with 1 + |O| + ... + |O|^(h-1) nodes."""
    nodes = sum(n_obs**d for d in range(h))
    return n_actions**nodes
