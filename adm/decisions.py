"""Simple decisions (ADM Ch. 6): utility, expected utility, decision
networks, and the value of information.

A decision network is stored as a BayesNet in which the decision is a
variable with no parents (its table is a placeholder, never used), plus a
utility table U whose axes are (decision, outcome variables...).
Lines stay under 80 characters because the Ch 6 page prints this source.
"""

import numpy as np

from adm.inference import exact_inference


def expected_utility(probs, utilities) -> float:
    """Utility of a lottery [S_1: p_1; ...; S_n: p_n] = sum_i p_i U(S_i)."""
    return float(np.dot(probs, utilities))


def exponential_utility(x, lam: float):
    """U(x) = 1 - exp(-lam x). Concave (risk averse) when lam > 0; the
    bigger lam, the more strongly sure things are preferred to gambles."""
    return 1 - np.exp(-lam * np.asarray(x, dtype=float))


def certainty_equivalent(outcomes, probs, lam: float) -> float:
    """The sure amount whose utility equals the lottery's expected utility,
    for exponential utility: solve U(ce) = EU by inverting U."""
    eu = expected_utility(probs, exponential_utility(outcomes, lam))
    return float(-np.log(1 - eu) / lam)       # U^-1(u) = -ln(1 - u) / lam


def action_utilities(bn, decision: str, outcomes: tuple, U, evidence: dict):
    """EU(a | evidence) for every action a:
    sum over outcome values of P(outcome | a, evidence) * U[a, outcome].

    The decision has no parents, so fixing it to a is just one more piece
    of evidence for inference (Ch 3).
    """
    return np.array([
        np.sum(exact_inference(bn, outcomes, {**evidence, decision: a}).table
               * U[a])
        for a in range(bn.card[decision])
    ])


def best_action(bn, decision: str, outcomes: tuple, U, evidence: dict):
    """The maximum-expected-utility action and its expected utility."""
    eu = action_utilities(bn, decision, outcomes, U, evidence)
    a = int(np.argmax(eu))
    return a, float(eu[a])


def value_of_information(bn, decision: str, outcomes: tuple, U,
                         evidence: dict, observe: str) -> float:
    """VOI of observing `observe` before deciding: the expected best EU
    after seeing its value, minus the best EU without it.

    `observe` must not depend on the decision (you see it before acting),
    so the decision's placeholder table never affects P(o | evidence).
    """
    _, now = best_action(bn, decision, outcomes, U, evidence)
    p = exact_inference(bn, (observe,), evidence).table    # P(o | evidence)
    later = 0.0
    for o in range(bn.card[observe]):
        if p[o] > 0:                           # skip values that can't occur
            seen = {**evidence, observe: o}
            later += p[o] * best_action(bn, decision, outcomes, U, seen)[1]
    return float(later - now)
