"""Morning commute: a five-variable Bayesian network (original to this site, Ch 2).

  Storm (S) ──► Traffic jam (J) ◄── Roadwork (R)
    │                │
    ▼                ▼
  Radio alert (A)  Late (L)

Every variable is binary: 0 = no, 1 = yes.
"""

import numpy as np

from adm.bayes_net import BayesNet

PARENTS = {"S": (), "R": (), "J": ("S", "R"), "L": ("J",), "A": ("S",)}
NAMES = {"S": "Storm", "R": "Roadwork", "J": "Traffic jam", "L": "Late", "A": "Radio alert"}


def _binary(p_yes):
    """CPT with last axis (no, yes), from an array of P(yes)."""
    p_yes = np.asarray(p_yes, dtype=float)
    return np.stack([1 - p_yes, p_yes], axis=-1)


COMMUTE = BayesNet(
    card={x: 2 for x in PARENTS},
    parents=PARENTS,
    cpt={
        "S": _binary(0.1),
        "R": _binary(0.2),
        "J": _binary([[0.1, 0.6],      # S = 0: R = 0, R = 1
                      [0.7, 0.9]]),    # S = 1: R = 0, R = 1
        "L": _binary([0.05, 0.5]),     # J = 0, J = 1
        "A": _binary([0.02, 0.95]),    # S = 0, S = 1
    },
)

# Commute times in minutes as a joint Gaussian (Ch 3): index 0 = morning,
# index 1 = evening. Standard deviations 8 and 10, correlation 0.6, so the
# covariance is 0.6 * 8 * 10 = 48.
COMMUTE_TIMES_MEAN = np.array([35.0, 40.0])
COMMUTE_TIMES_COV = np.array([[64.0, 48.0], [48.0, 100.0]])
