"""Coffee machine diagnosis: a naive Bayes model (original to this site, Ch 3).

One hidden class, the machine's condition C, and three symptoms that each
depend only on C:  C ──► slow flow (F), C ──► loud noise (N), C ──► weak taste (W).
"""

import numpy as np

from adm.bayes_net import BayesNet

CLASSES = ("ok", "scale buildup", "worn pump")
SYMPTOMS = {"F": "Slow flow", "N": "Loud noise", "W": "Weak taste"}

PRIOR = np.array([0.80, 0.15, 0.05])
# P(symptom present | class), one entry per class in CLASSES order.
P_SYMPTOM = {
    "F": np.array([0.05, 0.70, 0.60]),
    "N": np.array([0.10, 0.20, 0.80]),
    "W": np.array([0.10, 0.60, 0.30]),
}
# likelihoods[i][c, o] = P(symptom i = o | class c), with o = 0 absent, 1 present.
LIKELIHOODS = [np.stack([1 - p, p], axis=-1) for p in P_SYMPTOM.values()]

COFFEE = BayesNet(
    card={"C": 3, "F": 2, "N": 2, "W": 2},
    parents={"C": (), "F": ("C",), "N": ("C",), "W": ("C",)},
    cpt={"C": PRIOR, **{s: L for s, L in zip(P_SYMPTOM, LIKELIHOODS)}},
)
