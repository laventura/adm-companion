"""Ch. 1 helpers: two of the five ways to design a decision maker (ADM §1.3)
on Hex World. Planning, supervised learning and reinforcement learning use
the code of later chapters (adm.mdp, adm.imitation, adm.model_free).
"""

import numpy as np

from adm.problems.hex_world import DIRECTIONS, hex_steps, neighbor

Array = np.ndarray


def head_for(tiles, target) -> Array:
    """Explicit programming: a hand-written rule. On every tile, pick the
    direction whose neighbor is fewest moves from `target` (ignoring that
    moves can veer). Reward tiles and the terminal state get action 0."""
    tiles = list(tiles)
    steps = hex_steps(tiles, target)
    policy = np.zeros(len(tiles) + 1, dtype=int)       # last entry: terminal
    for i, tile in enumerate(tiles):
        best = min(range(len(DIRECTIONS)),
                   key=lambda a: steps.get(neighbor(tile, DIRECTIONS[a]),
                                           np.inf))
        policy[i] = best
    return policy


def simulated_return(mdp, policy, b, seed, n=100, depth=40) -> float:
    """Average discounted return of a deterministic policy over n simulated
    episodes from b. The seed is fixed, so two policies are compared on the
    same random draws (common random numbers)."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n):
        s = rng.choice(len(b), p=b)
        for t in range(depth):
            a = policy[s]
            total += mdp.gamma ** t * mdp.R[s, a]
            s = rng.choice(len(b), p=mdp.T[s, a])
    return total / n


def local_search(mdp, b, rng, proposals=300, n=100, depth=40, seed=0):
    """Optimization: start from a random policy; repeatedly change one
    state's action at random and keep the change if the simulated return
    improves. It uses the simulator only as a black box, never T or R
    directly. Returns the policy and the simulated return after each
    proposal."""
    n_s, n_a = mdp.R.shape
    policy = rng.integers(n_a, size=n_s)
    best = simulated_return(mdp, policy, b, seed, n, depth)
    history = [best]
    for _ in range(proposals):
        trial = policy.copy()
        s = rng.integers(n_s)
        trial[s] = rng.integers(n_a)
        u = simulated_return(mdp, trial, b, seed, n, depth)
        if u > best:
            policy, best = trial, u
        history.append(best)
    return policy, history
