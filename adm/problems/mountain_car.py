"""Mountain car (ADM Appendix F.4), implemented from its description.

An underpowered car sits in a valley and must reach the top of the right
hill (position 0.6). Its engine is too weak to drive straight up, so it has
to rock back and forth to build momentum.

State s = (x, v): position x in [-1.2, 0.6] and speed v in [-0.07, 0.07].
Actions: accelerate left (-1), coast (0), accelerate right (+1).
Every step costs 1 (reward -1) until the car reaches x = 0.6.
Transitions are deterministic. Mirrors julia/src/problems/mountain_car.jl.
"""

import numpy as np

X_BOUNDS = (-1.2, 0.6)
V_BOUNDS = (-0.07, 0.07)
ACTIONS = (-1, 0, 1)              # accelerate left, coast, accelerate right
ACTION_NAMES = ("accelerate left", "coast", "accelerate right")
GOAL_X = 0.6


def step(s, a):
    """Next state after action a from state s = (x, v).

    Speed changes first (engine plus gravity; the hill's slope at x is
    proportional to cos(3x)), then position moves by the new speed.
    Both are clamped to the state bounds. Works on arrays of states too:
    s[..., 0] is x and s[..., 1] is v.
    """
    s = np.asarray(s, dtype=float)
    x, v = s[..., 0], s[..., 1]
    v2 = np.clip(v + 0.001 * a - 0.0025 * np.cos(3 * x), *V_BOUNDS)
    x2 = np.clip(x + v2, *X_BOUNDS)
    return np.stack([x2, v2], axis=-1)


def is_goal(s) -> np.ndarray:
    """True where the car has reached the top of the right hill."""
    return np.asarray(s, dtype=float)[..., 0] >= GOAL_X


def rollout(policy, s0, max_steps: int = 1000) -> np.ndarray:
    """States visited from s0 until the goal (or max_steps), following
    policy(s) -> action. Row t is the state at time t."""
    path = [np.asarray(s0, dtype=float)]
    while not is_goal(path[-1]) and len(path) <= max_steps:
        path.append(step(path[-1], policy(path[-1])))
    return np.array(path)


def steps_to_goal(policy, starts, max_steps: int = 1000) -> np.ndarray:
    """Steps each start state needs to reach the goal under a vectorized
    policy(states) -> actions; max_steps + 1 means it never got there."""
    s = np.array(starts, dtype=float)
    steps = np.zeros(len(s), dtype=int)
    for _ in range(max_steps + 1):
        running = ~is_goal(s)
        if not running.any():
            break
        # Only cars still on their way move and pay for the step.
        s[running] = step(s[running], policy(s[running]))
        steps[running] += 1
    return steps
