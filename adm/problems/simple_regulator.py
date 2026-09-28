"""Simple regulator (ADM Appendix F.5), implemented from its description.

A single real-valued state s should be kept near 0. The action a is added
to it, plus noise: s' ~ N(s + a, 0.1^2). The reward is -s^2 whatever the
action. Episodes start from s ~ N(0.3, 0.1^2). The discount (0.9) and the
10-step horizon are this site's choices. Mirrors
julia/src/problems/simple_regulator.jl.
"""

import numpy as np

NOISE_SD = 0.1                 # standard deviation of the transition noise
START_MEAN, START_SD = 0.3, 0.1
GAMMA, HORIZON = 0.9, 10


def step(s, a, rng):
    """Sample the next state; the reward -s^2 depends on the current one."""
    s = np.asarray(s, dtype=float)
    return s + a + NOISE_SD * rng.standard_normal(s.shape), -s**2


def gaussian_policy(theta, s, rng):
    """a ~ N(theta_1 s, theta_2^2): a gain on the state plus action noise.
    |theta_2| + 1e-5 keeps the standard deviation positive."""
    s = np.asarray(s, dtype=float)
    sd = abs(theta[1]) + 1e-5
    return theta[0] * s + sd * rng.standard_normal(s.shape)


def rollout_returns(theta, m: int, rng, d: int = HORIZON):
    """Discounted returns of m independent d-step episodes, run side by
    side (one array entry per episode)."""
    s = START_MEAN + START_SD * rng.standard_normal(m)
    ret = np.zeros(m)
    for t in range(d):
        s, r = step(s, gaussian_policy(theta, s, rng), rng)
        ret += GAMMA**t * r
    return ret


def exact_value(theta, d: int = HORIZON) -> float:
    """U(theta) without sampling. With a linear Gaussian policy the next
    state is (1 + theta_1) s plus Gaussian noise of variance
    theta_2^2 + 0.1^2, so the mean square E[s^2] follows a recursion."""
    gain = 1 + theta[0]
    noise_var = (abs(theta[1]) + 1e-5)**2 + NOISE_SD**2
    ms = START_MEAN**2 + START_SD**2        # E[s_1^2]
    u = 0.0
    for t in range(d):
        u -= GAMMA**t * ms                  # expected reward -E[s_t^2]
        ms = gain**2 * ms + noise_var
    return u
