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


def sample_trajectories(theta, m: int, rng, d: int = HORIZON):
    """m episodes of d steps under the Gaussian policy, as arrays of shape
    (m, d): states, actions and rewards (row i is episode i, column k is
    step k + 1)."""
    s = START_MEAN + START_SD * rng.standard_normal(m)
    S, A, R = np.zeros((m, d)), np.zeros((m, d)), np.zeros((m, d))
    for k in range(d):
        a = gaussian_policy(theta, s, rng)
        S[:, k], A[:, k] = s, a
        s, R[:, k] = step(s, a, rng)
    return S, A, R


def log_policy(theta, s, a):
    """log N(a | theta_1 s, sd^2), sd = |theta_2| + 1e-5, elementwise."""
    sd = abs(theta[1]) + 1e-5
    z = (np.asarray(a) - theta[0] * np.asarray(s)) / sd
    return -0.5 * z**2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


def policy_kl(theta, theta2, s):
    """KL divergence from the policy theta to theta2 at states s:
    KL(N(m1, v1) || N(m2, v2))
      = log(sd2 / sd1) + (v1 + (m1 - m2)^2) / (2 v2) - 1/2."""
    s = np.asarray(s, dtype=float)
    sd1, sd2 = abs(theta[1]) + 1e-5, abs(theta2[1]) + 1e-5
    dm = (theta[0] - theta2[0]) * s          # difference of the mean actions
    return np.log(sd2 / sd1) + (sd1**2 + dm**2) / (2 * sd2**2) - 0.5


def grad_log_policy(theta, s, a):
    """Gradient of log N(a | theta_1 s, sd^2) with sd = |theta_2| + 1e-5,
    for arrays s and a of the same shape. Returns shape s.shape + (2,):
      d/d theta_1 = (a - theta_1 s) s / sd^2
      d/d theta_2 = ((a - theta_1 s)^2 - sd^2) / sd^3 * sign(theta_2)"""
    sd = abs(theta[1]) + 1e-5
    z = a - theta[0] * s                     # deviation from the mean action
    g1 = z * s / sd**2
    g2 = (z**2 - sd**2) / sd**3 * np.sign(theta[1])
    return np.stack([g1, g2], axis=-1)


def exact_gradient(theta, d: int = HORIZON) -> np.ndarray:
    """Gradient of exact_value, by differentiating its recursion:
    ms' = gain^2 ms + noise_var, so
    d ms'/d theta_1 = 2 gain ms + gain^2 d ms/d theta_1,
    d ms'/d theta_2 = gain^2 d ms/d theta_2 + 2 sd sign(theta_2)."""
    gain, sd = 1 + theta[0], abs(theta[1]) + 1e-5
    noise_var = sd**2 + NOISE_SD**2
    ms, dms = START_MEAN**2 + START_SD**2, np.zeros(2)
    grad = np.zeros(2)
    for t in range(d):
        grad -= GAMMA**t * dms              # U = -sum_t gamma^t E[s_t^2]
        dms = np.array([2 * gain * ms + gain**2 * dms[0],
                        gain**2 * dms[1] + 2 * sd * np.sign(theta[1])])
        ms = gain**2 * ms + noise_var
    return grad


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
