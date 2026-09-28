"""Policy search (ADM Ch. 10): improve a parameterized policy directly,
without a value function.

Every method maximizes an objective U(theta): the expected return of the
policy with parameters theta (a vector). U is usually a Monte Carlo
estimate, so two calls with the same theta can disagree. Methods that
use randomness take a NumPy generator `rng`, and all accept an optional
`trace` list that records their progress, one entry per iteration.

Lines stay under 80 characters because the Ch 10 page prints this source.
"""

import numpy as np

Array = np.ndarray


def hooke_jeeves(U, theta, alpha: float, c: float = 0.5,
                 eps: float = 0.01, trace: list | None = None) -> Array:
    """Local search: try a step of +-alpha along each coordinate, move to
    the best improvement, and shrink alpha by c when nothing improves.
    Stop when alpha drops below eps."""
    theta = np.array(theta, dtype=float)
    u = U(theta)
    while alpha > eps:
        best_i, best_sign, best_u = None, 0, u
        for i in range(len(theta)):          # the 2n neighbors of theta
            for sign in (-1, 1):
                neighbor = theta.copy()
                neighbor[i] += sign * alpha
                u_new = U(neighbor)
                if u_new > best_u:
                    best_i, best_sign, best_u = i, sign, u_new
        if best_i is None:
            alpha *= c                       # no neighbor helped: look closer
        else:
            theta[best_i] += best_sign * alpha
            u = best_u
        if trace is not None:
            trace.append((theta.copy(), alpha))
    return theta


def genetic_search(U, population, sigma: float, m_elite: int, k_max: int,
                   rng, trace: list | None = None) -> Array:
    """Keep a population of parameter vectors (one per row). Each
    generation, the m_elite best become parents: every child is a random
    parent plus Gaussian noise, and the best parent survives unchanged."""
    thetas = np.array(population, dtype=float)
    m, n = thetas.shape
    for _ in range(k_max):
        u = np.array([U(t) for t in thetas])
        order = np.argsort(-u)               # best first
        elite = thetas[order[:m_elite]]
        if trace is not None:
            trace.append((thetas.copy(), order[:m_elite]))
        parents = elite[rng.integers(m_elite, size=m - 1)]
        children = parents + sigma * rng.standard_normal((m - 1, n))
        thetas = np.vstack([children, elite[0]])
    return thetas[-1]                        # the last row is the best


def cross_entropy(U, mean, cov, m: int, m_elite: int, k_max: int, rng,
                  trace: list | None = None) -> tuple[Array, Array]:
    """Sample m parameter vectors from a Gaussian, keep the m_elite best,
    and refit the Gaussian to them by maximum likelihood."""
    mean, cov = np.array(mean, dtype=float), np.array(cov, dtype=float)
    for _ in range(k_max):
        thetas = rng.multivariate_normal(mean, cov, size=m)
        u = np.array([U(t) for t in thetas])
        elite = thetas[np.argsort(-u)[:m_elite]]
        if trace is not None:
            trace.append((mean.copy(), cov.copy(), thetas, elite))
        mean = elite.mean(axis=0)
        # Maximum likelihood covariance (divide by m_elite, not m_elite - 1).
        cov = np.cov(elite, rowvar=False, bias=True)
    return mean, cov


def rank_weights(m: int) -> Array:
    """Rank shaping: weight w_i for the i-th best of m samples. The best
    get positive weight, the rest a small negative one; weights sum to 0."""
    i = np.arange(1, m + 1)
    w = np.maximum(0, np.log(m / 2 + 1) - np.log(i))
    return w / w.sum() - 1 / m


def evolution_strategies(U, mean, log_sd, m: int, alpha: float, k_max: int,
                         rng, trace: list | None = None):
    """Ascend the expected utility of the search distribution
    N(mean, diag(sd^2)), with psi = (mean, log sd). Each sample
    theta = mean + sd * eps has search gradient
      d/d mean   log p = eps / sd,      d/d log sd  log p = eps^2 - 1,
    weighted by its rank. The steps are scaled by the inverse Fisher
    information (sd^2 and 1/2): the natural gradient. Then a step moves
    the mean by a multiple of sd, so it can't explode as sd shrinks."""
    mean = np.array(mean, dtype=float)
    log_sd = np.array(log_sd, dtype=float)
    w = rank_weights(m)
    for _ in range(k_max):
        sd = np.exp(log_sd)
        eps = rng.standard_normal((m, len(mean)))
        thetas = mean + sd * eps
        u = np.array([U(t) for t in thetas])
        e = eps[np.argsort(-u)]              # noise of the samples, best first
        if trace is not None:
            trace.append((mean.copy(), sd.copy(), thetas))
        grad_mean = (w[:, None] * e / sd).sum(axis=0)
        grad_log_sd = (w[:, None] * (e**2 - 1)).sum(axis=0)
        mean = mean + alpha * sd**2 * grad_mean     # = alpha * sd * sum w e
        log_sd = log_sd + alpha * grad_log_sd / 2
    return mean, np.exp(log_sd)


def isotropic_evolution_strategies(U, mean, sigma: float, m: int,
                                   alpha: float, k_max: int, rng,
                                   mirrored: bool = True,
                                   trace: list | None = None) -> Array:
    """Evolution strategies with a fixed spherical search distribution
    N(mean, sigma^2 I). With mirrored sampling, every noise vector eps is
    paired with -eps, which cancels much of the noise in the gradient."""
    mean = np.array(mean, dtype=float)
    half = m // 2
    w = rank_weights(2 * half)
    for _ in range(k_max):
        eps = rng.standard_normal((half, len(mean)))
        if mirrored:
            eps = np.vstack([eps, -eps])
        else:
            eps = np.vstack([eps, rng.standard_normal((half, len(mean)))])
        u = np.array([U(mean + sigma * e) for e in eps])
        if trace is not None:
            trace.append(mean.copy())
        # grad log p(theta | mean) = eps / sigma for theta = mean + sigma eps
        grad = (w[:, None] * eps[np.argsort(-u)]).sum(axis=0) / sigma
        mean = mean + alpha * grad
    return mean
