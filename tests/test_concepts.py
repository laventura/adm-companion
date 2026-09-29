"""Tests for adm.concepts (Appendix A)."""

import numpy as np

from adm import concepts as C
from adm.problems.hex_world import hex_world


def test_information_measures():
    assert C.entropy([0.5, 0.5]) == 1.0 and C.entropy([1.0, 0.0]) == 0.0
    p, q = [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]
    assert np.isclose(C.cross_entropy(p, q), C.entropy(p) + C.kl_divergence(p, q))
    assert C.kl_divergence(p, p) == 0.0 and C.kl_divergence(p, q) > 0
    assert not np.isclose(C.kl_divergence(p, q), C.kl_divergence(q, p))
    assert C.cross_entropy([0.5, 0.5], [1.0, 0.0]) == np.inf


def test_gradient_ascent_step_size_limit():
    grad = lambda x: np.array([-2 * (x[0] - 1), -4 * (x[1] + 0.5)])   # curvature 2 and 4
    ok = C.gradient_ascent(grad, [-2, 2], 0.45, 60)[-1]
    bad = C.gradient_ascent(grad, [-2, 2], 0.55, 60)[-1]
    assert np.allclose(ok, [1, -0.5], atol=1e-3) and abs(bad[1]) > 100   # 2/4 = 0.5 is the limit


def test_bellman_backup_is_a_gamma_contraction():
    H, rng = hex_world(), np.random.default_rng(0)
    for _ in range(200):
        U, V = rng.normal(0, 10, len(H.R)), rng.normal(0, 10, len(H.R))
        gap = np.abs(C.bellman_backup(H.T, H.R, H.gamma, U) - C.bellman_backup(H.T, H.R, H.gamma, V)).max()
        assert gap <= H.gamma * np.abs(U - V).max() + 1e-12
