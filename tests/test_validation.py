"""Tests for adm/validation.py and the collision avoidance problem (Ch 14)."""

import numpy as np
import pytest

from adm.problems import collision_avoidance as ca
from adm.validation import beta_interval, importance_estimate, mc_estimate, pareto_front

M = ca.Model()


@pytest.fixture(scope="module")
def optimal():
    return ca.optimal_policy(M)[0]


def changes(a_prev, a):
    return (a != a_prev).astype(float)


def zeros(model=M):
    return np.zeros((2 * model.h_max + 1, 2 * model.hdot_max + 1))


def test_mc_estimate_rare_event():
    # 10 hits in a million trials: relative standard error about 1/sqrt(10).
    x = np.zeros(1_000_000)
    x[:10] = 1
    mean, se, rel, (lo, hi) = mc_estimate(x)
    assert mean == pytest.approx(1e-5)
    assert rel == pytest.approx(0.316, abs=1e-3)
    assert 0 < lo < mean < hi
    # With 3 hits in 10,000 the relative error is about 0.58 > 1/1.96, so
    # the normal interval dips below 0; the Beta interval never does.
    x = np.zeros(10_000)
    x[:3] = 1
    lo = mc_estimate(x)[3][0]
    b_lo, b_hi = beta_interval(3, 10_000)
    assert lo < 0 < b_lo < 3e-4 < b_hi


def test_importance_estimate_optimal_proposal_has_zero_variance():
    # E[f(x)] for x ~ U(0, 4), f = 3 on [1, 2) and 0 elsewhere: exactly 0.75.
    # The ideal proposal (eq. 14.13) is U(1, 2); every weighted sample is 0.75.
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 2, 1000)
    f = np.where((x >= 1) & (x < 2), 3.0, 0.0)
    log_w = np.full(1000, np.log(0.25 / 1.0))   # p(x) / p'(x)
    mean, se, ess = importance_estimate(f, log_w)
    assert mean == pytest.approx(0.75) and se == pytest.approx(0, abs=1e-12)
    assert ess == pytest.approx(1000)


def test_pareto_front():
    pts = np.array([[1, 5], [2, 2], [3, 3], [5, 1], [2, 2.5]])
    assert pareto_front(pts).tolist() == [True, True, False, True, False]


def test_accel_reaches_rate_within_limit():
    hdot = np.array([-7, -5, -4, 0, 5, 9])
    assert ca.accel(M, hdot, 1).tolist() == [1, 0, -1, -1, -1, -1]   # descend at 5
    assert ca.accel(M, hdot, 2).tolist() == [1, 1, 1, 1, 0, -1]      # climb at 5
    assert ca.accel(M, hdot, 0).tolist() == [0] * 6


def test_lattice_is_large_enough(optimal):
    # A much larger lattice gives the same answers: clamping is harmless.
    big = ca.Model(h_max=800, hdot_max=50)
    for pol, pol_big in ((optimal, ca.optimal_policy(big)[0]),
                         (ca.simple_policy(M), ca.simple_policy(big))):
        assert ca.start_average(M, ca.evaluate(M, pol)) == pytest.approx(
            ca.start_average(big, ca.evaluate(big, pol_big)), rel=1e-9)


def test_optimal_cost_is_collision_plus_changes(optimal):
    # The optimal cost-to-go equals 1 x P(collision) + 0.01 x E[changes].
    cost = ca.optimal_policy(M)[1]
    pc = ca.start_average(M, ca.evaluate(M, optimal))
    ch = ca.start_average(M, ca.evaluate(M, optimal, changes, zeros()))
    assert ca.start_average(M, cost) == pytest.approx(pc + 0.01 * ch, rel=1e-12)
    # And the optimal policy beats the simple one on both metrics.
    simple = ca.simple_policy(M)
    assert pc < ca.start_average(M, ca.evaluate(M, simple))
    assert ch < ca.start_average(M, ca.evaluate(M, simple, changes, zeros()))


def test_monte_carlo_matches_exact():
    simple = ca.simple_policy(M)
    exact = ca.start_average(M, ca.evaluate(M, simple))
    rng = np.random.default_rng(1)
    hf, ch, lw = ca.simulate(M, simple, *ca.sample_starts(100_000, rng), rng)
    mean, se, _, _ = mc_estimate(np.abs(hf) < 50)
    assert abs(mean - exact) < 4 * se
    assert np.all(lw == 0)


@pytest.mark.parametrize("kw", [{"tilt": 0.25}, {"guide": "exact"}, {"guide": "noise3"}])
def test_importance_sampling_is_unbiased(optimal, kw):
    exact = ca.start_average(M, ca.evaluate(M, optimal))
    if "guide" in kw:
        model = M if kw["guide"] == "exact" else ca.Model(noise=3)
        kw = {"guide": ca.evaluate(model, optimal, history=True)}
    rng = np.random.default_rng(2)
    hf, _, lw = ca.simulate(M, optimal, *ca.sample_starts(50_000, rng), rng, **kw)
    mean, se, _ = importance_estimate(np.abs(hf) < 50, lw)
    assert abs(mean - exact) < 4 * se
    assert se < exact                         # far better than direct sampling


def test_exact_guide_gives_start_risk(optimal):
    # With the exact risk-to-go, each collision's weight telescopes to the
    # start state's risk, and every guided episode collides.
    G = ca.evaluate(M, optimal, history=True)
    rng = np.random.default_rng(3)
    h0, d0 = ca.sample_starts(200, rng)
    hf, _, lw = ca.simulate(M, optimal, h0, d0, rng, guide=G, eps=0)
    risk = G[-1][0][ca._index(M, h0, d0)]
    assert np.all(np.abs(hf) < 50)
    assert np.allclose(np.exp(lw), risk, rtol=1e-9)


def test_robust_policy_protects_the_worst_model(optimal):
    worse = ca.Model(compliance=0.8)
    robust = ca.optimal_policy(M, models=[M, worse])[0]
    for model in (M, worse):
        assert (ca.start_average(model, ca.evaluate(model, robust))
                < ca.start_average(model, ca.evaluate(model, optimal)))


def test_most_likely_failure_replays(optimal):
    W, choice = ca.adversary(M, optimal)
    i, j = 180 + M.h_max, 5 + M.hdot_max
    steps = ca.replay(M, optimal, choice, 180, 5)
    probs = dict(M.noise_outcomes())
    logp = sum(np.log(probs[nu]) for *_, nu in steps[:-1])
    assert logp == pytest.approx(W[0, i, j])
    assert abs(steps[-1][1]) < 50             # the path ends in a collision
    # The optimal policy needs at least three unlikely pushes from any start.
    pushes = sum(nu != 0 for *_, nu in steps[:-1])
    assert pushes >= 3


def test_adversary_trades_likelihood(optimal):
    # Small lambda: the adversary forces a collision; large: it gives up.
    for lam, crash in ((0.01, True), (1.0, False)):
        _, choice = ca.adversary(M, optimal, lam=lam)
        final_h = ca.replay(M, optimal, choice, 60, 0)[-1][1]
        assert (abs(final_h) < 50) == crash
