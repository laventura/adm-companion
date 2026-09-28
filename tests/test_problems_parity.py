"""Python and Julia must define the same problems (CLAUDE.md §8, Phase 2)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from adm.problems import collision_avoidance, crying_baby, fork, mountain_car, park, simple_regulator, trail
from adm.problems.coffee import COFFEE
from adm.problems.commute import (
    COMMUTE, COMMUTE_DECISION, COMMUTE_TIMES_COV, COMMUTE_TIMES_MEAN, COMMUTE_UTILITY,
)

ROOT = Path(__file__).resolve().parent.parent

# Julia flattens column-major; permuting the axes in reverse first gives the
# same order as NumPy's row-major ravel().
JULIA = """
using ADM, JSON
rowmajor(A) = vec(permutedims(A, reverse(1:ndims(A))))
cpts(bn) = Dict(string(x) => rowmajor(bn.cpt[x]) for x in keys(bn.cpt))
print(JSON.json(Dict(
    "commute" => cpts(commute_network()),
    "coffee" => cpts(coffee_network()),
    "commute_times" => [COMMUTE_TIMES_MEAN; rowmajor(COMMUTE_TIMES_COV)],
    "commute_decision" => cpts(commute_decision_network()),
    "commute_utility" => rowmajor(COMMUTE_UTILITY),
    # `let` makes s and path local, avoiding Julia's soft-scope rule for loops.
    "mountain_car" => let s = [-0.5, 0.0], path = Float64[]
        # A fixed action sequence from the valley floor; record every state.
        for t in 1:150
            s = mountain_car_step(s, (-1, 0, 1)[mod1(t ÷ 20, 3)])
            append!(path, s)
        end
        path
    end,
    "regulator_exact" => [regulator_exact_value(θ) for θ in ([0.0, 1.0], [-0.5, 0.3], [-1.0, 0.0])],
    "regulator_gradient" => [regulator_exact_gradient(θ) for θ in ([0.0, 1.0], [-0.5, 0.3], [-1.0, 0.2])],
    "park" => Dict(layout => let (p, β) = park(layout)
            Dict("T" => rowmajor(p.T), "R" => rowmajor(p.R), "beta" => rowmajor(β),
                 "b" => park_left_starts(layout))
        end for layout in ("A", "B")),
    "fork" => let p = fork()
        Dict("T" => rowmajor(p.T), "R" => rowmajor(p.R), "O" => rowmajor(p.O))
    end,
    "crying_baby" => let p = crying_baby()
        Dict("T" => rowmajor(p.T), "R" => rowmajor(p.R), "O" => rowmajor(p.O), "gamma" => p.γ)
    end,
    "trail" => let p = trail()
        Dict("T" => rowmajor(p.T), "R" => rowmajor(p.R), "gamma" => p.γ)
    end,
    "collision" => let m = CAModel(), (pol, U) = ca_optimal_policy(m), simple = ca_simple_policy(m)
        starts(W) = maximum(W[1, (-200:200) .+ 501, (-10:10) .+ 31])
        robust = ca_optimal_policy(m; models = [m, CAModel(compliance = 0.8)])[2]
        Dict("values" => [
                ca_start_average(m, U),
                ca_start_average(m, ca_evaluate(m, pol)),
                ca_start_average(m, ca_evaluate(m, simple)),
                ca_start_average(m, ca_evaluate(m, pol; step_metric = (p, a) -> p != a,
                                                 final = zeros(1001, 61))),
                ca_start_average(m, robust),
                starts(ca_most_likely_failure(m, pol))],
             # Julia advisories are 1-based; Python subtracts 1 before comparing.
             "policy" => [rowmajor(pol[t][1, :, :]) for t in (10, 30)])
    end,
)))
"""


@pytest.fixture(scope="module")
def julia_problems():
    if shutil.which("julia") is None:
        pytest.skip("julia not installed")
    out = subprocess.run(
        ["julia", f"--project={ROOT / 'julia'}", "--startup-file=no", "-e", JULIA],
        capture_output=True, text=True, check=True, timeout=600,
    ).stdout
    return json.loads(out)


@pytest.mark.parametrize("name, bn", [("commute", COMMUTE), ("coffee", COFFEE),
                                      ("commute_decision", COMMUTE_DECISION)])
def test_bayes_nets_match(julia_problems, name, bn):
    theirs = julia_problems[name]
    assert set(theirs) == set(bn.cpt)
    for x, table in bn.cpt.items():
        assert np.allclose(theirs[x], table.ravel()), x


def test_commute_utility_matches(julia_problems):
    assert np.allclose(julia_problems["commute_utility"], COMMUTE_UTILITY.ravel())


def test_mountain_car_matches(julia_problems):
    s, path = np.array([-0.5, 0.0]), []
    for t in range(1, 151):
        # Same sequence as the Julia side: mod1(t ÷ 20, 3) picks action 1..3.
        s = mountain_car.step(s, (-1, 0, 1)[(t // 20 - 1) % 3])
        path.extend(s)
    assert np.allclose(julia_problems["mountain_car"], path, rtol=0, atol=1e-12)


def test_regulator_exact_values_match(julia_problems):
    thetas = ([0.0, 1.0], [-0.5, 0.3], [-1.0, 0.0])
    ours = [simple_regulator.exact_value(t) for t in thetas]
    assert np.allclose(julia_problems["regulator_exact"], ours, rtol=1e-12)
    grads = [simple_regulator.exact_gradient(np.array(t))
             for t in ([0.0, 1.0], [-0.5, 0.3], [-1.0, 0.2])]
    assert np.allclose(julia_problems["regulator_gradient"], grads, rtol=1e-12)


def test_commute_times_match(julia_problems):
    ours = np.concatenate([COMMUTE_TIMES_MEAN, COMMUTE_TIMES_COV.ravel()])
    assert np.allclose(julia_problems["commute_times"], ours)


def test_collision_avoidance_matches(julia_problems):
    theirs = julia_problems["collision"]
    m = collision_avoidance.Model()
    policy, cost = collision_avoidance.optimal_policy(m)
    simple = collision_avoidance.simple_policy(m)
    avg = lambda V: collision_avoidance.start_average(m, V)  # noqa: E731
    changes = collision_avoidance.evaluate(
        m, policy, lambda p, a: (a != p).astype(float), np.zeros((1001, 61)))
    robust = collision_avoidance.optimal_policy(
        m, models=[m, collision_avoidance.Model(compliance=0.8)])[1]
    W = collision_avoidance.adversary(m, policy)[0]
    ours = [avg(cost), avg(collision_avoidance.evaluate(m, policy)),
            avg(collision_avoidance.evaluate(m, simple)), avg(changes), avg(robust),
            W[0, 300:701, 20:41].max()]
    assert np.allclose(theirs["values"], ours, rtol=1e-12)
    for k, t in enumerate((10, 30)):
        assert np.array_equal(np.array(theirs["policy"][k]) - 1, policy[t][0].ravel())


def test_trail_matches(julia_problems):
    theirs, ours = julia_problems["trail"], trail.trail()
    assert np.allclose(theirs["T"], ours.T.ravel()) and np.allclose(theirs["R"], ours.R.ravel())
    assert theirs["gamma"] == ours.gamma


@pytest.mark.parametrize("layout", ["A", "B"])
def test_park_matches(julia_problems, layout):
    theirs = julia_problems["park"][layout]
    mdp, beta = park.park(layout)
    assert np.allclose(theirs["T"], mdp.T.ravel()) and np.allclose(theirs["R"], mdp.R.ravel())
    assert np.allclose(theirs["beta"], beta.ravel())
    assert np.allclose(theirs["b"], park.left_edge_starts(layout))


def test_crying_baby_matches(julia_problems):
    theirs, ours = julia_problems["crying_baby"], crying_baby.crying_baby()
    for key, arr in (("T", ours.T), ("R", ours.R), ("O", ours.O)):
        assert np.allclose(theirs[key], arr.ravel()), key
    assert theirs["gamma"] == ours.gamma


def test_fork_matches(julia_problems):
    theirs, ours = julia_problems["fork"], fork.fork()
    for key, arr in (("T", ours.T), ("R", ours.R), ("O", ours.O)):
        assert np.allclose(theirs[key], arr.ravel()), key
