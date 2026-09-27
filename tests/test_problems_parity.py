"""Python and Julia must define the same problems (CLAUDE.md §8, Phase 2)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from adm.problems import mountain_car
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


def test_commute_times_match(julia_problems):
    ours = np.concatenate([COMMUTE_TIMES_MEAN, COMMUTE_TIMES_COV.ravel()])
    assert np.allclose(julia_problems["commute_times"], ours)
