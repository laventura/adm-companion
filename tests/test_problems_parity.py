"""Python and Julia must define the same problems (CLAUDE.md §8, Phase 2)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from adm.problems.coffee import COFFEE
from adm.problems.commute import COMMUTE, COMMUTE_TIMES_COV, COMMUTE_TIMES_MEAN

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


@pytest.mark.parametrize("name, bn", [("commute", COMMUTE), ("coffee", COFFEE)])
def test_bayes_nets_match(julia_problems, name, bn):
    theirs = julia_problems[name]
    assert set(theirs) == set(bn.cpt)
    for x, table in bn.cpt.items():
        assert np.allclose(theirs[x], table.ravel()), x


def test_commute_times_match(julia_problems):
    ours = np.concatenate([COMMUTE_TIMES_MEAN, COMMUTE_TIMES_COV.ravel()])
    assert np.allclose(julia_problems["commute_times"], ours)
