import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from adm.mdp import greedy, value_iteration
from adm.problems.hex_world import DIRECTIONS, hex_world, neighbor, straight_line_hex_world

ROOT = Path(__file__).resolve().parent.parent


def test_structure():
    mdp = hex_world()
    assert mdp.T.shape == (25, 6, 25) and mdp.R.shape == (25, 6)
    assert np.allclose(mdp.T.sum(axis=2), 1.0)
    assert mdp.states[-1] == "terminal" and mdp.gamma == 0.9
    assert straight_line_hex_world().T.shape == (8, 6, 8)


def test_neighbors_in_offset_coordinates():
    assert neighbor((0, 0), "SE") == (1, 0)      # even row: SE keeps the column
    assert neighbor((1, 8), "SE") == (2, 9)      # odd row: SE moves one column right
    assert neighbor((1, 0), "NW") == (0, 0)


def test_rewards_and_bump_cost():
    mdp = hex_world()
    i = mdp.states.index((0, 0))
    east = DIRECTIONS.index("E")
    # Heading east from the top-left tile: veering NE leaves the map (p = 0.15).
    assert np.isclose(mdp.R[i, east], -0.15)
    assert np.isclose(mdp.T[i, east, i], 0.15)
    goal = mdp.states.index((2, 9))
    assert np.all(mdp.R[goal] == 10.0) and np.all(mdp.T[goal, :, -1] == 1.0)


def test_optimal_policy_heads_for_the_big_reward():
    mdp = straight_line_hex_world()
    U, _ = value_iteration(mdp.T, mdp.R, mdp.gamma)
    policy, _ = greedy(mdp.T, mdp.R, mdp.gamma, U)
    assert all(mdp.actions[a] == "E" for a in policy[:6])
    assert np.all(np.diff(U[:7]) > 0)   # values rise toward the +10 tile


@pytest.mark.skipif(shutil.which("julia") is None, reason="julia not installed")
def test_julia_builds_the_same_arrays():
    code = (
        "using ADM; m = hex_world(); "
        'println(join(vec(permutedims(m.T, (3, 2, 1))), " ")); '
        'println(join(vec(permutedims(m.R)), " "))'
    )
    out = subprocess.run(
        ["julia", f"--project={ROOT / 'julia'}", "--startup-file=no", "-e", code],
        capture_output=True, text=True, check=True, timeout=600,
    ).stdout.splitlines()
    mdp = hex_world()
    assert np.allclose(np.array(out[0].split(), dtype=float), mdp.T.ravel())
    assert np.allclose(np.array(out[1].split(), dtype=float), mdp.R.ravel())
