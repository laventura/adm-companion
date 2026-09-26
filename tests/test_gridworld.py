import numpy as np

from adm.problems.gridworld import ACTIONS, ROVER, GridWorld


def test_transition_rows_are_distributions():
    T, R = ROVER.mdp()
    assert T.shape == (19, 4, 19) and R.shape == (19, 4)
    assert np.allclose(T.sum(axis=2), 1.0)


def test_reward_cell_pays_once_then_ends():
    T, R = ROVER.mdp()
    goal = ROVER.cells.index((0, 4))
    done = len(ROVER.cells)
    assert np.all(R[goal] == 10.0)
    assert np.all(T[goal, :, done] == 1.0)
    assert np.all(T[done, :, done] == 1.0) and np.all(R[done] == 0.0)


def test_slip_and_walls():
    world = GridWorld(rows=1, cols=2, rewards={}, walls=frozenset(), slip=0.3)
    T, _ = world.mdp()
    east = ACTIONS.index("E")
    # From (0,0): east succeeds w.p. 0.7; N, S, W all bump into the edge and stay.
    assert np.isclose(T[0, east, 1], 0.7)
    assert np.isclose(T[0, east, 0], 0.3)
    assert ROVER.move((1, 0), "E") == (1, 0)  # (1,1) is a wall
