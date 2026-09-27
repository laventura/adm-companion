"""Grid worlds: an agent moves N/E/S/W on a grid with walls and reward cells.

Each move goes in the chosen direction with probability 1 - slip, and in each of the
other three directions with probability slip / 3. Moving into a wall or off the grid
leaves the agent in place. Acting in a reward cell collects its reward and ends the
episode by moving to an absorbing "done" state, which is the last state index.
"""

from dataclasses import dataclass

import numpy as np

ACTIONS = ("N", "E", "S", "W")
MOVES = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
Cell = tuple[int, int]  # (row, column), row 0 at the top


@dataclass(frozen=True)
class GridWorld:
    rows: int
    cols: int
    rewards: dict[Cell, float]
    walls: frozenset[Cell] = frozenset()
    slip: float = 0.3

    @property
    def cells(self) -> list[Cell]:
        """Non-wall cells in row-major order; state i is cells[i]."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.walls]

    def move(self, cell: Cell, action: str) -> Cell:
        dr, dc = MOVES[action]
        target = (cell[0] + dr, cell[1] + dc)
        inside = 0 <= target[0] < self.rows and 0 <= target[1] < self.cols
        return target if inside and target not in self.walls else cell

    def mdp(self) -> tuple[np.ndarray, np.ndarray]:
        """Return T with shape (S, A, S) and R with shape (S, A); S includes the done state."""
        cells = self.cells
        index = {cell: i for i, cell in enumerate(cells)}
        n, done = len(cells) + 1, len(cells)
        T = np.zeros((n, len(ACTIONS), n))
        R = np.zeros((n, len(ACTIONS)))
        for i, cell in enumerate(cells):
            if cell in self.rewards:
                R[i, :] = self.rewards[cell]
                T[i, :, done] = 1.0
                continue
            for a, intended in enumerate(ACTIONS):
                # Every direction can happen; the intended one is most likely.
                # += (not =) because several directions can land on the same
                # cell, e.g. two bumps into walls both leave you in place.
                for actual in ACTIONS:
                    p = 1 - self.slip if actual == intended else self.slip / 3
                    T[i, a, index[self.move(cell, actual)]] += p
        T[done, :, done] = 1.0
        return T, R


# A rover looks for rock samples: a rich site (+10) beside a crater (-10), and a
# modest rock (+3) closer to the middle of the grid. Original to this site.
ROVER = GridWorld(
    rows=4,
    cols=5,
    rewards={(0, 4): 10.0, (1, 4): -10.0, (3, 0): 3.0},
    walls=frozenset({(1, 1), (2, 1)}),
    slip=0.3,
)
