"""Tests for adm.search (Appendix E)."""

import pytest

from adm import search as S

ROWS = ["..#....",
        "..#.##.",
        "....#G.",
        ".##.#..",
        "......."]


@pytest.fixture(scope="module")
def maze():
    P, goal, cells = S.maze(ROWS)
    man = lambda s: abs(s[0] - goal[0]) + abs(s[1] - goal[1])
    U_hi = lambda s: 0.0 if s == goal else 10.0 - man(s)
    U_lo = lambda s: 0.0 if s == goal else -10.0
    Q_hi = lambda s, a: P.R(s, a) + U_hi(P.T(s, a))
    return P, goal, cells, U_hi, U_lo, Q_hi


def test_every_method_finds_the_shortest_route(maze):
    P, goal, cells, U_hi, U_lo, Q_hi = maze
    dist = S.distances(P, goal, cells)
    assert max(dist.values()) == 11
    for s in cells:
        if s == goal:
            continue
        runs = [S.forward_search(P, s, 12, U_lo), S.branch_and_bound(P, s, 12, U_lo, Q_hi),
                S.memoized_search(P, s, 12, U_lo), S.heuristic_search(P, s, 12, U_hi, U_lo)]
        for a, u, _ in runs:
            assert u == 10 - dist[s]
            assert dist[P.T(s, a)] == dist[s] - 1      # the action is on a shortest route
        assert runs[3][2] <= runs[2][2] < runs[0][2]


def test_short_depth_cannot_reach_goal(maze):
    P, goal, cells, U_hi, U_lo, Q_hi = maze
    # From (0, 0) the goal is 11 moves away; with 3 moves left every leaf is worth -10.
    assert S.forward_search(P, (0, 0), 3, U_lo)[1] == -13
    assert S.heuristic_search(P, (0, 0), 3, U_hi, U_lo)[1] == -13


def test_branch_and_bound_prunes_next_to_goal(maze):
    P, goal, cells, U_hi, U_lo, Q_hi = maze
    # From (3, 6): expand N to (2, 6), then W into the goal; everything else is pruned.
    assert S.branch_and_bound(P, (3, 6), 12, U_lo, Q_hi) == ("N", 8.0, 3)
    assert [Q_hi((3, 6), a) for a in "NSW"] == [8.0, 6.0, 8.0]
    assert [Q_hi((2, 6), a) for a in "NSW"] == [7.0, 7.0, 9.0]


def test_exact_heuristic_walks_straight_to_goal(maze):
    P, goal, cells, U_hi, U_lo, Q_hi = maze
    dist = S.distances(P, goal, cells)
    exact = lambda s: 10.0 - dist[s] if s != goal else 0.0
    for s in cells:
        if s != goal:
            assert S.heuristic_search(P, s, 12, exact, U_lo)[2] == dist[s] + 1
