"""Exact solution methods for finite MDPs (ADM Ch. 7).

An MDP is a pair of arrays plus a discount factor gamma:
T[s, a, s2] = T(s2 | s, a) and R[s, a] = R(s, a).
A policy is an integer array: policy[s] is the action index taken in state s.

Lines stay under 80 characters because the Ch 7 page prints this source.
"""

import numpy as np

Array = np.ndarray


def lookahead(T: Array, R: Array, gamma: float, U: Array) -> Array:
    """Q[s, a] = R(s, a) + gamma * sum_s2 T(s2 | s, a) * U(s2).

    Computed for every state s and action a at once.
    """
    return R + gamma * T @ U


def greedy(T: Array, R: Array, gamma: float, U: Array) -> tuple[Array, Array]:
    """The greedy policy with respect to U, and each greedy action's value."""
    Q = lookahead(T, R, gamma, U)
    return Q.argmax(axis=1), Q.max(axis=1)


def policy_matrices(T: Array, R: Array, policy: Array) -> tuple[Array, Array]:
    """T_pi[s, s2] and R_pi[s]: the MDP with the policy's action fixed."""
    s = np.arange(len(policy))
    return T[s, policy], R[s, policy]


def policy_evaluation_iterative(
    T: Array, R: Array, gamma: float, policy: Array,
    tol: float = 1e-10, k_max: int = 10_000,
) -> Array:
    """Apply U <- R_pi + gamma * T_pi @ U until U stops changing."""
    T_pi, R_pi = policy_matrices(T, R, policy)
    U = np.zeros(len(policy))
    for _ in range(k_max):
        U_next = R_pi + gamma * T_pi @ U
        if np.max(np.abs(U_next - U)) < tol:
            return U_next
        U = U_next
    return U


def policy_evaluation_exact(
    T: Array, R: Array, gamma: float, policy: Array,
) -> Array:
    """Solve the linear system (I - gamma * T_pi) U = R_pi."""
    T_pi, R_pi = policy_matrices(T, R, policy)
    return np.linalg.solve(np.eye(len(policy)) - gamma * T_pi, R_pi)


def value_iteration(
    T: Array, R: Array, gamma: float,
    delta: float = 1e-6, k_max: int = 10_000,
) -> tuple[Array, list[Array]]:
    """Apply Bellman backups until the Bellman residual is below delta.

    Returns the final U and the list of iterates [U_0, U_1, ...].
    """
    U = np.zeros(R.shape[0])
    history = [U]
    for _ in range(k_max):
        U_next = lookahead(T, R, gamma, U).max(axis=1)
        history.append(U_next)
        if np.max(np.abs(U_next - U)) < delta:
            break
        U = U_next
    return history[-1], history


def policy_iteration(
    T: Array, R: Array, gamma: float, policy: Array, k_max: int = 100,
) -> tuple[Array, Array, list[Array]]:
    """Alternate exact evaluation and greedy improvement until stable.

    Returns the final policy, its value, and the list of policies visited.
    """
    history = [policy]
    U = policy_evaluation_exact(T, R, gamma, policy)
    for _ in range(k_max):
        improved, _ = greedy(T, R, gamma, U)
        if np.array_equal(improved, policy):
            break
        policy = improved
        history.append(policy)
        U = policy_evaluation_exact(T, R, gamma, policy)
    return policy, U, history
