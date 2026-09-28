# Mountain trail (this site's problem): a chain MDP where exploration matters.
# Mirrors adm/problems/trail.py; tests/test_problems_parity.py checks the arrays match.
# Positions 1 (trailhead) to 6 (summit) in Julia's 1-based indexing; actions 1 = back, 2 = climb.
# back: down one (rest at the trailhead, reward 0.05). climb: up 0.35, stay 0.6, slip 0.05;
# at the summit, climbing means staying for the view, reward 1. Discount 0.95.
const TRAIL_P_UP, TRAIL_P_SLIP = 0.35, 0.05
const TRAIL_SNACK, TRAIL_VIEW = 0.05, 1.0

function trail(; n = 6, γ = 0.95)
    T = zeros(n, 2, n)
    R = zeros(n, 2)
    for s in 1:n
        T[s, 1, max(s - 1, 1)] = 1.0                 # back: deterministic
        up, down = min(s + 1, n), max(s - 1, 1)
        # += because at the ends two outcomes land on the same position.
        T[s, 2, up] += TRAIL_P_UP
        T[s, 2, down] += TRAIL_P_SLIP
        T[s, 2, s] += 1 - TRAIL_P_UP - TRAIL_P_SLIP
    end
    R[1, 1] = TRAIL_SNACK
    R[n, 2] = TRAIL_VIEW
    return DiscreteMDP(T, R, γ, collect(Any, 0:n-1), ["back", "climb"])
end
