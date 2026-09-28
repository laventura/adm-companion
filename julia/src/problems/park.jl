# The park (this site's problem): a gridworld for imitation learning.
# Mirrors adm/problems/park.py; tests/test_problems_parity.py checks the arrays match.
# Cells in row-major order (Julia index = Python index + 1), then an end state.
# Features β[s, a, k] = one-hot terrain (path, grass, mud, gate); reward R = β ⋅ PARK_PHI_TRUE.
# Moves N, E, S, W go the intended way w.p. 0.8, else slip to one of the other three.
const PARK_LAYOUTS = Dict(
    "A" => ["GGGGPPPX", "GMMGPGGG", "GMMGPGMG", "PPPPPGMG", "PGGGGGMG", "PGMMMGGG", "PGMMMGPP", "PPPPPPPG"],
    "B" => ["PPPGGGGX", "GGPGMMGP", "GMPGMMGP", "GMPPPPPP", "GMMGGGMG", "GGGGMGMG", "PPPPMGGG", "PGGGMMMG"])
const PARK_TERRAIN = "PGMX"
const PARK_PHI_TRUE = [-1.0, -2.0, -6.0, 10.0]
const PARK_MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))

function park(layout = "A"; p_intended = 0.8, γ = 0.95)
    grid = PARK_LAYOUTS[layout]
    rows, cols = length(grid), length(grid[1])
    n = rows * cols + 1
    T, β = zeros(n, 4, n), zeros(n, 4, 4)
    for r in 1:rows, c in 1:cols
        s = (r - 1) * cols + c                     # row-major, 1-based
        kind = findfirst(grid[r][c], PARK_TERRAIN)
        β[s, :, kind] .= 1.0
        if grid[r][c] == 'X'
            T[s, :, n] .= 1.0                      # the gate ends the walk
            continue
        end
        for a in 1:4, (d, (dr, dc)) in enumerate(PARK_MOVES)
            p = d == a ? p_intended : (1 - p_intended) / 3
            r2, c2 = r + dr, c + dc
            (1 <= r2 <= rows && 1 <= c2 <= cols) || ((r2, c2) = (r, c))   # a wall: stay put
            T[s, a, (r2 - 1) * cols + c2] += p
        end
    end
    T[n, :, n] .= 1.0
    R = dropdims(sum(β .* reshape(PARK_PHI_TRUE, 1, 1, :); dims = 3); dims = 3)
    states = Any[(r - 1, c - 1) for r in 1:rows for c in 1:cols]
    return DiscreteMDP(T, R, γ, push!(states, "end"), ["N", "E", "S", "W"]), β
end

# Uniform over the left column's cells.
function park_left_starts(layout = "A")
    rows, cols = length(PARK_LAYOUTS[layout]), length(PARK_LAYOUTS[layout][1])
    b = zeros(rows * cols + 1)
    b[[(r - 1) * cols + 1 for r in 1:rows]] .= 1 / rows
    return b
end
