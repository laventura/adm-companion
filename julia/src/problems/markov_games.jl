# Markov games for Part V: the predator-prey hex world (ADM Appendix F.13) and
# this site's lake. Mirrors adm/problems/markov_games.py; tests/test_problems_parity.py
# checks the arrays match. T[s, a1, a2, s′] and R[i, s, a1, a2], 1-based.
"A two-agent Markov game: T[s, a1, a2, s′], rewards R[i, s, a1, a2], discount, labels."
struct MarkovGame
    T::Array{Float64,4}
    R::Array{Float64,4}
    γ::Float64
    states::Vector
    actions::Vector{Vector{String}}
    name::String
end

# The map of the book's figure F.16, in odd-r coordinates (odd rows shifted right).
const PREDATOR_PREY_TILES = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 2), (2, 4), (2, 5),
                             (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]
const PREDATOR_START, PREY_START = (1, 2), (3, 2)

# M[x, a, y]: an agent on tile x choosing direction a ends on tile y. As in Hex
# World: the chosen direction w.p. 0.7, each neighboring one w.p. 0.15; off the map, stay.
function hex_moves(tiles; p_intended = 0.7)
    index = Dict(t => k for (k, t) in enumerate(tiles))
    n, nA = length(tiles), length(HEX_DIRECTIONS)
    M = zeros(n, nA, n)
    for (x, tile) in enumerate(tiles), a in 1:nA
        for (da, p) in ((0, p_intended), (-1, (1 - p_intended) / 2), (1, (1 - p_intended) / 2))
            target = hex_neighbor(tile, HEX_DIRECTIONS[mod1(a + da, nA)])
            M[x, a, get(index, target, x)] += p
        end
    end
    return M
end

# F.13 with one predator (agent 1) and one prey (agent 2). State (x, y) has index
# (x - 1) * n + y, matching Python's x * n + y with 0-based x and y. When they share a
# tile the predator gets `capture`, the prey `eaten`, and the prey reappears uniformly at
# random. (Python calls `capture` `catch`, which is a reserved word in Julia.)
function predator_prey(; γ = 0.9, move_cost = 1.0, capture = 1.0, eaten = -100.0)
    tiles = PREDATOR_PREY_TILES
    n, nA = length(tiles), length(HEX_DIRECTIONS)
    M = hex_moves(tiles)
    T, R = zeros(n^2, nA, nA, n^2), zeros(2, n^2, nA, nA)
    for x in 1:n, y in 1:n
        s, caught = (x - 1) * n + y, x == y
        R[1, s, :, :] .= -move_cost + (caught ? capture : 0.0)
        R[2, s, :, :] .= caught ? eaten : 0.0
        for a1 in 1:nA, a2 in 1:nA
            prey_next = caught ? fill(1 / n, n) : M[y, a2, :]
            # Next state (x′, y′) at (x′ - 1) * n + y′: y′ varies fastest, so flatten
            # the outer product row by row (vec of its transpose).
            T[s, a1, a2, :] = vec(permutedims(M[x, a1, :] * prey_next'))
        end
    end
    states = [(tx, ty) for tx in tiles for ty in tiles]
    return MarkovGame(T, R, γ, states, [HEX_DIRECTIONS, HEX_DIRECTIONS], "predator-prey hex world")
end

# This site's problem: two boats fish a shared lake. Stock 1 = low, 2 = medium,
# 3 = high; action 1 = light, 2 = heavy. Catch: light 1, 2, 3; heavy 2, 3.5, 5.
# Stock moves with the number h of heavy boats: h = 0: up 0.7, same 0.3;
# h = 1: up 0.2, same 0.5, down 0.3; h = 2: down 0.7, same 0.3 (clamped to low..high).
function lake(; γ = 0.9)
    harvest = [1.0 2.0; 2.0 3.5; 3.0 5.0]      # harvest[s, effort]
    moves = Dict(0 => [(1, 0.7), (0, 0.3)], 1 => [(1, 0.2), (0, 0.5), (-1, 0.3)],
                 2 => [(-1, 0.7), (0, 0.3)])
    T, R = zeros(3, 2, 2, 3), zeros(2, 3, 2, 2)
    for s in 1:3, a1 in 1:2, a2 in 1:2
        R[1, s, a1, a2], R[2, s, a1, a2] = harvest[s, a1], harvest[s, a2]
        for (step, p) in moves[(a1 - 1) + (a2 - 1)]
            T[s, a1, a2, clamp(s + step, 1, 3)] += p
        end
    end
    return MarkovGame(T, R, γ, ["low", "medium", "high"], [["light", "heavy"], ["light", "heavy"]], "lake")
end
