# Hex World (ADM Appendix F.1). Mirrors adm/problems/hex_world.py; tests/test_hex_world.py
# checks that both languages build identical arrays.
# Tiles are (row, col) in "odd-r" offset coordinates: row 0 on top, odd rows shifted right.

# Counterclockwise from east, so neighbors in this list are neighboring directions.
const HEX_DIRECTIONS = ["E", "NE", "NW", "W", "SW", "SE"]
const HEX_STEPS = Dict(   # (row step, col step) for even rows, then for odd rows
    "E" => ((0, 1), (0, 1)), "W" => ((0, -1), (0, -1)),
    "NE" => ((-1, 0), (-1, 1)), "NW" => ((-1, -1), (-1, 0)),
    "SE" => ((1, 0), (1, 1)), "SW" => ((1, -1), (1, 0)),
)

const HEX_STANDARD_TILES = [[(0, c) for c in 0:9]; [(1, c) for c in (0, 1, 2, 4, 7, 8)];
                            [(2, c) for c in (0, 1, 2, 3, 5, 6, 7, 9)]]
const HEX_STANDARD_REWARDS = Dict((1, 0) => 5.0, (2, 2) => -10.0, (2, 9) => 10.0)

function hex_neighbor(tile, direction)
    r, c = tile
    dr, dc = HEX_STEPS[direction][mod(r, 2) + 1]
    return (r + dr, c + dc)
end

function hex_mdp(tiles, rewards; γ = 0.9, p_intended = 0.7, bump_cost = 1.0)
    index = Dict(tile => i for (i, tile) in enumerate(tiles))
    n, terminal, nA = length(tiles) + 1, length(tiles) + 1, length(HEX_DIRECTIONS)
    T, R = zeros(n, nA, n), zeros(n, nA)
    p_veer = (1 - p_intended) / 2
    for (i, tile) in enumerate(tiles)
        if haskey(rewards, tile)
            R[i, :] .= rewards[tile]
            T[i, :, terminal] .= 1.0
            continue
        end
        # da = 0 is the chosen direction; ±1 are its two neighbors. mod1 wraps 0 → 6 and 7 → 1.
        for a in 1:nA, (da, p) in ((0, p_intended), (-1, p_veer), (1, p_veer))
            target = hex_neighbor(tile, HEX_DIRECTIONS[mod1(a + da, nA)])
            if haskey(index, target)
                T[i, a, index[target]] += p
            else   # off the map: stay put and pay the bump cost in expectation
                T[i, a, i] += p
                R[i, a] -= p * bump_cost
            end
        end
    end
    T[terminal, :, terminal] .= 1.0
    return DiscreteMDP(T, R, γ, Any[tiles; "terminal"], copy(HEX_DIRECTIONS))
end

hex_world() = hex_mdp(HEX_STANDARD_TILES, HEX_STANDARD_REWARDS)
straight_line_hex_world() = hex_mdp([(0, c) for c in 0:6], Dict((0, 6) => 10.0))
