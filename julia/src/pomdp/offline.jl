# Offline belief state planning (ADM Ch. 21): bounds and approximate value functions.
# A POMDP p has p.T[s, a, s′], p.R[s, a], p.O[a, s′, o] and p.γ; alpha vectors
# are plain vectors over states, and U(b) = max α ⋅ b over a set Γ (utility).

nstates(p) = size(p.T, 1)
nactions(p) = size(p.T, 2)
nobs(p) = size(p.O, 3)

# QMDP (eq. 21.1): one alpha vector per action, by value iteration on the fully
# observed MDP. It assumes the state will be known after one step: an upper bound.
function qmdp(p; k_max = 100)
    Γ = [zeros(nstates(p)) for _ in 1:nactions(p)]
    for _ in 1:k_max
        U = [maximum(α[s′] for α in Γ) for s′ in 1:nstates(p)]  # best action per state
        Γ = [[p.R[s, a] + p.γ * (p.T[s, a, :] ⋅ U) for s in 1:nstates(p)]
             for a in 1:nactions(p)]
    end
    return Γ
end

# Fast informed bound (eq. 21.4): like QMDP, but the next alpha vector may be
# chosen per observation, so information that arrives is partly credited.
function fast_informed_bound(p; k_max = 100)
    Γ = [zeros(nstates(p)) for _ in 1:nactions(p)]
    for _ in 1:k_max
        Γ = [[p.R[s, a] + p.γ * sum(maximum(
                  sum(p.O[a, s′, o] * p.T[s, a, s′] * α[s′] for s′ in 1:nstates(p))
                  for α in Γ) for o in 1:nobs(p))
              for s in 1:nstates(p)] for a in 1:nactions(p)]
    end
    return Γ
end

# Best-action worst-state lower bound (eq. 21.5): the best action's worst
# reward, received forever.
baws_lowerbound(p) = fill(maximum(minimum(p.R[:, a]) for a in 1:nactions(p)) /
                          (1 - p.γ), nstates(p))

# Blind lower bound (eq. 21.6): commit to one action forever, ignoring
# observations; one alpha vector per action, iterated from the BAWS bound.
function blind_lowerbound(p; k_max = 100)
    Γ = [baws_lowerbound(p) for _ in 1:nactions(p)]
    for _ in 1:k_max
        Γ = [[p.R[s, a] + p.γ * (p.T[s, a, :] ⋅ Γ[a]) for s in 1:nstates(p)]
             for a in 1:nactions(p)]
    end
    return Γ
end

# Point-based backup at belief b (eqs. 21.8-21.10): for each action, pick the
# best alpha vector at each updated belief, combine them (as in eq. 20.17),
# and keep the action whose new vector is best at b.
function belief_backup(p, Γ, b)
    best, best_val = Float64[], -Inf
    for a in 1:nactions(p)
        αo = map(1:nobs(p)) do o
            po = observation_probability(p, b, a, o)
            # An impossible observation's subplan never matters; any vector works.
            po > 0 || return Γ[1]
            b′ = update_discrete(b, p.T, p.O, a, o)
            Γ[argmax([α ⋅ b′ for α in Γ])]
        end
        α = combine_alpha(p, a, αo)
        if α ⋅ b > best_val
            best, best_val = α, α ⋅ b
        end
    end
    return best
end

# Point-based value iteration (algorithm 21.7): one backup per belief in B.
function pbvi(p, B; k_max = 50, Γ = [baws_lowerbound(p)])
    for _ in 1:k_max
        Γ = [belief_backup(p, Γ, b) for b in B]
    end
    return Γ
end

# Randomized point-based value iteration (Perseus, algorithm 21.8): back up
# random beliefs until every belief in B has improved (or kept its value).
function perseus(p, B, rng; k_max = 50, Γ = [baws_lowerbound(p)])
    for _ in 1:k_max
        Γ′, todo = Vector{Float64}[], collect(B)
        while !isempty(todo)
            b = rand(rng, todo)
            α_old = Γ[argmax([α ⋅ b for α in Γ])]
            α_new = belief_backup(p, Γ, b)
            push!(Γ′, α_new ⋅ b ≥ α_old ⋅ b ? α_new : α_old)
            # Keep only beliefs whose value Γ′ has not yet matched.
            filter!(b -> utility(Γ′, b) < utility(Γ, b), todo)
        end
        Γ = Γ′
    end
    return Γ
end

# Sawtooth upper bound (eqs. 21.14-21.16). V maps beliefs to upper-bound
# utilities and must contain every corner belief e_i. Each non-corner pair
# (b′, u′) is a "tooth": the corners' plane, dipping down to u′ at b′.
function sawtooth(V, b)
    haskey(V, b) && return V[b]
    n = length(b)
    corners = [[i == j ? 1.0 : 0.0 for j in 1:n] for i in 1:n]
    uE = [V[e] for e in corners]
    best = uE ⋅ b                          # the corners' plane alone
    for (b′, u′) in V
        b′ in corners && continue
        # ρ: how much of b′ fits inside b (the minimum ratio, eq. 21.14).
        ρ = minimum(b[i] / b′[i] for i in 1:n if b′[i] > 0)
        best = min(best, uE ⋅ b + ρ * (u′ - uE ⋅ b′))
    end
    return best
end

# Sawtooth iteration (algorithm 21.10): replace each non-corner value in B by
# its one-step lookahead on the current sawtooth bound.
function sawtooth_iteration(p, V, B; k_max = 10)
    corners = [[i == j ? 1.0 : 0.0 for j in 1:nstates(p)] for i in 1:nstates(p)]
    for _ in 1:k_max
        U = b -> sawtooth(V, b)
        V = Dict(b => (b in corners ? V[b] : greedy_belief(p, U, b).u) for b in B)
    end
    return V
end

# One random step in belief space (algorithm 21.11): sample a state from b,
# move and observe with the model, and update the belief.
function randstep(p, b, a, rng)
    s = rand(rng, Categorical(b))                    # Distributions.Categorical
    s′ = rand(rng, Categorical(p.T[s, a, :]))
    o = rand(rng, Categorical(p.O[a, s′, :]))
    return update_discrete(b, p.T, p.O, a, o), p.R[s, a]
end

# Exploratory belief expansion (algorithm 21.13): from each belief, try every
# action once and keep the new belief farthest (in L1) from the set so far.
function exploratory_belief_expansion(p, B, rng)
    B′ = copy(B)
    for b in B
        cands = [randstep(p, b, a, rng)[1] for a in 1:nactions(p)]
        dist(c) = minimum(sum(abs.(c .- x)) for x in B′)
        c = cands[argmax(dist.(cands))]
        dist(c) > 0 && push!(B′, c)
    end
    return B′
end

# Sawtooth heuristic search (algorithm 21.14): from b0, repeatedly follow the
# upper bound's greedy action and the observation with the largest gap,
# then tighten both bounds on the way back. Returns (V, Γ, gaps at b0).
function sawtooth_heuristic_search(p, b0; depth = 10, k_max = 30, δ = 0.01)
    fib = fast_informed_bound(p)
    corners = [[i == j ? 1.0 : 0.0 for j in 1:nstates(p)] for i in 1:nstates(p)]
    V = Dict(e => utility(fib, e) for e in corners)          # upper bound
    Γ = [baws_lowerbound(p)]                                   # lower bound
    gap(b) = sawtooth(V, b) - utility(Γ, b)
    function explore(b, d)
        (d >= depth || gap(b) <= δ / p.γ^d) && return
        a = greedy_belief(p, b -> sawtooth(V, b), b).a
        # The observation leading to the belief with the widest gap.
        obs = [o for o in 1:nobs(p) if observation_probability(p, b, a, o) > 0]
        next = [update_discrete(b, p.T, p.O, a, o) for o in obs]
        b′ = next[argmax(gap.(next))]
        explore(b′, d + 1)
        b′ in corners || (V[b′] = greedy_belief(p, b -> sawtooth(V, b), b′).u)
        push!(Γ, belief_backup(p, Γ, b′))
        return
    end
    gaps = [gap(b0)]
    for _ in 1:k_max
        explore(b0, 0)
        V[b0] = min(sawtooth(V, b0), greedy_belief(p, b -> sawtooth(V, b), b0).u)
        push!(Γ, belief_backup(p, Γ, b0))
        push!(gaps, gap(b0))
        gaps[end] < δ && break
    end
    return V, Γ, gaps
end

# Freudenthal interpolation (Lovejoy, 1991) on the belief simplex with
# granularity m. Returns the vertices (beliefs whose entries are multiples of
# 1/m) of the sub-simplex containing b, and weights λ with Σλ = 1, Σλ v = b.
function freudenthal(b, m)
    n = length(b)
    # Cumulative coordinates: x_i = m (b_i + ... + b_n). x_1 = m exactly; set it
    # directly, since a sum a hair below 1 would floor to m - 1.
    x = [i == 1 ? float(m) : m * sum(b[i:n]) for i in 1:n]
    base = floor.(x)
    d = x .- base                                      # fractional parts
    # Visit coordinates by decreasing fraction; coordinate 1 (fraction 0) last.
    order = sortperm(1:n, by = i -> (i == 1, -d[i]))
    to_belief(v) = [(v[i] - (i < n ? v[i+1] : 0.0)) / m for i in 1:n]
    # Vertices: the base, then the base stepped up along order[1], order[2], ...
    verts, v = [to_belief(base)], copy(base)
    for k in 1:n-1
        v[order[k]] += 1
        push!(verts, to_belief(v))
    end
    # Weights: 1 - d_(1), then successive differences of the sorted fractions.
    λ = [1 - d[order[1]]; [d[order[k]] - d[order[k+1]] for k in 1:n-1]]
    return verts, λ
end

# Triangulated value iteration (algorithm 21.16): values at the grid beliefs,
# interpolated elsewhere; each sweep replaces them by one-step lookahead.
# Starting from 0 gives an upper bound when rewards are never positive.
function triangulated_iteration(p, m; k_max = 50)
    n = nstates(p)
    key(b) = round.(b; digits = 10)                    # grid beliefs as keys
    grid = [key(collect(c) ./ m)
            for c in Iterators.product(fill(0:m, n)...) if sum(c) == m]
    V = Dict(b => 0.0 for b in grid)
    function U(b)
        verts, λ = freudenthal(b, m)
        # Skip zero weights: on a tie their vertex may lie off the grid.
        return sum(l * V[key(v)] for (v, l) in zip(verts, λ) if l > 1e-12)
    end
    for _ in 1:k_max
        V = Dict(b => greedy_belief(p, U, b).u for b in grid)   # U sees the old V
    end
    return V, U
end
