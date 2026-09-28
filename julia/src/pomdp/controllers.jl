# Finite state controllers for POMDPs (ADM Ch. 23).
# A POMDP p has p.T[s, a, s′], p.R[s, a], p.O[a, s′, o] and p.γ (as in Ch. 20-22).
# A controller has nodes x = 1..n. At node x it takes action a with probability
# ψ[x, a]; after taking a and observing o it moves to node x′ with probability
# η[x, a, o, x′]. It keeps no belief: the node is its whole memory.
struct Controller
    ψ::Matrix{Float64}            # ψ[x, a]
    η::Array{Float64,4}           # η[x, a, o, x′]
end
nnodes(c::Controller) = size(c.ψ, 1)

# A deterministic controller: action acts[x] at node x, then node next[x][o]
# after observation o (whatever the action, since only one action is ever taken).
function deterministic_controller(p, acts, next)
    n, nA, nO = length(acts), size(p.T, 2), size(p.O, 3)
    ψ, η = zeros(n, nA), zeros(n, nA, nO, n)
    for x in 1:n
        ψ[x, acts[x]] = 1
        for a in 1:nA, o in 1:nO
            η[x, a, o, next[x][o]] = 1
        end
    end
    return Controller(ψ, η)
end

# The controller and the POMDP together form a Markov chain on pairs (x, s),
# stored as index (x - 1) * |S| + s. Returns its transition matrix and reward
# vector (eqs. 23.8-23.9):
#   Tθ[(x,s), (x′,s′)] = Σ_a ψ[x,a] T[s,a,s′] Σ_o O[a,s′,o] η[x,a,o,x′]
#   rθ[(x,s)]          = Σ_a ψ[x,a] R[s,a]
function controller_chain(p, c::Controller)
    nS, nA, nO, n = size(p.T, 1), size(p.T, 2), size(p.O, 3), nnodes(c)
    idx(x, s) = (x - 1) * nS + s
    Tθ, rθ = zeros(n * nS, n * nS), zeros(n * nS)
    for x in 1:n, s in 1:nS, a in 1:nA
        c.ψ[x, a] == 0 && continue
        rθ[idx(x, s)] += c.ψ[x, a] * p.R[s, a]
        for s′ in 1:nS, o in 1:nO, x′ in 1:n
            Tθ[idx(x, s), idx(x′, s′)] += c.ψ[x, a] * p.T[s, a, s′] * p.O[a, s′, o] * c.η[x, a, o, x′]
        end
    end
    return Tθ, rθ
end

# Policy evaluation (eq. 23.1) by one linear solve: u = (I - γ Tθ)⁻¹ rθ.
# Returns U[x, s]; row x is node x's alpha vector (eq. 23.2: U(x, b) = U[x, :] ⋅ b).
function evaluate_controller(p, c::Controller)
    Tθ, rθ = controller_chain(p, c)
    u = (I - p.γ * Tθ) \ rθ
    return permutedims(reshape(u, size(p.T, 1), nnodes(c)))   # reshape fills s fastest
end

# Value at belief b from the best starting node (eq. 23.3), and that node.
function controller_value(p, c::Controller, b; U = evaluate_controller(p, c))
    vals = [U[x, :] ⋅ b for x in 1:nnodes(c)]
    return (u = maximum(vals), x = argmax(vals))
end

# Nodes the controller can ever reach from node x: follow every successor that
# has positive probability after an action that has positive probability.
function reachable_nodes(c::Controller, x)
    seen, stack = Set([x]), [x]
    while !isempty(stack)
        y = pop!(stack)
        for a in axes(c.ψ, 2), o in axes(c.η, 3), y′ in axes(c.η, 4)
            if c.ψ[y, a] > 0 && c.η[y, a, o, y′] > 0 && !(y′ in seen)
                push!(seen, y′); push!(stack, y′)
            end
        end
    end
    return sort(collect(seen))
end

# Run the controller on the true POMDP from node x and state s for `steps`
# steps; returns the discounted return and the (node, action, observation) trace.
function run_controller(p, c::Controller, x, s, steps, rng)
    ret, trace = 0.0, Tuple{Int,Int,Int,Int}[]
    for t in 0:steps-1
        a = rand(rng, Categorical(c.ψ[x, :]))
        s′, r, o = pomdp_step(p, s, a, rng)
        push!(trace, (x, s, a, o))
        ret += p.γ^t * r
        x, s = rand(rng, Categorical(c.η[x, a, o, :])), s′
    end
    return ret, trace
end

# Policy iteration for controllers (§23.2; E. A. Hansen, 1998). Improvement
# adds every deterministic node "take a, then go to next[o]" whose alpha vector
# is not dominated state by state (eq. 23.4). A new node that dominates an old
# node replaces it in place, so every link into the old node now reaches the
# new one: that is how loops form. Stops when no node survives, or at k_max.
# Returns the controller and, per iteration, (nodes, value at b).
function controller_policy_iteration(p, b; k_max = 20, c = nothing, tol = 1e-9, max_nodes = 500)
    nS, nA, nO = size(p.T, 1), size(p.T, 2), size(p.O, 3)
    c === nothing && (c = Controller(fill(1 / nA, 1, nA), ones(1, nA, nO, 1)))   # one uniform node
    history = [(nnodes(c), controller_value(p, c, b).u)]
    for _ in 1:k_max
        U = evaluate_controller(p, c)
        n = nnodes(c)
        # Every candidate: an action and a successor for each observation.
        cands = Tuple{Int,Vector{Int},Vector{Float64}}[]
        for a in 1:nA, next in Iterators.product(fill(1:n, nO)...)
            α = [p.R[s, a] + p.γ * sum(p.T[s, a, s′] * p.O[a, s′, o] * U[next[o], s′]
                                       for s′ in 1:nS, o in 1:nO) for s in 1:nS]
            push!(cands, (a, collect(next), α))
        end
        dominated(α, β) = all(α .<= β .+ tol)                  # α ≤ β in every state
        # Drop candidates no better than an old node, or than an earlier survivor.
        keep = Int[]
        for (i, (_, _, α)) in enumerate(cands)
            any(dominated(α, U[x, :]) for x in 1:n) && continue
            any(dominated(α, cands[j][3]) for j in keep) && continue
            filter!(j -> !dominated(cands[j][3], α), keep)       # α beats earlier survivors
            push!(keep, i)
        end
        isempty(keep) && break                                   # no improvement anywhere
        ψ, η = copy(c.ψ), copy(c.η)
        added = Tuple{Int,Vector{Int}}[]
        for i in keep
            a, next, α = cands[i]
            olds = [x for x in 1:n if dominated(U[x, :], α)]
            if isempty(olds)
                push!(added, (a, next))                          # a genuinely new node
            else
                for x in olds                                    # overwrite dominated nodes
                    ψ[x, :] .= 0; ψ[x, a] = 1
                    η[x, :, :, :] .= 0
                    for a′ in 1:nA, o in 1:nO
                        η[x, a′, o, next[o]] = 1
                    end
                end
            end
        end
        n + length(added) > max_nodes && break
        # Grow the arrays; new nodes point only to old nodes.
        m = n + length(added)
        ψ2, η2 = zeros(m, nA), zeros(m, nA, nO, m)
        ψ2[1:n, :], η2[1:n, :, :, 1:n] = ψ, η
        for (k, (a, next)) in enumerate(added)
            ψ2[n + k, a] = 1
            for a′ in 1:nA, o in 1:nO
                η2[n + k, a′, o, next[o]] = 1
            end
        end
        c = Controller(ψ2, η2)
        push!(history, (nnodes(c), controller_value(p, c, b).u))
    end
    return c, history
end

# Solve M u = r in place by Gaussian elimination with partial pivoting; r
# becomes u. Used by the brute-force search below, which solves millions of
# tiny systems: avoiding allocation makes it about ten times faster.
function solve_inplace!(M, r)
    m = length(r)
    for k in 1:m
        piv = k
        for i in k+1:m
            abs(M[i, k]) > abs(M[piv, k]) && (piv = i)
        end
        if piv != k                                  # swap rows k and piv
            for j in 1:m
                M[k, j], M[piv, j] = M[piv, j], M[k, j]
            end
            r[k], r[piv] = r[piv], r[k]
        end
        for i in k+1:m                               # eliminate below the pivot
            f = M[i, k] / M[k, k]
            f == 0 && continue
            for j in k:m
                M[i, j] -= f * M[k, j]
            end
            r[i] -= f * r[k]
        end
    end
    for k in m:-1:1                                  # back substitution
        t = r[k]
        for j in k+1:m
            t -= M[k, j] * r[j]
        end
        r[k] = t / M[k, k]
    end
    return r
end

# Brute force, for validation: evaluate every deterministic n-node controller
# (|A|^n action choices times n^(n|O|) successor tables) and return the best
# value at belief b (from its best start node), with its actions and successors.
# acts and next are counted like the digits of a mixed-radix number.
function best_deterministic_controller(p, n, b)
    nS, nA, nO = size(p.T, 1), size(p.T, 2), size(p.O, 3)
    m = n * nS
    M, r = zeros(m, m), zeros(m)
    acts, next = ones(Int, n), ones(Int, n * nO)       # next[(x - 1) * nO + o]
    best = (u = -Inf, acts = copy(acts), next = copy(next))
    # Increment a digit vector with base `base`; false once it wraps to all ones.
    function increment!(v, base)
        for i in eachindex(v)
            v[i] < base && (v[i] += 1; return true)
            v[i] = 1
        end
        return false
    end
    while true
        while true
            fill!(M, 0.0)                              # build I - γTθ and rθ
            for x in 1:n, s in 1:nS
                a, i = acts[x], (x - 1) * nS + s
                r[i] = p.R[s, a]
                M[i, i] += 1
                for s′ in 1:nS, o in 1:nO
                    M[i, (next[(x - 1) * nO + o] - 1) * nS + s′] -= p.γ * p.T[s, a, s′] * p.O[a, s′, o]
                end
            end
            solve_inplace!(M, r)                       # r now holds u
            for x in 1:n
                v = 0.0
                for s in 1:nS
                    v += r[(x - 1) * nS + s] * b[s]
                end
                v > best.u + 1e-12 && (best = (u = v, acts = copy(acts), next = copy(next)))
            end
            increment!(next, n) || break
        end
        increment!(acts, nA) || break
    end
    return best
end

# Gradient of J(θ) = β ⋅ u_θ, the value of starting in node 1 with belief b
# (β holds b in node 1's block, zeros elsewhere; eq. 23.11). With Z = I - γTθ,
# u = Z⁻¹rθ and v = Z⁻ᵀβ (how much discounted time is spent in each (x, s)),
#   ∂J/∂θ = v ⋅ (∂rθ/∂θ + γ (∂Tθ/∂θ) u),
# which is eq. 23.18 multiplied by β, arranged to need only two linear solves.
function controller_gradient(p, c::Controller, b)
    nS, nA, nO, n = size(p.T, 1), size(p.T, 2), size(p.O, 3), nnodes(c)
    Tθ, rθ = controller_chain(p, c)
    Z = I - p.γ * Tθ
    β = [x == 1 ? b[s] : 0.0 for s in 1:nS, x in 1:n][:]     # column-major: s fastest
    u, v = Z \ rθ, Z' \ β
    U, V = permutedims(reshape(u, nS, n)), permutedims(reshape(v, nS, n))
    gψ, gη = zeros(n, nA), zeros(n, nA, nO, n)
    for x in 1:n, a in 1:nA, s in 1:nS
        V[x, s] == 0 && continue
        gψ[x, a] += V[x, s] * p.R[s, a]
        for s′ in 1:nS, o in 1:nO
            w = p.γ * V[x, s] * p.T[s, a, s′] * p.O[a, s′, o]
            w == 0 && continue
            for x′ in 1:n
                gψ[x, a] += w * c.η[x, a, o, x′] * U[x′, s′]
                gη[x, a, o, x′] += w * c.ψ[x, a] * U[x′, s′]
            end
        end
    end
    return (J = β ⋅ u, ψ = gψ, η = gη)
end

# Euclidean projection onto the probability simplex (J. Duchi et al., 2008):
# sort, find how many entries stay positive, and shift them all by one amount.
function project_simplex(y)
    μ = sort(y, rev = true)
    cs = cumsum(μ)
    ρ = findlast(j -> μ[j] - (cs[j] - 1) / j > 0, 1:length(μ))
    θ = (cs[ρ] - 1) / ρ
    return max.(y .- θ, 0.0)
end

# A random n-node controller: each distribution drawn uniformly from its simplex
# (normalized exponential draws are uniform on the simplex).
function random_controller(p, n, rng)
    nA, nO = size(p.T, 2), size(p.O, 3)
    unif(k) = (e = -log.(rand(rng, k)); e ./ sum(e))
    ψ = permutedims(reduce(hcat, [unif(nA) for _ in 1:n]))
    η = zeros(n, nA, nO, n)
    for x in 1:n, a in 1:nA, o in 1:nO
        η[x, a, o, :] = unif(n)
    end
    return Controller(ψ, η)
end

# Controller gradient ascent (§23.4): step along the gradient, then project
# each distribution back onto the simplex. Starts from `c` (default: random).
# Returns the controller and J after each step.
function controller_gradient_ascent(p, b, n, rng; α = 0.01, k_max = 2000, c = random_controller(p, n, rng))
    nA, nO, Js = size(p.T, 2), size(p.O, 3), Float64[]
    for _ in 1:k_max
        g = controller_gradient(p, c, b)
        push!(Js, g.J)
        ψ = reduce(vcat, [project_simplex(c.ψ[x, :] .+ α .* g.ψ[x, :])' for x in 1:n])
        η = copy(c.η)
        for x in 1:n, a in 1:nA, o in 1:nO
            η[x, a, o, :] = project_simplex(c.η[x, a, o, :] .+ α .* g.η[x, a, o, :])
        end
        c = Controller(ψ, η)
    end
    push!(Js, controller_gradient(p, c, b).J)
    return c, Js
end
