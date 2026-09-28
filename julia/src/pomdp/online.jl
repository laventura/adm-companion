# Online belief state planning (ADM Ch. 22): plan from the current belief only.
# A POMDP p has p.T[s, a, s′], p.R[s, a], p.O[a, s′, o] and p.γ (as in Ch. 20-21).
# Beliefs are probability vectors over states, updated with update_discrete.

# One step of the generative model from a known state: move, observe, reward.
function pomdp_step(p, s, a, rng)
    s′ = rand(rng, Categorical(p.T[s, a, :]))
    o = rand(rng, Categorical(p.O[a, s′, :]))
    return s′, p.R[s, a], o
end

# Rollout from belief b (§22.1): follow `policy` (a function of the belief) for d
# steps. randstep (Ch. 21) draws a state from the current belief at every step;
# that gives the same distribution of rewards and observations as tracking one
# hidden state, because the belief is exactly the posterior over that state.
function belief_rollout(p, b, policy, d, rng)
    ret = 0.0
    for t in 0:d-1
        b, r = randstep(p, b, policy(b), rng)
        ret += p.γ^t * r
    end
    return ret
end

# Forward search (eq. 22.1): expand every action and every possible observation
# to depth d, and value the leaves with U. `count` tallies belief nodes visited.
function forward_search_belief(p, b, d, U; count = Ref(0))
    count[] += 1
    d == 0 && return (a = 0, u = U(b))
    Ud(b′) = forward_search_belief(p, b′, d - 1, U; count).u   # value with d - 1 left
    q = [lookahead_belief(p, Ud, b, a) for a in 1:nactions(p)]
    return (a = argmax(q), u = maximum(q))
end

# Branch and bound (§22.3): forward search with Ulo at the leaves, but skip an
# action when its upper bound Qhi(b, a) can't beat the best value found so far.
# With true bounds, the answer is the same as forward_search_belief(p, b, d, Ulo).
function branch_and_bound_belief(p, b, d, Ulo, Qhi; count = Ref(0))
    count[] += 1
    d == 0 && return (a = 0, u = Ulo(b))
    Ud(b′) = branch_and_bound_belief(p, b′, d - 1, Ulo, Qhi; count).u
    best = (a = 0, u = -Inf)
    hi = [Qhi(b, a) for a in 1:nactions(p)]
    for a in sortperm(hi, rev = true)          # most promising action first
        hi[a] <= best.u && break               # this one and all after it: pruned
        u = lookahead_belief(p, Ud, b, a)
        u > best.u && (best = (a = a, u = u))
    end
    return best
end

# Sparse sampling (eq. 22.2): instead of summing over observations, average m
# sampled (reward, next belief) pairs per action. Cost O((|A| m)^d), whatever |O|.
function sparse_sampling_belief(p, b, d, m, U, rng)
    d == 0 && return (a = 0, u = U(b))
    best = (a = 0, u = -Inf)
    for a in 1:nactions(p)
        u = 0.0
        for _ in 1:m
            b′, r = randstep(p, b, a, rng)
            u += (r + p.γ * sparse_sampling_belief(p, b′, d - 1, m, U, rng).u) / m
        end
        u > best.u && (best = (a = a, u = u))
    end
    return best
end

# Monte Carlo tree search over histories (§22.5; POMCP, Silver and Veness, 2010).
# A history h is a tuple of (action, observation) pairs since the root. N and Q
# are keyed by (h, a). Each simulation draws a state from the root belief and
# follows it down the tree, so no belief is ever computed below the root.
struct HistoryMCTS
    p                        # the POMDP
    N::Dict{Any,Int}         # visit counts N(h, a)
    Q::Dict{Any,Float64}     # value estimates Q(h, a)
    d::Int                   # depth
    c::Float64               # exploration constant
    U                        # leaf value estimate: a function of the state
end
HistoryMCTS(p; d = 10, c = 1.0, U = s -> 0.0) =
    HistoryMCTS(p, Dict{Any,Int}(), Dict{Any,Float64}(), d, c, U)

# UCB1 (eq. 22.3). Untried actions get an infinite bonus, so each is tried
# once first; ties go to the lowest action index.
function mcts_explore(π::HistoryMCTS, h)
    A = 1:nactions(π.p)
    Nh = sum(π.N[(h, a)] for a in A)
    score(a) = π.N[(h, a)] == 0 ? Inf : π.Q[(h, a)] + π.c * sqrt(log(Nh) / π.N[(h, a)])
    return argmax(score, A)
end

function mcts_simulate!(π::HistoryMCTS, s, h, d, rng)
    d <= 0 && return π.U(s)
    if !haskey(π.N, (h, 1))                    # a new history: add it, value its leaf
        for a in 1:nactions(π.p)
            π.N[(h, a)], π.Q[(h, a)] = 0, 0.0
        end
        return π.U(s)
    end
    a = mcts_explore(π, h)
    s′, r, o = pomdp_step(π.p, s, a, rng)
    q = r + π.p.γ * mcts_simulate!(π, s′, (h..., (a, o)), d - 1, rng)   # h extended by (a, o)
    π.N[(h, a)] += 1
    π.Q[(h, a)] += (q - π.Q[(h, a)]) / π.N[(h, a)]    # running mean of the returns
    return q
end

# Run m simulations from belief b (at history h) and return the best root action.
function (π::HistoryMCTS)(b, m, rng; h = ())
    for _ in 1:m
        mcts_simulate!(π, rand(rng, Categorical(b)), h, π.d, rng)
    end
    return argmax(a -> π.Q[(h, a)], 1:nactions(π.p))
end

# Determinized sparse tree search (§22.6; DESPOT, Ye et al., 2017). Φ is an
# m × d matrix of uniform random numbers. A particle is a state on scenario i at
# depth j; Φ[i, j] fixes what happens to it next, for every action.
struct DeterminizedParticle
    s::Int                   # state
    i::Int                   # scenario: row of Φ
    j::Int                   # depth: column of Φ
end

# Inverse-CDF sampling with the stored number Φ[i, j]: walk the joint
# distribution P(s′, o | s, a) = T(s′ | s, a) O(o | a, s′) in a fixed order and
# stop where the running total first reaches Φ[i, j].
function successor(p, Φ, ϕ, a)
    u, total, last_pair = Φ[ϕ.i, ϕ.j], 0.0, (0, 0)
    for s′ in 1:nstates(p), o in 1:nobs(p)             # o varies fastest
        q = p.T[ϕ.s, a, s′] * p.O[a, s′, o]
        q > 0 || continue                              # impossible pairs never chosen
        total += q
        last_pair = (s′, o)
        total >= u && return last_pair
    end
    return last_pair          # rounding left the total a hair below u
end

# m particles, one per scenario, with states drawn from b.
determinized_belief(b, m, rng) = [DeterminizedParticle(rand(rng, Categorical(b)), i, 1) for i in 1:m]

# The belief a particle set stands for: the fraction of particles in each state.
particle_belief(p, particles) = [count(ϕ -> ϕ.s == s, particles) for s in 1:nstates(p)] ./ length(particles)

# Forward search on the determinized tree. Each particle's future is fixed, so an
# action splits the particles by the observation they produce; each group is
# weighted by its share of the particles. At most m histories survive per action
# sequence, so the tree has O(|A|^d m) nodes. U values a leaf's particle set.
function determinized_search(p, Φ, particles, d, U; count = Ref(0))
    count[] += 1
    d == 0 && return (a = 0, u = U(particles))
    best = (a = 0, u = -Inf)
    for a in 1:nactions(p)
        u = sum(p.R[ϕ.s, a] for ϕ in particles) / length(particles)
        groups = Dict{Int,Vector{DeterminizedParticle}}()      # observation => particles
        for ϕ in particles
            s′, o = successor(p, Φ, ϕ, a)
            push!(get!(groups, o, DeterminizedParticle[]), DeterminizedParticle(s′, ϕ.i, ϕ.j + 1))
        end
        for kids in values(groups)
            share = length(kids) / length(particles)
            u += p.γ * share * determinized_search(p, Φ, kids, d - 1, U; count).u
        end
        u > best.u && (best = (a = a, u = u))
    end
    return best
end

# Gap heuristic search (§22.7): keep lower and upper bounds at the beliefs seen,
# starting from the functions Ulo and Uhi. Each exploration follows the action
# that is best on the upper bound and the observation whose next belief has the
# widest gap, then backs both bounds up on the way back. Returns the action best
# on the lower bound, the bounds at b after each exploration, and both tables.
function gap_heuristic_search(p, b, Ulo, Uhi; δ = 0.01, k_max = 100, d_max = 10)
    lo, hi = Dict{Vector{Float64},Float64}(), Dict{Vector{Float64},Float64}()
    visit!(x) = haskey(hi, x) || (lo[x] = Ulo(x); hi[x] = Uhi(x))
    function explore(b, d)
        visit!(b)
        (d == 0 || hi[b] - lo[b] <= δ) && return
        # Bounds for every next belief, since the lookaheads below read them.
        nexts = Dict((a, o) => update_discrete(b, p.T, p.O, a, o)
                     for a in 1:nactions(p), o in 1:nobs(p) if observation_probability(p, b, a, o) > 0)
        foreach(visit!, values(nexts))
        a = argmax(a -> lookahead_belief(p, x -> hi[x], b, a), 1:nactions(p))
        # Candidates sorted, so ties go to the lowest observation index.
        _, o = argmax(ao -> hi[nexts[ao]] - lo[nexts[ao]], sort([k for k in keys(nexts) if k[1] == a]))
        explore(nexts[(a, o)], d - 1)
        # A backup of a lower (upper) bound is still one; keep the tighter value.
        lo[b] = max(lo[b], maximum(lookahead_belief(p, x -> lo[x], b, a) for a in 1:nactions(p)))
        hi[b] = min(hi[b], maximum(lookahead_belief(p, x -> hi[x], b, a) for a in 1:nactions(p)))
        return
    end
    history = Tuple{Float64,Float64}[]
    for _ in 1:k_max
        explore(b, d_max)
        push!(history, (lo[b], hi[b]))
        hi[b] - lo[b] < δ && break
    end
    a = argmax(a -> lookahead_belief(p, x -> haskey(lo, x) ? lo[x] : Ulo(x), b, a), 1:nactions(p))
    return (a = a, history = history, lo = lo, hi = hi)
end

# Run a belief-based policy on the true POMDP for `steps` steps from state s and
# belief b; returns the discounted return and the actions taken.
function run_pomdp(p, policy, b, s, steps, rng)
    ret, acts = 0.0, Int[]
    for t in 0:steps-1
        a = policy(b)
        push!(acts, a)
        s′, r, o = pomdp_step(p, s, a, rng)
        ret += p.γ^t * r
        b, s = update_discrete(b, p.T, p.O, a, o), s′
    end
    return ret, acts
end
