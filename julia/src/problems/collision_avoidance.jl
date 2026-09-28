# Aircraft collision avoidance (ADM Appendix F.6). Mirrors adm/problems/collision_avoidance.py;
# tests/test_problems_parity.py checks that both languages give the same policies and values.
# State: altitude h relative to the intruder (m), vertical rate ḣ (m/s), previous advisory,
# time to collision t_col (s). Each second: h ← h + ḣ, ḣ ← ḣ + ḧ + ν, ν ∈ {-2, 0, 2}
# w.p. {¼, ½, ¼}. Integer starts keep h and ḣ integers, so the problem is solved
# exactly on an integer lattice by backward induction.
#
# Julia indexing: advisories are 1 = none, 2 = descend, 3 = climb (Python uses 0, 1, 2).
# Value arrays are U[a_prev, i_h, i_ḣ] with i_h = h + h_max + 1, i_ḣ = ḣ + ḣ_max + 1,
# and policy[t][a_prev, i_h, i_ḣ] is the advisory at t_col = t (t = 1…40).
const CA_ADVISORIES = ("none", "descend", "climb")
const CA_RATES = (0, -5, 5)            # advised vertical rate (m/s); unused for "none"
const CA_T_START = 40
const CA_H_START, CA_HDOT_START = 200, 10
const CA_COLLISION_H = 50

Base.@kwdef struct CAModel
    noise::Int = 2                     # ν is -noise, 0 or +noise (m/s²)
    p_noise::Float64 = 0.5             # probability that ν ≠ 0 (split evenly)
    accel_limit::Int = 1               # largest |ḧ| the pilot uses (m/s²)
    compliance::Float64 = 1.0          # probability the pilot follows an advisory
    collision_cost::Float64 = 1.0
    change_cost::Float64 = 0.01
    h_max::Int = 500                   # lattice bounds; states beyond are clamped
    hdot_max::Int = 30
end

ca_noise(m::CAModel) = ((-m.noise, m.p_noise / 2), (0, 1 - m.p_noise), (m.noise, m.p_noise / 2))

# ḧ for advisory a: toward the advised rate, capped at the limit (0 for "none").
ca_accel(m::CAModel, ḣ, a) = a == 1 ? 0 : clamp(CA_RATES[a] - ḣ, -m.accel_limit, m.accel_limit)

# The branches of one transition as (complies, ν, probability), in the same order as
# Python's outcomes(): complying first, then the three noise values.
function ca_branches(m::CAModel)
    out = Tuple{Bool,Int,Float64}[]
    for (comply, pc) in ((true, m.compliance), (false, 1 - m.compliance))
        pc == 0 && continue
        for (ν, pn) in ca_noise(m)
            push!(out, (comply, ν, pc * pn))
        end
    end
    return out
end

ca_index(m::CAModel, h, ḣ) = (clamp(h, -m.h_max, m.h_max) + m.h_max + 1,
                              clamp(ḣ, -m.hdot_max, m.hdot_max) + m.hdot_max + 1)
ca_grid(m::CAModel) = (-m.h_max:m.h_max, -m.hdot_max:m.hdot_max)
ca_collided(m::CAModel) = [abs(h) < CA_COLLISION_H ? 1.0 : 0.0 for (h, _) in Iterators.product(ca_grid(m)...)]

# EU[a, i, j] = E[U[a, next state]] after advisory a from lattice state (i, j).
function ca_expect(m::CAModel, U)
    EU = zero(U)
    hs, ḣs = ca_grid(m)
    for a in 1:3, (comply, ν, p) in ca_branches(m), (j, ḣ) in enumerate(ḣs), (i, h) in enumerate(hs)
        acc = comply ? ca_accel(m, ḣ, a) : 0      # a pilot who ignores it keeps ḧ = 0
        EU[a, i, j] += p * U[a, ca_index(m, h + ḣ, ḣ + acc + ν)...]
    end
    return EU
end

# Backward induction for the minimum expected cost; with several models it is robust
# dynamic programming (eq. 14.14): score each advisory by its worst model, pick the best.
function ca_optimal_policy(m::CAModel = CAModel(); models = [m])
    change = [m.change_cost * (p != a) for p in 1:3, a in 1:3]
    U = repeat(reshape(m.collision_cost .* ca_collided(m), 1, size(ca_collided(m))...), 3)
    policy = Vector{Array{Int8,3}}(undef, CA_T_START)
    for t in 1:CA_T_START
        EU = reduce((x, y) -> max.(x, y), [ca_expect(mi, U) for mi in models])
        # Q[a_prev, a, i, j]: cost of switching to a, then continuing.
        Q = reshape(change, 3, 3, 1, 1) .+ reshape(EU, 1, size(EU)...)
        best = dropdims(argmin(Q; dims = 2); dims = 2)   # CartesianIndex per entry
        policy[t] = Int8.(getindex.(best, 2))            # its 2nd coordinate is a
        U = dropdims(minimum(Q; dims = 2); dims = 2)
    end
    return policy, U
end

# Threshold policy: climb above the intruder, descend below, when |h| < h_thresh and
# t_col < t_thresh; otherwise no advisory. Ignores ḣ and the previous advisory.
function ca_simple_policy(m::CAModel = CAModel(); h_thresh = 50, t_thresh = 30)
    hs, ḣs = ca_grid(m)
    act = Int8[abs(h) < h_thresh ? (h > 0 ? 3 : 2) : 1 for _ in 1:3, h in hs, _ in ḣs]
    return [t < t_thresh ? act : ones(Int8, size(act)) for t in 1:CA_T_START]
end

# Exact policy evaluation of an additive metric (eq. 14.3): step_metric(a_prev, a) per
# step (default 0), final[i, j] at t_col = 0 (default: the collision indicator).
# With history = true, returns [V at t_col = 0, 1, …, 40] (index t + 1 holds t_col = t).
function ca_evaluate(m::CAModel, policy; step_metric = nothing, final = ca_collided(m),
                     history = false)
    V = repeat(reshape(final, 1, size(final)...), 3)
    Vs = [V]
    for t in 1:CA_T_START
        EV = ca_expect(m, V)
        V = [EV[policy[t][p, i, j], i, j] for p in 1:3, i in axes(V, 2), j in axes(V, 3)]
        if step_metric !== nothing
            V .+= [step_metric(p, policy[t][p, i, j]) for p in 1:3, i in axes(V, 2), j in axes(V, 3)]
        end
        push!(Vs, V)
    end
    return history ? Vs : V
end

# Average of V[a_prev = none, …] over the integer start states (eq. 14.4).
function ca_start_average(m::CAModel, V)
    ih = (-CA_H_START:CA_H_START) .+ (m.h_max + 1)
    iḣ = (-CA_HDOT_START:CA_HDOT_START) .+ (m.hdot_max + 1)
    return sum(V[1, ih, iḣ]) / (length(ih) * length(iḣ))
end

# Roll out one episode from (h, ḣ); returns (final h, advisory changes). Simulation uses
# the exact dynamics; only the policy lookup clamps to the lattice.
function ca_simulate(m::CAModel, policy, h, ḣ, rng)
    a_prev, changes = 1, 0
    noise = ca_noise(m)
    for t in CA_T_START:-1:1
        a = policy[t][a_prev, ca_index(m, h, ḣ)...]
        changes += a != a_prev
        acc = rand(rng) < m.compliance ? ca_accel(m, ḣ, a) : 0
        u = rand(rng)
        ν = u < noise[1][2] ? noise[1][1] : u < noise[1][2] + noise[2][2] ? 0 : noise[3][1]
        h, ḣ, a_prev = h + ḣ, ḣ + acc + ν, a
    end
    return h, changes
end

# Most likely failure (eq. 14.16) for a fixed policy: the adversary picks every branch.
# W[a_prev, i, j] is the best log-probability of a path that ends in a collision.
function ca_most_likely_failure(m::CAModel, policy)
    W = repeat(reshape([c > 0 ? 0.0 : -Inf for c in ca_collided(m)], 1, size(ca_collided(m))...), 3)
    hs, ḣs = ca_grid(m)
    for t in 1:CA_T_START
        W2 = similar(W)
        for p in 1:3, (j, ḣ) in enumerate(ḣs), (i, h) in enumerate(hs)
            a = policy[t][p, i, j]
            W2[p, i, j] = maximum(log(q) + W[a, ca_index(m, h + ḣ, ḣ + (c ? ca_accel(m, ḣ, a) : 0) + ν)...]
                                  for (c, ν, q) in ca_branches(m))
        end
        W = W2
    end
    return W
end
