# Conditional plans and alpha vectors (ADM §20.2-20.3).
# A POMDP p has p.T[s, a, s′], p.R[s, a], p.O[a, s′, o] and discount p.γ.

# A plan: an action at the root and one subplan per observation (none at a leaf).
struct ConditionalPlan
    a::Int
    subplans::Vector{ConditionalPlan}      # subplans[o] follows observation o
end
ConditionalPlan(a::Int) = ConditionalPlan(a, ConditionalPlan[])

# Utility of plan π from state s (eq. 20.8): the root action's reward, then the
# discounted average over next states s′ and observations o of subplan π(o) from s′.
function evaluate_plan(p, π::ConditionalPlan, s)
    r = p.R[s, π.a]
    isempty(π.subplans) && return r
    nS, nO = size(p.T, 3), size(p.O, 3)
    future = sum(p.T[s, π.a, s′] * p.O[π.a, s′, o] *
                 evaluate_plan(p, π.subplans[o], s′) for s′ in 1:nS, o in 1:nO)
    return r + p.γ * future
end

# Alpha vector: the plan's utility from every state (algorithm 20.3).
# A belief's utility is then the dot product α ⋅ b (eq. 20.11).
alphavector(p, π) = [evaluate_plan(p, π, s) for s in axes(p.T, 1)]

# The same alpha vector built from the subplans' alpha vectors αs[o]
# (eq. 20.17), without re-evaluating the whole tree.
function combine_alpha(p, a, αs)
    nS, nO = size(p.T, 1), size(p.O, 3)
    return [p.R[s, a] + p.γ * sum(p.T[s, a, s′] * p.O[a, s′, o] * αs[o][s′]
                                  for s′ in 1:nS, o in 1:nO) for s in 1:nS]
end

# P(o | b, a) (eqs. 20.14-20.15): move with T, then observe with O.
function observation_probability(p, b, a, o)
    nS = length(b)
    return sum(b[s] * p.T[s, a, s′] * p.O[a, s′, o] for s in 1:nS, s′ in 1:nS)
end

# One-step lookahead from a belief with value function U (eq. 20.13):
# expected reward now plus the discounted value of each updated belief.
function lookahead_belief(p, U, b, a)
    r = sum(p.R[s, a] * b[s] for s in eachindex(b))
    future = 0.0
    for o in axes(p.O, 3)
        po = observation_probability(p, b, a, o)
        po > 0 || continue                 # impossible observations add nothing
        future += po * U(update_discrete(b, p.T, p.O, a, o))
    end
    return r + p.γ * future
end

# The best action and its value by one-step lookahead on U.
function greedy_belief(p, U, b)
    q = [lookahead_belief(p, U, b, a) for a in axes(p.R, 2)]
    return (a = argmax(q), u = maximum(q), q = q)
end

# The value function of a set of alpha vectors Γ: the upper surface (eq. 20.12).
utility(Γ, b) = maximum(α ⋅ b for α in Γ)
