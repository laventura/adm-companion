# Shared container for finite POMDPs (mirrors adm.problems.crying_baby's arrays).
# T[s, a, s′] = T(s′ | s, a), R[s, a] = R(s, a), O[a, s′, o] = O(o | a, s′): the observation
# depends on the action just taken and the state it led to, as in the Ch 19 filters.
"A finite POMDP with transition, reward and observation arrays, a discount and labels."
struct DiscretePOMDP
    T::Array{Float64,3}
    R::Matrix{Float64}
    O::Array{Float64,3}
    γ::Float64
    states::Vector{String}
    actions::Vector{String}
    observations::Vector{String}
end
