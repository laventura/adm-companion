# Shared container for finite MDPs (mirrors adm.problems.DiscreteMDP).
"A finite MDP: T[s, a, s′] = T(s′ | s, a), R[s, a] = R(s, a), with labels for states and actions."
struct DiscreteMDP
    T::Array{Float64, 3}
    R::Matrix{Float64}
    γ::Float64
    states::Vector{Any}
    actions::Vector{String}
end
