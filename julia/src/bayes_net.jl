# Discrete Bayesian network (mirrors adm.bayes_net.BayesNet). Values are 1-based:
# a binary variable takes 1 = no, 2 = yes.
#   card[X]    number of values of X
#   parents[X] tuple of X's parents
#   cpt[X]     array with axes (parent values..., X's value): P(X | parents)
struct BayesNet
    card::Dict{Symbol, Int}
    parents::Dict{Symbol, Tuple}
    cpt::Dict{Symbol, Array{Float64}}
end

# CPT with last axis (no, yes), from P(yes) for each combination of parent values.
# `cat` stacks P(no) and P(yes) along a new last axis.
binary_cpt(p_yes) = cat(1 .- p_yes, p_yes; dims = ndims(p_yes) + 1)
