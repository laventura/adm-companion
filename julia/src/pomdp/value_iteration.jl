# Pruning and value iteration for POMDPs (ADM §20.4-20.5).

# Largest utility gap of α over the vectors in Γ at any belief (eq. 20.16):
#   maximize δ  subject to  (α - γ′) ⋅ b ≥ δ for all γ′ in Γ,  b ≥ 0,  Σb = 1.
# Solved here with Σb = 1 relaxed to Σb ≤ 1 and δ ≥ 0, which keeps the origin
# feasible (so one phase of the simplex method suffices) and changes nothing
# when the best gap is positive: rescaling b up to Σb = 1 only raises δ.
# Returns the belief where α gains most, or nothing if α never beats Γ.
function find_maximal_belief(α, Γ; tol = 1e-9)
    m = length(α)
    isempty(Γ) && return fill(1 / m, m)
    # Variables x = [b_1..b_m, δ]. Rows: δ - (α - γ′) ⋅ b ≤ 0, and Σb ≤ 1.
    A = vcat([[-(α .- g)' 1.0] for g in Γ]..., [ones(1, m) 0.0])
    rhs = [zeros(length(Γ)); 1.0]
    x = simplex_max([zeros(m); 1.0], A, rhs)
    δ, b = x[end], x[1:m]
    return δ > tol ? b ./ sum(b) : nothing
end

# Maximize c ⋅ x subject to A x ≤ rhs, x ≥ 0, for rhs ≥ 0 (so x = 0 is a
# starting corner). Tableau simplex with Bland's rule, which cannot cycle.
function simplex_max(c, A, rhs; tol = 1e-12)
    m, n = size(A)
    tab = [A Matrix(1.0I, m, m) rhs; -c' zeros(1, m) 0.0]   # last row: -c
    basis = collect(n+1:n+m)               # slack variables start basic
    while true
        # Entering column: the first with a negative reduced cost.
        j = findfirst(<(-tol), tab[end, 1:end-1])
        j === nothing && break             # optimal
        # Leaving row: smallest ratio; ties go to the smallest basic index.
        rows = [i for i in 1:m if tab[i, j] > tol]
        isempty(rows) && error("unbounded linear program")
        ratio(i) = tab[i, end] / tab[i, j]
        best = minimum(ratio, rows)
        i = argmin(i -> basis[i], [i for i in rows if ratio(i) <= best + tol])
        tab[i, :] ./= tab[i, j]            # pivot on (i, j)
        for k in axes(tab, 1)
            k == i || (tab[k, :] .-= tab[k, j] .* tab[i, :])
        end
        basis[i] = j
    end
    x = zeros(n + m)
    x[basis] = tab[1:m, end]
    return x[1:n]
end

# Keep only alpha vectors that are best at some belief (algorithm 20.7).
# A candidate that never beats the current dominating set is dropped; if it
# does beat it somewhere, the candidate best at that belief is added.
function find_dominating(Γ)
    n = length(Γ)
    candidates, dominating = trues(n), falses(n)
    while any(candidates)
        i = findfirst(candidates)
        b = find_maximal_belief(Γ[i], Γ[dominating])
        if b === nothing
            candidates[i] = false
        else
            k = argmax([candidates[j] ? b ⋅ Γ[j] : -Inf for j in 1:n])
            candidates[k], dominating[k] = false, true
        end
    end
    return dominating
end

function prune(plans, Γ)
    keep = find_dominating(Γ)
    return plans[keep], Γ[keep]
end

# Every (k+1)-step plan: a first action, then any k-step plan after each
# observation (|A| × |plans|^|O| of them), with alpha vectors from eq. 20.17.
function expand(plans, Γ, p)
    nA, nO = size(p.R, 2), size(p.O, 3)
    new_plans, new_Γ = ConditionalPlan[], Vector{Float64}[]
    for a in 1:nA, pick in Iterators.product(fill(eachindex(plans), nO)...)
        # pick[o] is the index of the subplan followed after observation o.
        push!(new_plans, ConditionalPlan(a, [plans[i] for i in pick]))
        push!(new_Γ, combine_alpha(p, a, [Γ[i] for i in pick]))
    end
    return new_plans, new_Γ
end

# POMDP value iteration (algorithm 20.8): start from the one-step plans,
# then alternate expanding and pruning. Returns the pruned (plans, Γ) for
# every horizon 1..k_max, and how many plans each expansion produced.
function value_iteration_pomdp(p, k_max)
    plans = [ConditionalPlan(a) for a in axes(p.R, 2)]
    Γ = [p.R[:, a] for a in axes(p.R, 2)]
    counts = [length(plans)]
    plans, Γ = prune(plans, Γ)
    history = [(plans, Γ)]
    for k in 2:k_max
        plans, Γ = expand(plans, Γ, p)
        push!(counts, length(plans))
        plans, Γ = prune(plans, Γ)
        push!(history, (plans, Γ))
    end
    return history, counts
end
