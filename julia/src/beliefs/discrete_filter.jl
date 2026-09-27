# Discrete state filter (ADM §19.2).
# b[s] is the belief; T[s, a, s′] = T(s′ | s, a); O[a, s′, o] = O(o | a, s′).

# Predict: push the belief through the transition model.
# b_pred(s′) adds up every way of reaching s′: start in s (prob b[s]), then move (prob T[s, a, s′]).
predict_discrete(b, T, a) = [sum(T[s, a, s′] * b[s] for s in eachindex(b)) for s′ in axes(T, 3)]

# Correct: weight each state by how well it explains the observation, then normalize.
function correct_discrete(b_pred, O, a, o)
    b′ = [O[a, s′, o] * b_pred[s′] for s′ in eachindex(b_pred)]
    total = sum(b′)
    total > 0 || throw(ArgumentError("observation $o is impossible under this belief"))
    return b′ ./ total
end

update_discrete(b, T, O, a, o) = correct_discrete(predict_discrete(b, T, a), O, a, o)
