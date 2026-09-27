# Coffee machine diagnosis, a naive Bayes model (Ch 3). Mirrors adm/problems/coffee.py.
# Class C: 1 = ok, 2 = scale buildup, 3 = worn pump. Symptoms (1 = absent, 2 = present):
# slow flow F, loud noise N, weak taste W, each depending only on C.
const COFFEE_CLASSES = ["ok", "scale buildup", "worn pump"]
const COFFEE_PRIOR = [0.80, 0.15, 0.05]
const COFFEE_P_SYMPTOM = Dict(:F => [0.05, 0.70, 0.60],   # P(symptom present | class)
                              :N => [0.10, 0.20, 0.80],
                              :W => [0.10, 0.60, 0.30])

function coffee_network()
    parents = Dict{Symbol, Tuple}(:C => (), :F => (:C,), :N => (:C,), :W => (:C,))
    cpt = Dict{Symbol, Array{Float64}}(:C => COFFEE_PRIOR)
    for (s, p) in COFFEE_P_SYMPTOM
        cpt[s] = binary_cpt(p)      # rows: class; columns: absent, present
    end
    return BayesNet(Dict(:C => 3, :F => 2, :N => 2, :W => 2), parents, cpt)
end
