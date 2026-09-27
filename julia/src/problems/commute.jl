# Morning commute (Ch 2, 3). Mirrors adm/problems/commute.py; tests/test_problems_parity.py
# checks that both languages hold the same numbers.
#   Storm (S) → Traffic jam (J) ← Roadwork (R);  J → Late (L);  S → Radio alert (A)
function commute_network()
    parents = Dict{Symbol, Tuple}(:S => (), :R => (), :J => (:S, :R), :L => (:J,), :A => (:S,))
    cpt = Dict{Symbol, Array{Float64}}(
        :S => binary_cpt(0.1),
        :R => binary_cpt(0.2),
        :J => binary_cpt([0.1 0.6; 0.7 0.9]),   # rows: S = no, yes; columns: R = no, yes
        :L => binary_cpt([0.05, 0.5]),          # J = no, yes
        :A => binary_cpt([0.02, 0.95]),         # S = no, yes
    )
    return BayesNet(Dict(x => 2 for x in keys(parents)), parents, cpt)
end

# Commute times (minutes) as a joint Gaussian: 1 = morning, 2 = evening.
# Standard deviations 8 and 10, correlation 0.6, so the covariance is 0.6 × 8 × 10 = 48.
const COMMUTE_TIMES_MEAN = [35.0, 40.0]
const COMMUTE_TIMES_COV = [64.0 48.0; 48.0 100.0]
