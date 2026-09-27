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

# The commute as a decision (Ch 6). D: 1 = leave on time, 2 = leave early, 3 = stay home.
# Late now depends on the jam and on D. D's table is a placeholder: D is chosen, not inferred.
const COMMUTE_ACTIONS = ["leave on time", "leave early", "stay home"]
function commute_decision_network()
    c = commute_network()
    parents = Dict{Symbol, Tuple}(:S => (), :R => (), :J => (:S, :R), :A => (:S,), :D => (), :L => (:J, :D))
    cpt = Dict{Symbol, Array{Float64}}(:S => c.cpt[:S], :R => c.cpt[:R], :J => c.cpt[:J], :A => c.cpt[:A],
        :D => fill(1 / 3, 3),
        :L => binary_cpt([0.05 0.0 0.0;     # no jam: on time, early, home
                          0.5  0.1 0.0]))   # jam
    return BayesNet(Dict(:S => 2, :R => 2, :J => 2, :A => 2, :D => 3, :L => 2), parents, cpt)
end
# Utility U[D, L]: late costs 10, leaving early 2 (lost sleep), staying home 6 (a lost workday).
const COMMUTE_UTILITY = [0.0 -10.0; -2.0 -12.0; -6.0 -16.0]

# Commute times (minutes) as a joint Gaussian: 1 = morning, 2 = evening.
# Standard deviations 8 and 10, correlation 0.6, so the covariance is 0.6 × 8 × 10 = 48.
const COMMUTE_TIMES_MEAN = [35.0, 40.0]
const COMMUTE_TIMES_COV = [64.0 48.0; 48.0 100.0]
