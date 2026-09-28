# The multicaregiver crying baby (ADM Appendix F.14), a two-agent POMG. Mirrors
# adm/problems/multicaregiver.py; tests/test_problems_parity.py checks the arrays match.
# T[s, a1, a2, s′], O[a1, a2, s′, o1, o2], R[i, s, a1, a2]; 1-based.
"A two-agent POMG: transitions, joint observation model, rewards, discount, labels."
struct POMG
    T::Array{Float64,4}
    O::Array{Float64,5}
    R::Array{Float64,4}
    γ::Float64
    states::Vector{String}
    actions::Vector{Vector{String}}
    observations::Vector{Vector{String}}
    name::String
end

# States: 1 = sated, 2 = hungry. Actions: 1 = feed, 2 = sing, 3 = ignore.
# Observations: 1 = crying, 2 = quiet; both caregivers always hear the same thing.
# Either caregiver feeding sates the baby; otherwise hungry stays hungry and sated
# stays sated w.p. 0.5. Hungry cries w.p. 0.9, sated never. Both pay 10 while hungry;
# feeding costs caregiver 1 2.5 and caregiver 2 5; singing costs 0.5 and 0.25.
function multicaregiver(; γ = 0.9)
    T, O, R = zeros(2, 3, 3, 2), zeros(3, 3, 2, 2, 2), zeros(2, 2, 3, 3)
    extra = [-2.5 -0.5 0.0;      # caregiver 1: feed, sing, ignore
             -5.0 -0.25 0.0]     # caregiver 2
    for a1 in 1:3, a2 in 1:3
        fed = a1 == 1 || a2 == 1
        T[1, a1, a2, :] = fed ? [1.0, 0.0] : [0.5, 0.5]
        T[2, a1, a2, :] = fed ? [1.0, 0.0] : [0.0, 1.0]
        O[a1, a2, 1, 2, 2] = 1.0                        # sated: always quiet
        O[a1, a2, 2, 1, 1], O[a1, a2, 2, 2, 2] = 0.9, 0.1
        for s in 1:2
            R[1, s, a1, a2] = (s == 2 ? -10.0 : 0.0) + extra[1, a1]
            R[2, s, a1, a2] = (s == 2 ? -10.0 : 0.0) + extra[2, a2]
        end
    end
    acts, obs = ["feed", "sing", "ignore"], ["crying", "quiet"]
    return POMG(T, O, R, γ, ["sated", "hungry"], [acts, acts], [obs, obs], "multicaregiver crying baby")
end
