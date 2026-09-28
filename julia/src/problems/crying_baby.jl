# Crying baby (ADM Appendix F.7). Mirrors adm/problems/crying_baby.py;
# tests/test_problems_parity.py checks that both languages build the same arrays.
# States: 1 = sated, 2 = hungry. Actions: 1 = feed, 2 = sing, 3 = ignore.
# Observations: 1 = crying, 2 = quiet.
function crying_baby(; γ = 0.9)
    T = zeros(2, 3, 2)
    T[:, 1, 1] .= 1.0                        # feeding always sates
    for a in (2, 3)                          # singing and ignoring move the same way
        T[1, a, :] = [0.9, 0.1]              # a sated baby gets hungry 10% of the time
        T[2, a, 2] = 1.0                     # a hungry baby stays hungry
    end
    p_cry = [0.1 0.8;                        # feed:   P(crying | sated), P(crying | hungry)
             0.0 0.9;                        # sing:   never cries when sated, cries more when hungry
             0.1 0.8]                        # ignore
    O = zeros(3, 2, 2)
    for a in 1:3, s′ in 1:2
        O[a, s′, :] = [p_cry[a, s′], 1 - p_cry[a, s′]]
    end
    # -10 while hungry (the state before the transition), -5 to feed, -0.5 to sing.
    R = [-10.0 * (s == 2) + (a == 1 ? -5.0 : a == 2 ? -0.5 : 0.0) for s in 1:2, a in 1:3]
    return DiscretePOMDP(T, R, O, γ, ["sated", "hungry"], ["feed", "sing", "ignore"],
                         ["crying", "quiet"])
end
