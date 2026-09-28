# The lost hiker: this site's Dec-POMDP (two rescuers, one shared reward). Mirrors
# adm/problems/lost_hiker.py; tests/test_problems_parity.py checks the arrays match.
# States: 1 = left, 2 = right. Actions: 1 = listen, 2 = search left, 3 = search right.
# Observations: 1 = "left", 2 = "right" (each rescuer's own report). Both searching the
# right valley: +20; one alone: +5; each wrong search: -10; each listener: -1.
# Any search resets the hiker to a random valley; reports are right w.p. 0.8 only
# after both listened, 50-50 otherwise. Both reward rows hold the shared reward.
function lost_hiker(; γ = 0.9, accuracy = 0.8, together = 20.0, alone = 5.0, wrong = -10.0,
                    listen = -1.0)
    T, O, R = zeros(2, 3, 3, 2), zeros(3, 3, 2, 2, 2), zeros(2, 2, 3, 3)
    for a1 in 1:3, a2 in 1:3
        listening = a1 == 1 && a2 == 1
        for s in 1:2
            T[s, a1, a2, :] = listening ? (s == 1 ? [1.0, 0.0] : [0.0, 1.0]) : [0.5, 0.5]
            found = count(a -> a == s + 1, (a1, a2))            # searching valley s
            missed = count(a -> a != 1 && a != s + 1, (a1, a2))
            r = listen * ((a1 == 1) + (a2 == 1)) + wrong * missed
            r += found == 2 ? together : found == 1 ? alone : 0.0
            R[:, s, a1, a2] .= r
        end
        for s′ in 1:2
            p = listening ? accuracy : 0.5
            hear = s′ == 1 ? [p, 1 - p] : [1 - p, p]
            O[a1, a2, s′, :, :] = hear * hear'     # independent reports
        end
    end
    acts, obs = ["listen", "search left", "search right"], ["left", "right"]
    return POMG(T, O, R, γ, ["left", "right"], [acts, acts], [obs, obs], "lost hiker")
end
