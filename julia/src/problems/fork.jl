# The fork (this site's problem): a POMDP where information has to be paid for.
# Mirrors adm/problems/fork.py; tests/test_problems_parity.py checks the arrays match.
# A hiker at a fork doesn't know which trail leads home. States: 1 = home is left,
# 2 = home is right. Actions: 1 = wait (free, learns nothing), 2 = ask a passerby
# (cost 1; the answer is right 85% of the time), 3 = go left, 4 = go right.
# Going pays +10 on the right trail and -50 on the wrong one; then a new hike starts
# with the home trail equally likely either way. Observations: 1 = "left", 2 = "right"
# (uninformative 50/50 noise after anything but asking). Discount 0.95.
# Structured like the tiger problem (Kaelbling, Littman and Cassandra, 1998), plus a wait.
function fork(; γ = 0.95, p_correct = 0.85, reward = 10.0, penalty = -50.0, ask_cost = 1.0)
    T = zeros(2, 4, 2)
    for s in 1:2
        T[s, 1, s] = T[s, 2, s] = 1.0              # waiting and asking change nothing
        T[s, 3, :] .= T[s, 4, :] .= 0.5            # going ends the hike; a new one starts
    end
    O = fill(0.5, 4, 2, 2)                         # uninformative by default
    O[2, 1, :] = [p_correct, 1 - p_correct]        # asked, home is left
    O[2, 2, :] = [1 - p_correct, p_correct]        # asked, home is right
    R = [0.0 -ask_cost reward penalty;
         0.0 -ask_cost penalty reward]
    return DiscretePOMDP(T, R, O, γ, ["home left", "home right"], ["wait", "ask", "go left", "go right"],
                         ["left", "right"])
end
