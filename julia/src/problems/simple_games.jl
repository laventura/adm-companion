# Simple games (ADM Appendix F.10-F.12, plus this site's intersection). Mirrors
# adm/problems/simple_games.py; tests/test_problems_parity.py checks the arrays match.
# R[i, a1, ..., ak] is agent i's reward for the joint action (1-based actions).
"A simple game: rewards R[i, a1, ..., ak], action labels per agent, a discount."
struct SimpleGame
    R::Array{Float64}
    actions::Vector{Vector{String}}
    γ::Float64
    name::String
end

# Stack two payoff tables (rows: agent 1's action, columns: agent 2's) into R[i, a1, a2].
function two_player(R1, R2, actions1, actions2, name; γ = 0.9)
    R = zeros(2, size(R1)...)
    R[1, :, :], R[2, :, :] = R1, R2
    return SimpleGame(R, [collect(actions1), collect(actions2)], γ, name)
end

# F.10: years in prison as negative rewards. Actions: cooperate, defect.
function prisoners_dilemma()
    R1 = [-1.0 -4.0;
           0.0 -3.0]
    return two_player(R1, permutedims(R1), ("cooperate", "defect"), ("cooperate", "defect"), "prisoner's dilemma")
end

# F.11: +1 to the winner, -1 to the loser, 0 for a draw.
function rock_paper_scissors()
    R1 = [ 0.0 -1.0  1.0;       # rock vs rock, paper, scissors
           1.0  0.0 -1.0;       # paper
          -1.0  1.0  0.0]       # scissors
    acts = ("rock", "paper", "scissors")
    return two_player(R1, -R1, acts, acts, "rock-paper-scissors")
end

# F.12: equal claims pay the claim; otherwise the lower claimant gets its claim + 2
# and the higher one gets the lower claim - 2.
function travelers_dilemma(; low = 2, high = 100)
    c = low:high
    R1 = [x == y ? float(x) : x < y ? x + 2.0 : y - 2.0 for x in c, y in c]
    labels = ["\$$v" for v in c]
    return two_player(R1, permutedims(R1), labels, labels, "traveler's dilemma")
end

# This site's problem: two drivers at an intersection choose go or stop. One goes
# and one stops: the goer gains `win`, the other 0. Both go: `crash` each. Both
# stop: `wait` each. A traffic light (a correlated equilibrium) beats every Nash equilibrium.
function intersection(; win = 3.0, crash = -10.0, wait = -1.0)
    R1 = [crash win;
          0.0   wait]
    return two_player(R1, permutedims(R1), ("go", "stop"), ("go", "stop"), "intersection")
end
