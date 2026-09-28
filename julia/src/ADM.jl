module ADM

import JSON
using Distributions: Categorical
using LinearAlgebra

include("style.jl")
include("beliefs/discrete_filter.jl")
include("beliefs/kalman_filter.jl")
include("beliefs/particle_filter.jl")
include("bayes_net.jl")
include("problems/discrete_mdp.jl")
include("problems/hex_world.jl")
include("problems/commute.jl")
include("problems/coffee.jl")
include("problems/mountain_car.jl")
include("problems/simple_regulator.jl")
include("problems/collision_avoidance.jl")
include("problems/trail.jl")
include("problems/park.jl")
include("problems/discrete_pomdp.jl")
include("problems/crying_baby.jl")
include("pomdp/conditional_plans.jl")
include("pomdp/value_iteration.jl")
include("problems/fork.jl")
include("pomdp/offline.jl")
include("pomdp/online.jl")
include("pomdp/controllers.jl")
include("problems/simple_games.jl")
include("problems/markov_games.jl")
include("problems/multicaregiver.jl")

export write_json, show_source
export predict_discrete, correct_discrete, update_discrete
export LinearGaussianModel, kalman_predict, kalman_correct, kalman_update
export update_particles
export DiscreteMDP, hex_mdp, hex_world, straight_line_hex_world
export BayesNet, binary_cpt, probability, commute_network, COMMUTE_TIMES_MEAN, COMMUTE_TIMES_COV
export commute_decision_network, COMMUTE_ACTIONS, COMMUTE_UTILITY
export coffee_network, COFFEE_CLASSES, COFFEE_PRIOR, COFFEE_P_SYMPTOM
export mountain_car_step, mountain_car_goal, MOUNTAIN_CAR_X, MOUNTAIN_CAR_V, MOUNTAIN_CAR_ACTIONS
export regulator_step, regulator_policy, regulator_exact_value, regulator_exact_gradient
export REGULATOR_NOISE_SD, REGULATOR_START, REGULATOR_γ, REGULATOR_HORIZON
export CAModel, CA_ADVISORIES, CA_T_START, CA_H_START, CA_HDOT_START, CA_COLLISION_H
export ca_optimal_policy, ca_simple_policy, ca_evaluate, ca_start_average, ca_simulate
export ca_most_likely_failure, ca_index
export trail
export park, park_left_starts, PARK_PHI_TRUE, PARK_LAYOUTS
export DiscretePOMDP, crying_baby
export ConditionalPlan, evaluate_plan, alphavector, combine_alpha, observation_probability
export lookahead_belief, greedy_belief, utility
export find_maximal_belief, simplex_max, find_dominating, prune, expand, value_iteration_pomdp
export fork, qmdp, fast_informed_bound, baws_lowerbound, blind_lowerbound, belief_backup, pbvi, perseus
export sawtooth, sawtooth_iteration, randstep, exploratory_belief_expansion, sawtooth_heuristic_search
export freudenthal, triangulated_iteration
export pomdp_step, belief_rollout, forward_search_belief, branch_and_bound_belief, sparse_sampling_belief
export HistoryMCTS, mcts_explore, mcts_simulate!, DeterminizedParticle, successor, determinized_belief
export particle_belief, determinized_search, gap_heuristic_search, run_pomdp
export Controller, nnodes, deterministic_controller, controller_chain, evaluate_controller, controller_value
export run_controller, reachable_nodes, controller_policy_iteration, controller_gradient, project_simplex
export controller_gradient_ascent, random_controller, best_deterministic_controller
export SimpleGame, prisoners_dilemma, rock_paper_scissors, travelers_dilemma, intersection
export MarkovGame, predator_prey, lake, hex_moves, PREDATOR_PREY_TILES, PREDATOR_START, PREY_START
export POMG, multicaregiver

"""
    write_json(obj, path) -> path

Write `obj` to `path` as compact JSON, creating parent folders. Mirrors `adm.io.write_json`.
Julia matrices are column-major: pass `collect.(eachrow(A))` to get the same row-major
nesting that NumPy produces.
"""
function write_json(obj, path::AbstractString)
    text = JSON.json(obj)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    write(path, text * "\n")
    return path
end

"""
    show_source(relpath)

Print a file under `julia/src` as one fenced Julia block. Use in an executed cell with
`#| output: asis`, so a code tab shows exactly the code that produced the page's results.
"""
function show_source(relpath::AbstractString, io::IO=stdout)
    code = read(joinpath(@__DIR__, relpath), String)
    println(io, "```julia\n", rstrip(code), "\n```")
end

end
