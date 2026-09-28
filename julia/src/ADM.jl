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
