module ADM

import JSON
using Distributions: Categorical
using LinearAlgebra

include("style.jl")
include("beliefs/discrete_filter.jl")
include("beliefs/kalman_filter.jl")
include("beliefs/particle_filter.jl")
include("problems/discrete_mdp.jl")
include("problems/hex_world.jl")

export write_json, show_source
export predict_discrete, correct_discrete, update_discrete
export LinearGaussianModel, kalman_predict, kalman_correct, kalman_update
export update_particles
export DiscreteMDP, hex_mdp, hex_world, straight_line_hex_world

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
