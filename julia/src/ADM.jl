module ADM

import JSON

include("style.jl")

export write_json

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

end
