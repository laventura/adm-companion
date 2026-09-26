# CairoMakie theme built from style.jl, so Julia figures match the Python ones.
# Not part of the ADM module (that keeps `using ADM` fast). In a Julia page:
#   using CairoMakie, ADM
#   include("../julia/src/makie_theme.jl"); set_theme!(adm_theme())
using CairoMakie

function adm_theme()
    ink2 = ADM.INK_SECONDARY
    axis = (
        backgroundcolor = ADM.SURFACE, xgridcolor = ADM.GRID, ygridcolor = ADM.GRID,
        leftspinecolor = ADM.AXIS, bottomspinecolor = ADM.AXIS,
        topspinevisible = false, rightspinevisible = false,
        xtickcolor = ADM.MUTED, ytickcolor = ADM.MUTED,
        xticklabelcolor = ink2, yticklabelcolor = ink2, xlabelcolor = ink2, ylabelcolor = ink2,
        titlecolor = ADM.INK, titlefont = :regular,
    )
    return Theme(
        size = round.(Int, 96 .* ADM.FIG_SIZE),       # Makie units ≈ CSS pixels
        fontsize = round(Int, ADM.FONT_SIZE * 4 / 3),  # points → pixels
        backgroundcolor = ADM.SURFACE,
        textcolor = ADM.INK,
        palette = (color = CairoMakie.Makie.to_color.(ADM.CATEGORICAL),),
        Axis = axis,
        Lines = (linewidth = 2,),
        Scatter = (markersize = 8,),
        Legend = (framevisible = false,),
    )
end
