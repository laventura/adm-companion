"""Shared figure style for Matplotlib and Plotly.

Mirrors julia/src/style.jl; tests/test_style.py fails if the two drift apart.
Matplotlib is imported inside functions so `adm` itself needs only NumPy/SciPy.
"""

# Categorical hues in fixed order: assign series by slot, never cycle past 8.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

# One hue, light to dark (steps 100..700), for magnitudes such as value functions.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# Negative pole (red), neutral midpoint, positive pole (blue).
DIVERGING = ("#e34948", "#f0efec", "#2a78d6")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

FONT_FAMILY = ["Helvetica", "Arial", "DejaVu Sans"]
FONT_SIZE = 11
FIG_SIZE = (6.4, 4.0)  # inches


def apply() -> None:
    """Set Matplotlib defaults for every figure on the page. Call once per page."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "axes.prop_cycle": cycler(color=CATEGORICAL),
        "figure.figsize": FIG_SIZE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.edgecolor": AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
    })


def sequential_cmap():
    """Matplotlib colormap from SEQUENTIAL_BLUE."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("adm_blue", SEQUENTIAL_BLUE)


def diverging_cmap():
    """Matplotlib colormap from DIVERGING; center it with a TwoSlopeNorm."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("adm_diverging", list(DIVERGING))


def plotly_layout() -> dict:
    """Layout settings for Plotly: `fig.update_layout(**plotly_layout())`."""
    axis = {"gridcolor": GRID, "linecolor": AXIS, "zeroline": False}
    return {
        "colorway": CATEGORICAL,
        "font": {"family": ", ".join(FONT_FAMILY), "size": FONT_SIZE, "color": INK},
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "xaxis": axis,
        "yaxis": dict(axis),
        "margin": {"l": 50, "r": 20, "t": 40, "b": 50},
    }
