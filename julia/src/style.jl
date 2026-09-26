# Shared figure style. Mirrors adm/style.py; tests/test_style.py fails if the two drift apart.
# The plotting library is chosen in Phase 1, so this file holds constants only.

# Categorical hues in fixed order: assign series by slot, never cycle past 8.
const CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
const CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

# One hue, light to dark (steps 100..700), for magnitudes such as value functions.
const SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# Negative pole, neutral midpoint, positive pole.
const DIVERGING = ("#2a78d6", "#f0efec", "#e34948")

const SURFACE = "#fcfcfb"
const INK = "#0b0b0b"
const INK_SECONDARY = "#52514e"
const MUTED = "#898781"
const GRID = "#e1e0d9"
const AXIS = "#c3c2b7"

const FONT_SIZE = 11
const FIG_SIZE = (6.4, 4.0)  # inches
