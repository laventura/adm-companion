"""Python and Julia figures must share one style (CLAUDE.md §6.3)."""

import re
from pathlib import Path

import pytest

from adm import style

JULIA_STYLE = (Path(__file__).parent.parent / "julia" / "src" / "style.jl").read_text()


def julia_const(name: str) -> str:
    match = re.search(rf"^const {name} = (.*?)(?=^const |\Z)", JULIA_STYLE, re.MULTILINE | re.DOTALL)
    assert match, f"{name} missing from style.jl"
    return re.sub(r"#\s.*", "", match[1]).strip()  # drop "# comments"; hex colors have no space after #


@pytest.mark.parametrize("name", ["CATEGORICAL", "CATEGORICAL_DARK", "SEQUENTIAL_BLUE", "DIVERGING"])
def test_color_lists_match(name):
    assert re.findall(r"#[0-9a-f]{6}", julia_const(name)) == list(getattr(style, name))


@pytest.mark.parametrize("name", ["SURFACE", "INK", "INK_SECONDARY", "MUTED", "GRID", "AXIS"])
def test_colors_match(name):
    assert julia_const(name).strip('"') == getattr(style, name)


def test_sizes_match():
    assert int(julia_const("FONT_SIZE")) == style.FONT_SIZE
    assert tuple(float(x) for x in julia_const("FIG_SIZE").strip("()").split(",")) == style.FIG_SIZE


def test_apply_sets_palette():
    import matplotlib as mpl

    style.apply()
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == style.CATEGORICAL
