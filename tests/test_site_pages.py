"""Checks on this site's pages (not copied into the companion template)."""

from pathlib import Path

from adm.page import cheat_sheet

ROOT = Path(__file__).resolve().parent.parent


def test_every_chapter_has_a_cheat_sheet():
    pages = sorted(ROOT.glob("intro/ch*.qmd")) + sorted(ROOT.glob("part*/ch*.qmd"))
    assert len(pages) == 27
    for page in pages:
        table = cheat_sheet(page, "x")
        assert table.count("\n") >= 3 and "@eq-" not in table, page
