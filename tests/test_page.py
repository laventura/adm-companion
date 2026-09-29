"""Tests for adm.page."""

from adm.page import cheat_sheet, page_title


def test_cheat_sheet_links_references(tmp_path):
    page = tmp_path / "p.qmd"
    page.write_text('---\ntitle: "7 · Exact"\n---\n\n## Formalism\n\n| a | b |\n|---|---|\n\n'
                    "## Cheat sheet & connections\n\n| Name | Ref |\n|---|---|\n"
                    "| Backup | @eq-backup |\n| Map | @fig-map-1 |\n\n: caption\n\n| other | table |\n")
    assert page_title(page) == "7 · Exact"
    assert cheat_sheet(page, "x/p.qmd") == (
        "| Name | Ref |\n|---|---|\n| Backup | [eq.](x/p.qmd#eq-backup) |\n"
        "| Map | [fig.](x/p.qmd#fig-map-1) |")


def test_page_without_cheat_sheet(tmp_path):
    page = tmp_path / "q.qmd"
    page.write_text("---\ntitle: Plain\n---\n\nNo tables.\n")
    assert page_title(page) == "Plain" and cheat_sheet(page, "q.qmd") == ""
