"""The template generator produces a self-consistent project."""

import os
import re
import subprocess
import sys

import pytest
import yaml

import make_template


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    dest = tmp_path_factory.mktemp("site") / "pbml-companion"
    make_template.main([str(dest), "--package", "pbml", "--title", "PBML Companion",
                        "--book", "Probabilistic Machine Learning"])
    return dest


def test_expected_files(site):
    for rel in ["_quarto.yml", "pyproject.toml", "index.qmd", "cheatsheets.qmd", "notation.qmd",
                "chapters/ch01-example.qmd", "templates/_chapter.qmd", "theme.scss",
                "pbml/__init__.py", "pbml/style.py", "pbml/page.py", "pbml/io.py",
                "julia/Project.toml", "julia/src/PBML.jl", "julia/src/style.jl",
                "julia/test/runtests.jl", "tools/check_snippets.py", "tools/check_freeze.py"]:
        assert (site / rel).is_file(), rel


def test_names_replaced(site):
    for path in list(site.rglob("*.py")) + list(site.rglob("*.jl")) + list(site.rglob("*.qmd")):
        text = path.read_text()
        assert not re.search(r"\badm\b|\bADM\b|AA228", text), path
    config = yaml.safe_load((site / "_quarto.yml").read_text())
    assert config["website"]["title"] == "PBML Companion"
    assert 'name = "pbml"' in (site / "pyproject.toml").read_text()
    assert 'name = "PBML"' in (site / "julia/Project.toml").read_text()


def test_licenses(site):
    mit = (site / "LICENSE").read_text()
    assert mit.startswith("MIT License") and "Atul Acharya" in mit
    assert "pbml/" in mit and "adm/" not in mit
    content = (site / "LICENSE-content.md").read_text()
    assert "CC BY-NC 4.0" in content and "Probabilistic Machine Learning" in content
    assert not re.search(r"\b(YEAR|AUTHOR|TITLE|SITE_URL|BOOK|PKG)\b|adm|Kochenderfer", content)
    footer = yaml.safe_load((site / "_quarto.yml").read_text())["website"]["page-footer"]
    assert "CC BY-NC 4.0" in footer["center"] and "MIT" in footer["center"]


def test_refuses_non_empty_destination(site):
    with pytest.raises(SystemExit):
        make_template.main([str(site), "--package", "x", "--title", "t", "--book", "b"])


def test_generated_tests_pass(site):
    """The copied tests (style parity, JSON export, page helpers, checkers) run
    against the renamed package."""
    env = {**os.environ, "PYTHONPATH": f"{site}{os.pathsep}{site / 'tools'}"}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                            cwd=site, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
