"""Start a new companion site from this one's site-agnostic parts.

Copies the figure style (Python and Julia), page helpers, JSON export, the
theme, the chapter template, the snippet and freeze checkers and their
tests, and writes a minimal Quarto website around them: config, home page,
notation and cheat-sheet pages, one example chapter, pyproject.toml, a
Julia project, and the licenses (MIT for code, CC BY-NC 4.0 for content). The Python package and Julia module are renamed on the way.

This repository stays the single source: rerun the script to pick up later
improvements, rather than editing a copy by hand.

Usage:
  uv run python tools/make_template.py DEST --package pbml --title "PBML Companion" \
      --book "Probabilistic Machine Learning" --site-url https://user.github.io/pbml/
"""

import argparse
import re
import sys
import uuid
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (source, destination) pairs; "PKG" in a destination is the new package name.
COPIES = [
    ("theme.scss", "theme.scss"),
    ("_syntax/python.xml", "_syntax/python.xml"),
    ("templates/_chapter.qmd", "templates/_chapter.qmd"),
    ("tools/check_snippets.py", "tools/check_snippets.py"),
    ("tools/check_freeze.py", "tools/check_freeze.py"),
    ("adm/style.py", "PKG/style.py"),
    ("adm/io.py", "PKG/io.py"),
    ("adm/page.py", "PKG/page.py"),
    ("tests/test_style.py", "tests/test_style.py"),
    ("tests/test_io.py", "tests/test_io.py"),
    ("tests/test_page.py", "tests/test_page.py"),
    ("tests/test_check_snippets.py", "tests/test_check_snippets.py"),
    ("tests/test_check_freeze.py", "tests/test_check_freeze.py"),
    ("julia/src/style.jl", "julia/src/style.jl"),
    ("julia/src/makie_theme.jl", "julia/src/makie_theme.jl"),
    (".gitignore", ".gitignore"),
    (".python-version", ".python-version"),
]


def rename(text: str, package: str, module: str) -> str:
    """Replace this site's package and module names: `adm` (Python),
    `ADM` (Julia) and the `adm_` prefix of helper names such as adm_theme."""
    text = re.sub(r"\badm_", f"{package}_", text)
    text = re.sub(r"\badm\b", package, text)
    return re.sub(r"\bADM\b", module, text)


def quarto_yml(a) -> str:
    return f"""project:
  type: website
  # Explicit list: otherwise Quarto would also publish CLAUDE.md, README.md, and templates/.
  render:
    - index.qmd
    - notation.qmd
    - cheatsheets.qmd
    - "chapters/*.qmd"
  resources:
    - "chapters/data/*.json"

execute:
  freeze: auto

website:
  title: "{a.title}"
  description: "An independent, unofficial companion to {a.book}."
  site-url: "{a.site_url}"
  page-navigation: true
  sidebar:
    style: docked
    search: true
    collapse-level: 1
    contents:
      - index.qmd
      - notation.qmd
      - cheatsheets.qmd
      - section: "Chapters"
        contents: "chapters/*.qmd"
  page-footer:
    center: >-
      An independent, unofficial companion to *{a.book}*.
      Site text and figures © {a.author},
      [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/);
      code [MIT](https://opensource.org/license/mit).

format:
  html:
    theme:
      light: [cosmo, theme.scss]
      dark: [darkly, theme.scss]
    toc: true
    code-copy: hover
    syntax-highlighting: github
    syntax-definitions:
      - _syntax/python.xml
    code-overflow: wrap
    code-block-border-left: true
    lang: en
"""


def pyproject(a) -> str:
    """This site's pyproject.toml with the package name and description replaced."""
    text = (ROOT / "pyproject.toml").read_text()
    text = re.sub(r'^name = ".*"$', f'name = "{a.package}"', text, count=1, flags=re.M)
    return re.sub(r'^description = ".*"$', f'description = "Code for the {a.title} site"',
                  text, count=1, flags=re.M)


def julia_project(a) -> str:
    """This site's Julia dependencies, under the new module name and a fresh UUID."""
    text = (ROOT / "julia/Project.toml").read_text()
    text = re.sub(r'^name = ".*"$', f'name = "{a.module}"', text, count=1, flags=re.M)
    return re.sub(r'^uuid = ".*"$', f'uuid = "{uuid.uuid4()}"', text, count=1, flags=re.M)


CHEATSHEETS = """---
title: "Cheat sheets"
subtitle: "Every chapter's key equations on one page"
jupyter: python3
execute:
  echo: false
  freeze: false     # rebuilt on every render, so it never lags the chapter pages
---

```{python}
#| output: asis
from pathlib import Path

from PKG.page import cheat_sheet, page_title

for page in sorted(Path("chapters").glob("ch*.qmd")):
    href = f"chapters/{page.name}"
    print(f"\\n## [{page_title(page)}]({href})\\n")
    print(cheat_sheet(page, href) or "*No cheat sheet yet.*")
    print()
```
"""

EXAMPLE_CHAPTER = """---
title: "1 · Example Chapter"
subtitle: "Replace with a real chapter, following templates/_chapter.qmd"
jupyter: python3
execute:
  echo: false
---

```{python}
#| include: false
import numpy as np
import matplotlib.pyplot as plt

from PKG import style

style.apply()
```

## Big picture

A placeholder page that shows the pieces working: a styled figure and a cheat sheet.

```{python}
#| label: fig-example
#| fig-cap: "A styled example figure."
#| fig-alt: "A rising blue curve."
x = np.linspace(0, 1, 50)
plt.plot(x, x ** 2)
plt.show()
```

## Cheat sheet & connections

| Name | Formula | Ref |
|---|---|---|
| Square | $y = x^2$ | @eq-square |

$$
y = x^2
$$ {#eq-square}
"""

JULIA_TEST = """# Run with: julia --project=julia -e 'using Pkg; Pkg.test()'
using MODULE, Test

@testset "style" begin
    @test length(MODULE.CATEGORICAL) == 8
end
"""

README = """# TITLE

An independent, unofficial companion site for *BOOK*.

Generated from the adm-companion template (`tools/make_template.py`).

## Setup

Install [Quarto](https://quarto.org) 1.9+, [uv](https://docs.astral.sh/uv/), and Julia 1.12. Then:

```bash
uv sync                                                    # Python environment; installs PKG/ editable
julia --project=julia -e 'using Pkg; Pkg.instantiate()'    # Julia environment
```

## Commands

```bash
uv run quarto preview                        # live preview
uv run quarto render                         # build the site into _site/
uv run pytest                                # tests
uv run python tools/check_snippets.py        # run display-only snippets
uv run python tools/check_freeze.py          # cached results match the sources
```

## Next steps

- Write chapter pages in `chapters/` (or one folder per part; add each folder to `_quarto.yml`), starting from `templates/_chapter.qmd`.
- Replace the footer and home page text in `_quarto.yml` and `index.qmd`.
- Copy the working rules (stack, conventions, verification) from the adm-companion spec into this project's `CLAUDE.md`.
"""


def mit_license(a) -> str:
    """This site's MIT license with the copyright holder and code folders replaced."""
    text = (ROOT / "LICENSE").read_text()
    text = re.sub(r"^Copyright \(c\) .*$", f"Copyright (c) {date.today().year} {a.author}",
                  text, count=1, flags=re.M)
    return rename(text, a.package, a.module)


CONTENT_LICENSE = """# Content license

Copyright (c) YEAR AUTHOR

The prose, figures, diagrams, worked examples and solved problems of this site
(the `.qmd` pages and the images and data they produce) are licensed under the
**Creative Commons Attribution-NonCommercial 4.0 International License**
(CC BY-NC 4.0).

- Summary: <https://creativecommons.org/licenses/by-nc/4.0/>
- Full legal code: <https://creativecommons.org/licenses/by-nc/4.0/legalcode>

You may share and adapt this material for non-commercial purposes, as long as
you give appropriate credit, link to the license, and indicate if changes were
made.

Suggested credit: *TITLE* by AUTHOR, <SITE_URL>, CC BY-NC 4.0.

## What this license does not cover

- **Source code.** The code in `PKG/`, `julia/`, `tools/` and `tests/`, and the
  code shown on the pages, is licensed under the MIT License; see `LICENSE`.
- **The book.** This is an independent, unofficial companion to *BOOK*, which
  is licensed separately by its authors. This site does not reproduce the
  book's text, figures, exercises or code.
- **Embedded videos** and other third-party material remain under their
  owners' terms.
"""


def build(a) -> list[Path]:
    dest = Path(a.dest)
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"{dest} exists and is not empty")
    written = []

    def write(rel: str, text: str) -> None:
        path = dest / rel.replace("PKG", a.package)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        written.append(path)

    for src, rel in COPIES:
        text = (ROOT / src).read_text()
        if src.endswith((".py", ".jl")):
            text = rename(text, a.package, a.module)
        if src == "templates/_chapter.qmd":
            text = text.replace("ADM Ch. N", "Ch. N").replace("AA228 video", "lecture video")
        write(rel, text)
    write("PKG/__init__.py", f'"""Code for the {a.title} site."""\n')
    write(f"julia/src/{a.module}.jl", f'module {a.module}\n\ninclude("style.jl")\n\nend\n')
    write("julia/test/runtests.jl", JULIA_TEST.replace("MODULE", a.module))
    write("julia/Project.toml", julia_project(a))
    write("pyproject.toml", pyproject(a))
    write("_quarto.yml", quarto_yml(a))
    write("index.qmd", f'---\ntitle: "{a.title}"\n---\n\nAn independent, unofficial '
                       f"companion to *{a.book}*.\n\n- [Notation](notation.qmd)\n"
                       "- [Cheat sheets](cheatsheets.qmd)\n")
    write("notation.qmd", '---\ntitle: "Notation"\n---\n\n| Symbol | Meaning | First used |\n'
                          "|---|---|---|\n| $x$ | An example symbol | Ch 1 |\n")
    write("cheatsheets.qmd", CHEATSHEETS.replace("PKG", a.package))
    write("chapters/ch01-example.qmd", EXAMPLE_CHAPTER.replace("PKG", a.package))
    write("chapters/data/.gitkeep", "")
    write("README.md", README.replace("TITLE", a.title).replace("BOOK", a.book)
                             .replace("PKG", a.package))
    write("LICENSE", mit_license(a))
    content = CONTENT_LICENSE
    for key, value in [("YEAR", str(date.today().year)), ("AUTHOR", a.author),
                       ("TITLE", a.title), ("SITE_URL", a.site_url), ("BOOK", a.book),
                       ("PKG", a.package)]:
        content = content.replace(key, value)
    write("LICENSE-content.md", content)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dest", help="new project folder (must not exist or be empty)")
    p.add_argument("--package", required=True, help="Python package name, e.g. pbml")
    p.add_argument("--module", help="Julia module name (default: package in upper case)")
    p.add_argument("--title", required=True, help='site title, e.g. "PBML Companion"')
    p.add_argument("--book", required=True, help="the book's title")
    p.add_argument("--site-url", default="https://example.github.io/companion/")
    p.add_argument("--author", default="Atul Acharya", help="copyright holder in the licenses")
    a = p.parse_args(argv)
    if not a.package.isidentifier():
        sys.exit(f"--package must be a Python identifier, got {a.package!r}")
    a.module = a.module or a.package.upper()
    for path in build(a):
        print(path.relative_to(a.dest))


if __name__ == "__main__":
    main()
