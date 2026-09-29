# ADM Companion

An independent, unofficial companion site for *Algorithms for Decision Making* (Kochenderfer, Wheeler & Wray, MIT Press, 2022). The book is freely available at [algorithmsbook.com](https://algorithmsbook.com).

Site: <https://laventura.github.io/adm-companion/>

## Setup

Install [Quarto](https://quarto.org) 1.9+, [uv](https://docs.astral.sh/uv/), and Julia 1.12 (via [juliaup](https://github.com/JuliaLang/juliaup)). Then:

```bash
uv sync                                                    # Python 3.13 environment; installs adm/ editable
julia --project=julia -e 'using Pkg; Pkg.instantiate()'    # Julia environment
```

## Commands

```bash
uv run quarto preview                        # live preview
uv run quarto render                         # build the site into _site/
uv run pytest                                # tests
uv run python tools/check_snippets.py        # run display-only snippets (--chapter 7 for one chapter)
uv run python tools/check_freeze.py          # cached results match the sources
uv run quarto publish gh-pages               # deploy
```

Always run Quarto through `uv run`, so it uses the project's Python.

## Layout

- `intro/`, `part1/` … `part5/`, `appendices/`: one page per chapter.
- `adm/`: Python package (problems, algorithms, figure style, JSON export).
- `julia/`: Julia package `ADM`, the Julia counterpart of `adm/`.
- `tools/check_snippets.py`: runs the code shown in the non-executed language tab.
- `tools/check_freeze.py`: finds pages whose cached results no longer match their source.
- `_freeze/`: cached execution results, committed so the site rebuilds without running code.
- `index.qmd` (book map), `notation.qmd` (symbols by Part), `cheatsheets.qmd` (every chapter's cheat sheet, rebuilt on each render).

## Starting another companion site

The site-agnostic parts (figure style in both languages, theme, chapter template, page helpers, snippet and freeze checkers, and their tests) can seed a new project:

```bash
uv run python tools/make_template.py ../pbml-companion --package pbml \
    --title "PBML Companion" --book "Probabilistic Machine Learning" \
    --site-url https://user.github.io/pbml-companion/
```

It writes a minimal Quarto website with one example chapter, renames the Python package and Julia module, and copies the tests, which pass in the new project. This repository stays the source of those parts; rerun the script into a fresh folder to pick up later changes.
