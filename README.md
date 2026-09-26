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
uv run quarto publish gh-pages               # deploy
```

Always run Quarto through `uv run`, so it uses the project's Python.

## Layout

- `intro/`, `part1/` … `part5/`, `appendices/`: one page per chapter.
- `adm/`: Python package (problems, algorithms, figure style, JSON export).
- `julia/`: Julia package `ADM`, the Julia counterpart of `adm/`.
- `tools/check_snippets.py`: runs the code shown in the non-executed language tab.
- `_freeze/`: cached execution results, committed so the site rebuilds without running code.
