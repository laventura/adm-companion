"""Helpers for Quarto pages."""

import inspect
import re
from pathlib import Path

REF_WORDS = {"eq": "eq.", "fig": "fig.", "tbl": "table"}   # short: Ref columns are narrow


def show_source(*objects) -> None:
    """Print the objects' source as one fenced Python block.

    Use in an executed cell with `#| output: asis`, so a code tab shows exactly the
    code that produced the page's results.
    """
    code = "\n\n".join(inspect.getsource(obj).rstrip() for obj in objects)
    print(f"```python\n{code}\n```")


def page_title(page) -> str:
    """The `title:` from a page's YAML front matter, without quotes."""
    match = re.search(r'^title:\s*"?(.*?)"?\s*$', Path(page).read_text(), re.M)
    return match[1] if match else Path(page).stem


def cheat_sheet(page, href) -> str:
    """The first Markdown table under a page's "## Cheat sheet" heading.

    Cross-references such as @eq-bellman only resolve on their own page, so
    they become links to it: [eq.](href#eq-bellman). Returns "" if the
    page has no cheat sheet.
    """
    parts = Path(page).read_text().split("\n## Cheat sheet", 1)
    if len(parts) < 2:
        return ""
    rows = []
    for line in parts[1].splitlines()[1:]:
        if line.startswith("|"):
            rows.append(line)
        elif rows:
            break               # the table has ended
    table = "\n".join(rows)
    return re.sub(r"@((eq|fig|tbl)-[\w-]+)",
                  lambda m: f"[{REF_WORDS[m[2]]}]({href}#{m[1]})", table)
