"""Check that Quarto's frozen results match the current page sources.

With `freeze: auto`, Quarto re-executes a page only when the md5 of its
source differs from the `hash` stored in _freeze/<page>/execute-results/
html.json; otherwise it reuses the markdown saved there, prose included.
If two renders overlap (for example, `quarto preview` re-rendering on save
while another render runs), the cache can end up with the new hash but the
old text, and Quarto keeps serving the old text. This finds such pages.

Usage: uv run python tools/check_freeze.py
Exits with status 1 if any page is STALE.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def prose_lines(text: str) -> list[str]:
    """Source lines that Quarto copies verbatim into the frozen markdown:
    everything outside the YAML front matter and code blocks, except lines
    with inline code such as `{python} x`, which freezing replaces by a value."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":            # skip the front matter
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        lines = lines[end + 1:]
    kept, fence = [], None
    for line in lines:
        match = FENCE.match(line)
        if fence:
            # A block closes on a line of only the same fence character,
            # at least as long as the one that opened it.
            if match and line.strip() == match[1] and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                fence = None
            continue
        if match:
            fence = match[1]
            continue
        stripped = line.strip()
        if stripped and "`{python}" not in stripped and "`{julia}" not in stripped:
            kept.append(stripped)
    return kept


def check(page: Path) -> tuple[str, list[str]]:
    """Return (status, missing lines) for one page: OK, OUTDATED or STALE."""
    frozen = json.loads((ROOT / "_freeze" / page.relative_to(ROOT).with_suffix("") /
                         "execute-results" / "html.json").read_text(encoding="utf-8"))
    source = page.read_text(encoding="utf-8")
    if frozen.get("hash") != hashlib.md5(source.encode("utf-8")).hexdigest():
        return "OUTDATED", []          # the next render re-executes this page anyway
    cached = {line.strip() for line in frozen["result"]["markdown"].splitlines()}
    missing = [line for line in prose_lines(source) if line not in cached]
    return ("STALE" if missing else "OK"), missing


def main() -> int:
    pages = sorted(p for p in ROOT.glob("**/*.qmd")
                   if (ROOT / "_freeze" / p.relative_to(ROOT).with_suffix("")).is_dir())
    stale = 0
    for page in pages:
        status, missing = check(page)
        if status == "OUTDATED":
            note = "  (source changed since the last run; the next render re-executes it)"
        elif status == "STALE":
            note = f"  ({len(missing)} source lines missing from the cache, e.g. {missing[0][:70]!r})"
        else:
            note = ""
        print(f"{status:<9} {page.relative_to(ROOT)}{note}")
        stale += status == "STALE"
    if stale:
        print(f"\n{stale} stale page(s). Fix: uv run quarto render <page>.qmd for each (a single-file render always re-executes).")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
