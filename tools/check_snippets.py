"""Run each page's display-only code snippets and report pass/fail (CLAUDE.md §6.2).

A snippet is a fenced block with a language class and the `.snippet` class, e.g.
```{.julia .snippet}. Per page and language, snippets are joined in page order and
run as one script in a temporary folder, so they cannot overwrite committed files.

Usage: uv run python tools/check_snippets.py [--chapter 7 | --chapter F]
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE_DIRS = ["intro", "part1", "part2", "part3", "part4", "part5", "appendices"]
LANGS = {"python": "py", "julia": "jl"}
FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*(.*?)\s*$")


def extract_snippets(text: str) -> dict[str, list[str]]:
    """Return snippet bodies per language, in page order."""
    snippets: dict[str, list[str]] = {lang: [] for lang in LANGS}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        opening = FENCE.match(lines[i])
        i += 1
        if not opening:
            continue
        marker, info = opening.groups()
        body = []
        while i < len(lines):
            closing = FENCE.match(lines[i])
            i += 1
            if closing and closing[1][0] == marker[0] and len(closing[1]) >= len(marker) and not closing[2]:
                break
            body.append(lines[i - 1])
        if not (info.startswith("{") and info.endswith("}")):
            continue
        classes = {tok[1:] for tok in info[1:-1].split() if tok.startswith(".")}
        lang = next((name for name in LANGS if name in classes), None)
        if lang and "snippet" in classes:
            snippets[lang].append("\n".join(body))
    return snippets


def find_pages(chapter: str | None) -> list[Path]:
    if chapter is None:
        pattern = "*.qmd"
    elif chapter.isdigit():
        pattern = f"ch{int(chapter):02d}-*.qmd"
    else:
        pattern = f"{chapter.lower()}-*.qmd"
    return sorted(p for d in PAGE_DIRS for p in (ROOT / d).glob(pattern))


def run(lang: str, code: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / f"snippets.{LANGS[lang]}"
        script.write_text(code + "\n", encoding="utf-8")
        if lang == "python":
            cmd = [sys.executable, str(script)]
        else:
            cmd = ["julia", f"--project={ROOT / 'julia'}", "--startup-file=no", str(script)]
        return subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chapter", help="chapter number (7) or appendix letter (F)")
    args = parser.parse_args()

    pages = find_pages(args.chapter)
    if not pages:
        print(f"No page found for chapter {args.chapter!r}")
        return 2

    failures = checked = 0
    for page in pages:
        for lang, blocks in extract_snippets(page.read_text(encoding="utf-8")).items():
            if not blocks:
                continue
            checked += 1
            result = run(lang, "\n\n".join(blocks))
            status = "PASS" if result.returncode == 0 else "FAIL"
            print(f"{status}  {page.relative_to(ROOT)}  {lang} ({len(blocks)} snippets)")
            if result.returncode != 0:
                failures += 1
                print("\n".join((result.stdout + result.stderr).splitlines()[-30:]))
    if checked == 0:
        print("No snippets found.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
