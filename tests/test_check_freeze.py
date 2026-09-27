import hashlib
import json

import check_freeze
from check_freeze import check, prose_lines

PAGE = """---
title: "Demo"
---

## Big picture

New prose here.

````markdown
```{python}
not prose
```
````

```{python}
x = 1
```

The answer is `{python} x`.
"""


def test_prose_lines_skip_front_matter_code_and_inline_code():
    assert prose_lines(PAGE) == ["## Big picture", "New prose here."]


def freeze(root, source_text, markdown):
    """Write a page and a matching _freeze entry under root."""
    (root / "part").mkdir()
    page = root / "part" / "demo.qmd"
    page.write_text(source_text)
    cache = root / "_freeze" / "part" / "demo" / "execute-results"
    cache.mkdir(parents=True)
    (cache / "html.json").write_text(json.dumps({
        "hash": hashlib.md5(source_text.encode()).hexdigest(),
        "result": {"markdown": markdown},
    }))
    return page


def test_ok_outdated_and_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(check_freeze, "ROOT", tmp_path)
    page = freeze(tmp_path, PAGE, "## Big picture\n\nNew prose here.\n\n(outputs)\n")
    assert check(page) == ("OK", [])

    page.write_text(PAGE + "\nAn edit after the last render.\n")
    assert check(page)[0] == "OUTDATED"

    # The bug this tool exists for: the hash matches the new source, but the
    # cached markdown still holds the old prose.
    page.write_text(PAGE)
    stale_cache = tmp_path / "_freeze" / "part" / "demo" / "execute-results" / "html.json"
    data = json.loads(stale_cache.read_text())
    data["result"]["markdown"] = "## Big picture\n\nOld prose here.\n"
    stale_cache.write_text(json.dumps(data))
    assert check(page) == ("STALE", ["New prose here."])
