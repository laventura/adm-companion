from check_snippets import extract_snippets, run

PAGE = """\
```{python}
x = 1  # executed cell: not a snippet
```

```{.julia .snippet}
a = 1
```

````markdown
```{.julia .snippet}
shown as an example, not a snippet
```
````

::: {.panel-tabset group="lang"}
## Python
```{.python .snippet}
print("hi")
```
## Julia
```{.snippet .julia}
b = a + 1
```
:::
"""


def test_extracts_snippets_in_page_order():
    snippets = extract_snippets(PAGE)
    assert snippets["julia"] == ["a = 1", "b = a + 1"]
    assert snippets["python"] == ['print("hi")']


def test_ignores_code_that_is_not_a_snippet():
    snippets = extract_snippets("```{python}\nx = 1\n```\n\n```julia\ny = 2\n```\n")
    assert snippets == {"python": [], "julia": []}


def test_run_reports_python_failures():
    assert run("python", "x = 1").returncode == 0
    assert run("python", "raise SystemExit(3)").returncode == 3
