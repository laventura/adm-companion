"""Helpers for Quarto pages."""

import inspect


def show_source(*objects) -> None:
    """Print the objects' source as one fenced Python block.

    Use in an executed cell with `#| output: asis`, so a code tab shows exactly the
    code that produced the page's results.
    """
    code = "\n\n".join(inspect.getsource(obj).rstrip() for obj in objects)
    print(f"```python\n{code}\n```")
