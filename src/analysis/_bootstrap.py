"""
Put `src/` on sys.path so analysis modules can `import params` / `import utils`.

Pipeline scripts sit directly in `src/`, which Python adds to sys.path[0] automatically when they
are run. Modules in this subfolder get `src/analysis` instead, so they need this first:

    import _bootstrap  # noqa: F401  — must precede the params/utils imports
    import params
    import utils

Notebooks in this folder need the same, or an equivalent `sys.path` insert in their first cell.
"""
import pathlib
import sys

_SRC = str(pathlib.Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
