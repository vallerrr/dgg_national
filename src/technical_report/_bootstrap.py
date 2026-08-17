"""
Put `src/` on sys.path so notebooks can `import params` / `import utils`.

Jupyter adds only the notebook's own directory to sys.path, so a notebook nested in
`src/notebooks/<stage>/` cannot see `src/`. This walks up until it finds the directory holding
`params.py`, which makes one identical copy of this file work at any depth — hence the copy in
every notebook folder.

    import _bootstrap  # noqa: F401  — must precede the params/utils imports
    import params
"""
import sys
from pathlib import Path

SRC = next(p for p in Path(__file__).resolve().parents if (p / 'params.py').exists())
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
