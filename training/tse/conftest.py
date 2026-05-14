"""pytest bootstrap: make the ``tse`` package importable without an install.

The package directory is ``training/tse``; importing it as ``tse`` requires
``training/`` on ``sys.path``. Adding it here keeps ``pytest training/tse``
working straight from a fresh checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))
