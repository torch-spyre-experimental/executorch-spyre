"""Strict prerequisites for the physical-Spyre regression suite."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import executorch.backends

    _our_backends = str(_PROJECT_ROOT / "executorch" / "backends")
    if _our_backends not in executorch.backends.__path__:
        executorch.backends.__path__.append(_our_backends)
except ImportError:
    pass

from .preflight import require_physical_spyre as require_physical_spyre  # noqa: E402
