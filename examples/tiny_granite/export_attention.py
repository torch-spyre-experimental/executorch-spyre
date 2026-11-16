# Copyright 2026 The ExecuTorch-Spyre Authors.
# Licensed under the Apache License, Version 2.0.
"""CLI entry point for ``executorch.backends.spyre.export_attention``.

The implementation lives in the package so the tests and other modules can
import it; this wrapper keeps ``python examples/tiny_granite/export_attention.py ...`` working.
"""

import sys
from pathlib import Path

# examples/<demo>/ -> examples/ -> repo root (the ``backends`` package root).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from executorch.backends.spyre.export_attention import main  # noqa: E402

if __name__ == "__main__":
    main()
