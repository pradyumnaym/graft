"""Import shim for the vendored MapAnything.

GRAFT depends on a patched MapAnything (it adds an `early_exit` path to `infer`,
which returns the encoder features GRAFT samples without running the prediction
heads). That patch is why a copy is vendored under `third_party/` instead of the
upstream package being installed.

If the repository is pip-installed, `mapanything` is a real top-level package and
imports normally. If it is merely cloned, `third_party/` is put on `sys.path`
here so that `import graft` works straight out of the checkout.
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"

try:  # installed, or already importable
    from mapanything.models.mapanything.model import MapAnything
except ModuleNotFoundError:  # plain checkout
    if not _THIRD_PARTY.is_dir():
        raise
    sys.path.insert(0, str(_THIRD_PARTY))
    from mapanything.models.mapanything.model import MapAnything

__all__ = ["MapAnything"]
