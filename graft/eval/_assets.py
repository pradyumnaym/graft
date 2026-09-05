"""Resolve benchmark support files inside the GRAFT data root.

Note these are resolved at import time (several are default argument
values), so `$GRAFT_DATA_ROOT` must be set before `graft.eval` is imported.
"""
from .. import paths


def asset(*parts: str) -> str:
    """Path to a benchmark support file, e.g. the PROX GT contact cache."""
    return str(paths.data_root().joinpath(*parts))
