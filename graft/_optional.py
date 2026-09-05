"""Turn a missing optional dependency into an actionable message.

Hand refinement and scene inpainting pull in much heavier dependencies, so they
live behind extras and are imported lazily. This turns the resulting
`ModuleNotFoundError` into a message naming the extra to install.
"""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def optional(extra: str, feature: str):
    """Wrap a lazy import of an optional dependency.

    Args:
        extra: the requirements file that provides it, e.g. ``"demo"``.
        feature: what the user was trying to do, named as they would name it.
    """
    try:
        yield
    except ImportError as e:
        missing = getattr(e, "name", None) or "a dependency"
        raise ImportError(
            f"{feature} needs the '{extra}' extra, which is not installed "
            f"({missing} is missing).\n"
            f'    pip install -e ".[{extra}]"' 
        ) from e
