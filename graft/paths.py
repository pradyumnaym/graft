"""Where GRAFT looks for weights and body models.

Two kinds of file, kept apart on purpose:

* **Assets** — a few hundred KB of index arrays (sampled SMPL-X point subsets,
  head/neck vertex ids) that define the architecture. These ship inside the
  package, so they are always in sync with the code.
* **Data** — SMPL-X body models, the NLF detector, and the GRAFT checkpoints.
  These are large and/or licence-gated, so they live in a *data root* you
  populate once with ``scripts/download_assets.py``.

The data root is ``$GRAFT_DATA_ROOT`` if set, else ``./data`` relative to the
current working directory. Importing this module also exports ``DATA_ROOT`` for
``smplfitter``, which resolves ``$DATA_ROOT/body_models/<model>`` internally.
"""
from __future__ import annotations

import os
from pathlib import Path

#: Index arrays shipped with the package.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def data_root() -> Path:
    """Root of the downloaded data tree (body models, NLF, checkpoints)."""
    return Path(os.environ.get("GRAFT_DATA_ROOT", "data")).expanduser().resolve()


def body_model_dir() -> Path:
    """Directory holding ``SMPLX_NEUTRAL.npz`` (and the other genders)."""
    return data_root() / "body_models" / "smplx"


def smplx_neutral() -> Path:
    return body_model_dir() / "SMPLX_NEUTRAL.npz"


#: NLF TorchScript builds we accept, most-preferred first. The published numbers
#: were produced with `patch1` (md5 017838f287342f2fe95da5271919526c), so it is
#: preferred: swapping in `patch4` moves PROX PA-MPJPE by ~5 mm, which is a real
#: change in the results, not drift. The others are accepted so an existing data
#: root keeps working, but they do not reproduce the table.
NLF_BUILDS = (
    "nlf_l_multi_0.3.2.patch1.torchscript",
    "nlf_l_0.3.2.patch1.torchscript",
    "nlf_l_0.3.2.patch4.torchscript",
    "nlf_l_multi_0.3.2.torchscript",
)


def nlf_torchscript() -> Path:
    """The NLF pose detector, as a TorchScript archive.

    Accepts any build in :data:`NLF_BUILDS`, preferring the one the paper used.
    Returns the preferred path when none is present, so the error message names
    the file to fetch.
    """
    nlf_dir = data_root() / "nlf"
    for name in NLF_BUILDS:
        if (nlf_dir / name).exists():
            return nlf_dir / name
    return nlf_dir / NLF_BUILDS[0]


def checkpoint_dir() -> Path:
    return data_root() / "checkpoints"


def export_smplfitter_data_root() -> None:
    """Point ``smplfitter`` at our data root.

    ``smplfitter.BodyModel(...)`` without an explicit ``model_root`` reads
    ``$DATA_ROOT`` and looks under ``$DATA_ROOT/body_models/``. Setting it here
    keeps every path in one place instead of relying on the caller's shell.
    """
    os.environ["DATA_ROOT"] = str(data_root())


def require(path: Path, what: str) -> Path:
    """Fail with an actionable message rather than a bare FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(
            f"{what} not found at {path}.\n"
            f"Run `python scripts/download_assets.py` to populate the data root, "
            f"or set $GRAFT_DATA_ROOT to a tree that already has it."
        )
    return path
