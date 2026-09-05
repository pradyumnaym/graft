"""Fast-inference defaults, and a guard that keeps them.

`ultralytics` -- pulled in transitively by the optional WiLoR hand refiner -- sets
`CUBLAS_WORKSPACE_CONFIG=":4096:8"` at import, which puts cuBLAS on its slow
deterministic GEMM kernels for the whole process and roughly doubles the cost of
every matmul. GRAFT does not need bitwise determinism at inference, so
`enable_fast_gemm()` removes the variable and turns deterministic algorithms off.

It is called at import, again after the lazy WiLoR import, and once inside
`GRAFTModel.infer` with `warn_if_replanted=True` -- that last call warns if it
finds the flag set again, so a future dependency that plants it shows up as a
warning rather than a silent slowdown.
"""
import os
import warnings

import torch

_DET_ENV = "CUBLAS_WORKSPACE_CONFIG"
_replant_warned = False


def enable_fast_gemm(warn_if_replanted: bool = False) -> bool:
    """Pop the deterministic-cuBLAS env var + disable deterministic algos so cuBLAS/cuDNN
    use their fast (non-deterministic) kernels. Idempotent and ~µs; safe to call on every
    inference. Bit-level results differ only by GEMM reduction order (~1e-6).

    Returns True if it had to UNDO a deterministic setting (i.e. the env var was set — a
    "re-plant"). With `warn_if_replanted=True`, also emits a one-time RuntimeWarning on a
    re-plant — used as the inference-time canary so a regression is loud, not silent."""
    replanted = os.environ.pop(_DET_ENV, None) is not None
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if replanted and warn_if_replanted:
        global _replant_warned
        if not _replant_warned:
            _replant_warned = True
            warnings.warn(
                f"[graft.perf] {_DET_ENV} was set again at inference time. Some import "
                "re-planted the deterministic-cuBLAS flag, which roughly doubles the cost "
                "of every GEMM. It has been popped so this run stays fast; to fix it "
                "properly, call enable_fast_gemm() immediately after whichever import set it.",
                RuntimeWarning, stacklevel=2,
            )
    return replanted
