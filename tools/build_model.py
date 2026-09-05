"""Generate `graft/model.py` from the research tree's `SceneFitScale`.

Every numerical method is copied VERBATIM by AST source range, so the released
model computes exactly what produced the published numbers. This script exists so
that provenance is auditable and re-runnable rather than a one-off hand-port.

What changes, and why:

  * `SceneFitScale` -> `GRAFTModel`, and the module scaffold (imports, asset
    loading, __init__) is rewritten against `graft.paths`.
  * `process_input` loses its training-time branch (`'boxes_xywh' in batch`,
    i.e. ground-truth person boxes and the WiLoR hand initialisation that rides
    with them). Inference never takes it. The kept else-body is asserted
    character-identical after dedent.
  * `process_input` also loses an env-var-gated debug print (`SCENEFIT_DEBUG_NLFBAD`).
  * Dropped members: `require_renderer` (the depth renderer was replaced by the
    head ray-anchor and is unreferenced), `scale_smplx_data_with_scale` (dead --
    it calls a `fit_smplx_to_vertices_joints` that does not exist on the class).
  * Dropped from __init__: the TorchScript `BodyFitter` stored at
    `_ext['fitter']`, which nothing ever reads and which costs seconds of
    `torch.jit.script` at construction.
  * Added: `from_pretrained`, `predict`, and the `GRAFTConfig` dataclass.

Run:  python tools/build_model.py
"""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import methods  # noqa: E402
from scrub import scrub  # noqa: E402

SRC = Path(__file__).parent.parent.parent / "network" / "scenefitter.py"
OUT = Path(__file__).parent.parent / "graft" / "model.py"
CLS = "SceneFitScale"

def get(name):
    """Verbatim source of one method of the research class."""
    return methods.method_source(str(SRC), CLS, name)


def build_process_input():
    """process_input with the training-time (ground-truth boxes) branch removed."""
    s = get("process_input")
    s = methods.drop_if_keep_else(s, "'boxes_xywh' in batch")   # training branch
    s = methods.drop_if(s, "'boxes_xywh' in batch")             # WiLoR GT-hand init
    # A debug print gated on an env var; the only reason `os` was imported.
    s = methods.drop_if(s, "SCENEFIT_DEBUG_NLFBAD")
    # The NLF-uncertainty surfacing exists for a video model that is not part of
    # this release; nothing here reads `batch['nlf_uncertainty']`.
    s = s.replace(UNCERTAINTY_COMMENT, "")
    s = methods.drop_if(s, "'uncertainty' in smplx_params_nlf[0]")
    assert "uncertainty" not in s
    assert "boxes_xywh" not in s and "estimate_parametric_fast" not in s
    assert "os.environ" not in s
    return s


UNCERTAINTY_COMMENT = """
        # Surface per-human NLF uncertainty as a batch key (B, N_h, 1079: 1024 verts + 55
        # joints; zeros on padded slots). Present only on the fast-NLF path (the slow
        # `nlf_preds_to_smplx_params` path carries no 'uncertainty'). Consumer: the
        # uncertainty-conditioned temporal write gate (gate_uncertainty_input); mirrors the
        # vgraft input-smoother surfacing (PATCHES.md #4). Additive: unread unless enabled.
"""[1:]

# Comments that describe machinery outside this release (the video/temporal model).
COMMENT_EDITS = [
    ("""                # Scatter by MASK, not into the first-n SLOTS. Every consumer reads
                # these back with [batch['valid_humans']] (e.g. :432), so a prefix
                # write is only correct when the valid mask is contiguous. The image
                # collate guarantees that (dataloaders/hsi.py:280) but the VIDEO path
                # does not -- dataloaders/video.py:388 assigns humans to clip slots by
                # Hungarian matching, so a frame missing a person yields a holed mask.
                # The old prefix write then left the gathered row NEVER WRITTEN: an
                # all-zero human (betas/transl/global_orient = 0) = T-pose at the
                # camera origin, which also poisons human_origins (:490) and hence the
                # query tokens of BOTH branches. Measured 0/1100 frames on
                # bedlam2/mamma/rich but 1.25% on mammasyn (HH lineage).""",
     """                # Scatter by MASK, not into the first-n slots: every consumer reads these
                # back with [batch['valid_humans']], so a prefix write is only correct when
                # the valid mask happens to be contiguous. An unwritten row would be an
                # all-zero human -- a T-pose at the camera origin -- which then also
                # poisons human_origins and hence the query tokens of both branches."""),
    ("""        # The scaled vertices/joints feed only the training loss (and temporal's mesh output);
        # skip this SMPLX forward when the caller doesn't need them (base inference reads only the
        # returned params). Verified bit-identical to computing-then-discarding.""",
     """        # The scaled vertices/joints feed only the training loss; skip this SMPL-X forward
        # when the caller doesn't need them (inference reads only the returned params).
        # Verified bit-identical to computing-then-discarding."""),
    ("""        # Cache per-level feature projection ONCE (iteration-invariant), lazily here rather than
        # in process_input: this path is SceneFitScale-only, so the temporal subclass (frozen base
        # under inference_mode, grad rollout via _update_branches) never computes it -- avoiding
        # both wasted compute and an inference-tensor/BPTT hazard. Bit-identical (Linear commutes).""",
     """        # Cache the per-level feature projection ONCE: it does not change across refinement
        # iterations. Bit-identical to projecting every iteration, since the projection is a
        # Linear and commutes with the gather."""),
]


# Emission plan. ("v", name) copies a method verbatim from the research class;
# ("t", file)   splices a hand-written template (the parts that had to change);
# ("p", None)   emits the edited process_input.
PLAN = [
    ("t", "model_scaffold.py.in"),   # GRAFTConfig / GRAFTOutput / class + __init__
    ("v", "train"),
    ("v", "to"),
    ("v", "cuda"),
    ("v", "cpu"),
    ("t", "model_require.py.in"),    # require_moge / require_nlf / require_wilor (repathed)
    ("v", "get_hand_pose"),
    ("p", None),                     # process_input, training branch removed
    ("v", "update_parameters"),
    ("v", "call_update_param"),
    ("v", "forward"),
    ("v", "infer"),
    ("t", "model_api.py.in"),        # from_pretrained / predict
]


HEADER = '''"""GRAFT: Geometric Refinement and Fitting Transformer for Human Scene Reconstruction.

`GRAFTModel` is a learned human-scene-interaction prior. It takes an off-the-shelf
single-frame pose estimate (NLF) and a metric point map of the scene (MapAnything,
optionally rescaled by MoGe), and predicts iterative, geometry-grounded
refinements to pose, shape, translation and a per-person scale, so the
reconstruction agrees with the scene geometry the body is in contact with.

Two published checkpoints load into this class: ``graft``, the main model, and
``graft-no-visual``, the "without visual features" ablation -- the same
architecture with ``drop_vis_feats=True``, so the transformer sees scene geometry
only.

Quick start::

    from graft import GRAFTModel

    model = GRAFTModel.from_pretrained("graft").cuda().eval()
    out = model.predict(image, scene_image=empty_scene)
    out.vertices                               # (n_people, 10475, 3), scene frame
    out.smplx                                  # dict of SMPL-X parameters
    out.points                                 # (H, W, 3) scene point map

This file is GENERATED by ``tools/build_model.py``; the numerical methods are
copied verbatim from the research implementation so that the release reproduces
the published numbers exactly. Edit the generator, not this file.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import smplx
import torch

from . import paths
from ._optional import optional
from ._vendor import MapAnything
from .nets.feature_sampler import FusedFeatureSampler
from .nets.layers import MLP, LearnableFourierFeatures
from .nets.transformer import TransformerCrossAttn
from .ops.fast_fitter import detect_parametric_fast
from .ops.geometry import (
    align_points_robust_batched,
    compute_point_normals,
    get_nearest_points_and_normals,
    pointmap_to_intrinsics_batch,
    project_points_torch_batch,
)
from .ops.head_anchor import head_anchor_points
from .ops.images import ImageLike, prepare_image, to_pil
from .ops.joint_names import JOINT_NAMES
from .ops.perf import enable_fast_gemm
from .ops.smplx_params import (
    convert_smplx_params_6d_to_aa,
    convert_smplx_params_6d_to_rotmat,
    convert_smplx_params_to_6d,
    recursive_detach,
    recursive_materialize,
    scale_smplx_params,
)

# `wilor_mini` is imported lazily in require_wilor(): importing it pulls in ultralytics,
# which plants os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8" (deterministic-slow cuBLAS).
# Pop it here so the whole process runs fast regardless of import order.
enable_fast_gemm()
paths.export_smplfitter_data_root()

hand_joint_names = ['_index3', '_middle3', '_ring3', '_pinky3', '_thumb3']
all_hand_joint_names = ['_wrist']
all_hand_joint_names += [x + str(i) for i in range(1, 4) for x in ['_index', '_middle', '_ring', '_pinky', '_thumb']]

# Sampled SMPL-X surface points that become the model's query/context tokens.
with open(paths.ASSETS_DIR / 'smplx_sampled_points_27.pkl', 'rb') as f:
    sampled_point_indices = pickle.load(f)

with open(paths.ASSETS_DIR / 'smplx_sampled_points_64.pkl', 'rb') as f:
    sampled_point_indices_64 = pickle.load(f)
'''

with open(OUT, "w") as fh:
    fh.write(HEADER)
    for kind, what in PLAN:
        if kind == "t":
            fh.write(Path(__file__).parent.joinpath(what).read_text())
        elif kind == "v":
            fh.write(get(what) + "\n\n")
        else:
            fh.write(build_process_input() + "\n\n")

# Docstrings for the two methods the research code left bare.
DOCSTRINGS = [
    ("    def forward(self, batch, num_iters=1, noisy_align=True, use_moge=True, use_moge_scale=False, skip_last_fit=True, current_smplx_params=None, use_wilor_hands=False):\n",
     "    def forward(self, batch, num_iters=1, noisy_align=True, use_moge=True, use_moge_scale=False, skip_last_fit=True, current_smplx_params=None, use_wilor_hands=False):\n"
     '        """Run the refinement loop over a prepared batch.\n\n'
     "        `batch` needs `image` and `scene_image`, each (B, 3, H, W) float in [0, 1];\n"
     "        everything else is filled in by `_process_input`. Each iteration re-samples\n"
     "        scene geometry and image features at the current body's joints and surface\n"
     "        points, and predicts a residual on top.\n\n"
     "        Returns the raw per-iteration dict. Most callers want `predict` (images in,\n"
     '        meshes out) or `infer` (batch in, parameters out in a stable layout)."""\n'),
    ("    def infer(self, batch, num_iters=3, noisy_align=False, use_moge=True, use_moge_scale=False, current_smplx_params=None, use_wilor_hands=False):\n",
     "    def infer(self, batch, num_iters=3, noisy_align=False, use_moge=True, use_moge_scale=False, current_smplx_params=None, use_wilor_hands=False):\n"
     '        """Run inference on a prepared batch and return SMPL-X parameters.\n\n'
     "        The batch-dictionary entry point, used by the benchmark harness. `predict`\n"
     "        wraps this with image loading and mesh posing.\n\n"
     "        Returns a dict with `smplx_data` (each value (B, N_people, ...), valid\n"
     "        entries selected by `valid_humans`), the NLF initialisation it started\n"
     "        from, the scene point map `pts3d`, the intrinsics `K`, and the parameters\n"
     '        after each intermediate iteration."""\n'),
]

# Rewrite the comments that describe machinery outside this release.
text = OUT.read_text()
for old, new in COMMENT_EDITS + DOCSTRINGS:
    assert old in text, f"source changed upstream:\n{old}"
    text = text.replace(old, new)
OUT.write_text(scrub(text))

print(f"wrote {OUT} ({len(OUT.read_text().splitlines())} lines)")
