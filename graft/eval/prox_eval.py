"""Reusable PROX-quantitative evaluation.

Two layers:

- `build_prox_context(...)`: do the expensive setup (load PROX scene, GT cache,
  contact-vertex weights, SMPL-X helpers) **once**. Returns a `ProxEvalContext`
  that can be reused across many checkpoints.
- `run_prox_eval(model, ctx, ...)`: run the per-frame loop for a single
  already-loaded model and return the 6 headline metrics.

The CLI in `evaluate.py` is a thin wrapper around these two.
The orchestrator in `eval_all_checkpoints.py` reuses one context and calls
`run_prox_eval` per checkpoint, paying SAM/scene/GT-cache setup only once.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import trimesh  # noqa: F401  (registers the .ply loader)
from tqdm import tqdm

from ._assets import asset
from .metrics import (
    contact_prf1,
    pa_mpjpe,
    weighted_v2s_angle_deg,
    weighted_v2s_l2,
)
from .prefetch import prefetch_iter
from .prox import (
    ProxPaths,
    ProxScene,
    iter_prox_frames,
    load_contact_vertex_weights,
    load_or_init_gt_contact_cache,
    load_prox_scene,
)
from .smplx_utils import decode_mosh_smplx, decode_graft_smplx, make_smplx_models
from .contacts import get_contact_mask, get_contact_mask_from_pointcloud_pt
from .scene_geometry import apply_transformation, extend_floor_plane, get_scene_normals


@dataclass
class ProxEvalContext:
    """Frame-independent PROX evaluation state. Build once, reuse across checkpoints."""
    paths: ProxPaths
    scene: ProxScene
    aa_neutral: Any
    male_pca: Any
    contact_vertices_mask: np.ndarray
    contact_vertices_weights: np.ndarray
    gt_cache: dict
    gt_cache_path: str
    interpenetration_threshold: float


def build_prox_context(
    benchmark_root: str,
    sam_model,
    scene_image_path: str = asset("scene_image.png"),
    interpenetration_threshold: float = 0.15,
    flat_hand_mean: bool = True,
) -> ProxEvalContext:
    """One-time setup: PROX scene + floor mask (uses SAM) + GT contact cache."""
    paths = ProxPaths(benchmark_root=benchmark_root)
    scene = load_prox_scene(paths, scene_image_path=scene_image_path, sam_model=sam_model)
    aa_neutral, male_pca = make_smplx_models(flat_hand_mean=flat_hand_mean)
    cv_mask, cv_weights = load_contact_vertex_weights()
    gt_cache, gt_cache_path = load_or_init_gt_contact_cache(interpenetration_threshold)
    return ProxEvalContext(
        paths=paths,
        scene=scene,
        aa_neutral=aa_neutral,
        male_pca=male_pca,
        contact_vertices_mask=cv_mask,
        contact_vertices_weights=cv_weights,
        gt_cache=gt_cache,
        gt_cache_path=gt_cache_path,
        interpenetration_threshold=interpenetration_threshold,
    )


def default_infer_kwargs(model=None) -> dict:
    """The `infer(...)` settings both published benchmarks are run under.

    MapAnything supplies the scene geometry and MoGe is used only to put it in
    metric scale. GRAFT predicts a per-person scale on top, which is what makes
    this the right pairing; the alternative (`use_moge=True`, MoGe geometry) is a
    different model family and is not what the reported numbers used.
    """
    return {"use_moge": False, "use_moge_scale": True}


def run_prox_eval(
    model,
    ctx: ProxEvalContext,
    *,
    infer_kwargs: Optional[dict] = None,
    limit: Optional[int] = None,
    progress: bool = True,
    dump_path: Optional[str] = None,
) -> dict:
    """Evaluate `model` on PROX-quant. Returns a dict of metrics; also persists
    the GT contact cache to disk if it's been populated.

    If `dump_path` is given, also pickle the raw per-vertex contact classification
    (vertex-id space, coordinate-free) and the per-vertex V2S vectors *as scored*
    (pred aligned to scene frame, matching the metric) so any vertex subset's
    F1/V2S can be recomputed offline. Does not change the returned metrics."""
    infer_kwargs = infer_kwargs if infer_kwargs is not None else default_infer_kwargs(model)
    scene = ctx.scene

    all_final_joints: list[np.ndarray] = []
    all_mosh_joints: list[np.ndarray] = []
    all_pred_contacts: list[np.ndarray] = []
    all_gt_contacts: list[np.ndarray] = []
    all_pred_vecs: list[np.ndarray] = []
    all_gt_vecs: list[np.ndarray] = []
    all_frame_keys: list[str] = []
    skipped = 0

    n_frames = limit if limit is not None else 178
    iterator = prefetch_iter(iter_prox_frames(ctx.paths, limit=limit), max_ahead=2)
    if progress:
        iterator = tqdm(iterator, total=n_frames, desc="PROX")

    for f in iterator:
        with torch.inference_mode():
            try:
                outputs = model.infer(
                    {"image": f.image, "scene_image": scene.scene_image},
                    **infer_kwargs,
                )
            except (KeyError, RuntimeError) as e:
                print(f"Inference error on {f.recording}/{f.frame}: {e}")
                skipped += 1
                continue

            smplx_data = outputs["smplx_data"]
            if smplx_data["betas"].shape[0] != 1 or smplx_data["betas"].shape[1] != 1:
                # only single-human case (matches the reference notebook)
                skipped += 1
                continue

            smplx_out = decode_graft_smplx(smplx_data, ctx.aa_neutral)
            mosh_out = decode_mosh_smplx(f.mosh, ctx.male_pca)

            final_joints = apply_transformation(
                smplx_out.joints[0].detach().cpu().numpy(), scene.c2w
            )
            mosh_joints = apply_transformation(
                mosh_out.joints[0].detach().cpu().numpy(), scene.w2s
            )

            scene_points = outputs["pts3d"][0].reshape(-1, 3)
            pts3d_np = scene_points.detach().cpu().numpy().astype(np.float64)
            scene_normals = get_scene_normals(pts3d_np, device=scene_points.device)

            plane_points, plane_normal, _ = extend_floor_plane(
                scene_points, scene_normals, scene.floor_mask.reshape(-1)
            )

            pred_contact_mask, pred_vecs = get_contact_mask_from_pointcloud_pt(
                torch.cat([scene_points, plane_points], dim=0),
                torch.cat(
                    [
                        scene_normals,
                        plane_normal.unsqueeze(0).repeat(plane_points.shape[0], 1),
                    ],
                    dim=0,
                ),
                smplx_out.vertices[0],
                return_vecs=True,
                interpenetration_threshold=ctx.interpenetration_threshold,
            )

            cache_key = f"{f.recording}_{f.frame}"
            if cache_key in ctx.gt_cache:
                gt_contact_mask, gt_vecs = ctx.gt_cache[cache_key]
            else:
                mosh_verts_scene = apply_transformation(
                    mosh_out.vertices[0].detach().cpu().numpy(), scene.w2s
                )
                gt_contact_mask, gt_vecs = get_contact_mask(
                    scene.scene_mesh,
                    mosh_verts_scene,
                    return_vecs=True,
                    interpenetration_threshold=ctx.interpenetration_threshold,
                )
                ctx.gt_cache[cache_key] = (gt_contact_mask, gt_vecs)

            all_final_joints.append(final_joints)
            all_mosh_joints.append(mosh_joints)
            all_pred_contacts.append(np.asarray(pred_contact_mask))
            all_gt_contacts.append(np.asarray(gt_contact_mask))
            all_pred_vecs.append(np.asarray(pred_vecs))
            all_gt_vecs.append(np.asarray(gt_vecs))
            all_frame_keys.append(cache_key)

    # Persist GT cache if it didn't exist on disk yet.
    if not os.path.exists(ctx.gt_cache_path):
        with open(ctx.gt_cache_path, "wb") as fh:
            pickle.dump(ctx.gt_cache, fh)

    if not all_final_joints:
        return {"frames": 0, "skipped": skipped}

    pa = pa_mpjpe(np.stack(all_final_joints), np.stack(all_mosh_joints))
    contact = contact_prf1(all_gt_contacts, all_pred_contacts)
    cam_to_scene_R = scene.c2w[:3, :3]
    pred_vecs_aligned = [vv @ cam_to_scene_R.T for vv in all_pred_vecs]
    v2s_l2 = weighted_v2s_l2(
        pred_vecs_aligned, all_gt_vecs, ctx.contact_vertices_mask, ctx.contact_vertices_weights
    )
    v2s_angle = weighted_v2s_angle_deg(
        pred_vecs_aligned, all_gt_vecs, ctx.contact_vertices_mask, ctx.contact_vertices_weights
    )

    if dump_path:
        with open(dump_path, "wb") as _fh:
            pickle.dump(
                {
                    "eval": "prox",
                    "frame_keys": all_frame_keys,
                    "pred_contacts": np.stack(all_pred_contacts).astype(bool),
                    "gt_contacts": np.stack(all_gt_contacts).astype(bool),
                    # pred vecs are stored ALIGNED to the scene frame (as scored); gt raw.
                    "pred_vecs": np.stack(pred_vecs_aligned).astype(np.float32),
                    "gt_vecs": np.stack(all_gt_vecs).astype(np.float32),
                    "contact_vertices_mask": np.asarray(ctx.contact_vertices_mask),
                    "contact_vertices_weights": np.asarray(ctx.contact_vertices_weights),
                    "interpenetration_threshold": ctx.interpenetration_threshold,
                },
                _fh,
            )

    return {
        "frames": len(all_final_joints),
        "skipped": skipped,
        "pa_mpjpe_mm": float(pa),
        "v2s_l2_mm": float(v2s_l2 * 1000),
        "v2s_angle_deg": float(v2s_angle),
        "precision": float(contact["precision"]),
        "recall": float(contact["recall"]),
        "f1": float(contact["f1"]),
    }
