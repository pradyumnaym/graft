"""Reusable RICH-sample100 evaluation.

Mirrors `evaluation.prox_eval`:

- `build_rich_context(...)` does the one-time setup (LangSAM scene cache,
  GT contact cache, BodyFitter, contact-vertex weights) and returns a
  `RichEvalContext`.
- `run_rich_eval(model, ctx, ...)` runs the per-frame loop for one model
  and returns the metrics dict.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import smplx
import torch
import trimesh  # noqa: F401
from tqdm import tqdm

from ._assets import asset
from .metrics import (
    contact_prf1,
    pa_mpjpe,
    weighted_v2s_angle_deg,
    weighted_v2s_l2,
)
from .prefetch import prefetch_iter
from .prox import load_contact_vertex_weights
from .rich import (
    RichPaths,
    RichSceneCache,
    iter_rich_frames,
    load_or_init_gt_contact_cache,
)
from .rich_gt import RichGtFitter
from .smplx_utils import make_smplx_models
from .contacts import get_contact_mask, get_contact_mask_from_pointcloud_pt
from .alignment import reconstruction_error as _re
from .scene_geometry import apply_transformation, extend_floor_plane, get_scene_normals


@dataclass
class RichEvalContext:
    """Frame-independent RICH evaluation state. Build once, reuse across checkpoints."""
    paths: RichPaths
    sam_model: Any
    scene_cache: RichSceneCache
    aa_neutral: Any
    gt_fitter: RichGtFitter
    contact_vertices_mask: np.ndarray
    contact_vertices_weights: np.ndarray
    gt_cache: dict
    gt_cache_path: str
    interpenetration_threshold: float
    contact_threshold: float


def build_rich_context(
    benchmark_root: str,
    sam_model,
    interpenetration_threshold: float = 0.15,
    contact_threshold: float = 0.04,
    flat_hand_mean: bool = True,
) -> RichEvalContext:
    """One-time setup for RICH eval. SAM is reused for per-frame person masks."""
    paths = RichPaths(benchmark_root=benchmark_root)
    scene_cache = RichSceneCache(sam_model=sam_model)
    aa_neutral, _ = make_smplx_models(flat_hand_mean=flat_hand_mean)
    smplx_layer = smplx.SMPLXLayer(
        model_path=asset("body_models", "smplx", "SMPLX_NEUTRAL.npz"), num_betas=10
    ).cuda()
    gt_fitter = RichGtFitter(smplx_layer)
    cv_mask, cv_weights = load_contact_vertex_weights()
    gt_cache, gt_cache_path = load_or_init_gt_contact_cache(
        interpenetration_threshold, contact_threshold
    )
    return RichEvalContext(
        paths=paths,
        sam_model=sam_model,
        scene_cache=scene_cache,
        aa_neutral=aa_neutral,
        gt_fitter=gt_fitter,
        contact_vertices_mask=cv_mask,
        contact_vertices_weights=cv_weights,
        gt_cache=gt_cache,
        gt_cache_path=gt_cache_path,
        interpenetration_threshold=interpenetration_threshold,
        contact_threshold=contact_threshold,
    )


from .prox_eval import default_infer_kwargs  # noqa: F401


def run_rich_eval(
    model,
    ctx: RichEvalContext,
    *,
    infer_kwargs: Optional[dict] = None,
    num_iters: int = 3,
    limit: Optional[int] = None,
    progress: bool = True,
    return_per_frame: bool = False,
    dump_path: Optional[str] = None,
) -> dict:
    """Evaluate `model` on RICH-sample100. Returns a dict of metrics; persists
    the GT contact cache if newly populated.

    If `dump_path` is given, also pickle the raw per-vertex contact classification
    and per-vertex V2S vectors (raw, as scored for RICH) so any vertex subset's
    F1/V2S can be recomputed offline. Does not change the returned metrics."""
    infer_kwargs = infer_kwargs if infer_kwargs is not None else default_infer_kwargs(model)

    all_final_joints: list[np.ndarray] = []
    all_mosh_joints: list[np.ndarray] = []
    all_pred_contacts: list[np.ndarray] = []
    all_gt_contacts: list[np.ndarray] = []
    all_pred_vecs: list[np.ndarray] = []
    all_gt_vecs: list[np.ndarray] = []
    all_frame_keys: list[str] = []
    per_frame_log: list[dict] = []
    skipped = 0

    n_frames = limit if limit is not None else 100
    iterator = prefetch_iter(
        iter_rich_frames(ctx.paths, ctx.sam_model, ctx.scene_cache, limit=limit),
        max_ahead=2,
    )
    if progress:
        iterator = tqdm(iterator, total=n_frames, desc="RICH")

    for f in iterator:
        non_human_mask = ~(f.person_mask > 0)
        with torch.inference_mode():
            try:
                outputs = model.infer(
                    {
                        "scene_image": f.scene_image,
                        "image": f.image,
                        "non_human_mask": non_human_mask,
                    },
                    num_iters=num_iters,
                    **infer_kwargs,
                )
            except (KeyError, RuntimeError) as e:
                print(f"Inference error on {f.frame_path}: {e}")
                skipped += 1
                continue

            smplx_data = outputs["smplx_data"]
            n_humans = smplx_data["betas"].shape[1]
            params = {k: v[0] for k, v in smplx_data.items()}
            smplx_out = ctx.aa_neutral(
                betas=params["betas"],
                transl=params["transl"],
                global_orient=params["global_orient"].view(n_humans, 3),
                body_pose=params["body_pose"].view(n_humans, 21, 3),
                left_hand_pose=params["left_hand_pose"].view(n_humans, 15, 3),
                right_hand_pose=params["right_hand_pose"].view(n_humans, 15, 3),
                jaw_pose=params["jaw_pose"].view(n_humans, 3),
                expression=params["expression"].view(n_humans, -1),
                leye_pose=torch.zeros((n_humans, 3), device=params["betas"].device),
                reye_pose=torch.zeros((n_humans, 3), device=params["betas"].device),
            )

            mosh_out = ctx.gt_fitter.fit(f.gt_mesh)
            mosh_joints = apply_transformation(
                mosh_out.joints[0].detach().cpu().numpy(), f.transform
            )
            mosh_verts = apply_transformation(
                mosh_out.vertices[0].detach().cpu().numpy(), f.transform
            )

            mean_gt = mosh_verts.mean(axis=0)
            mean_gt_t = torch.as_tensor(
                mean_gt, device=smplx_out.vertices.device, dtype=smplx_out.vertices.dtype
            )
            mean_preds_t = smplx_out.vertices.mean(dim=1)
            closest_idx_t = (mean_preds_t - mean_gt_t).norm(dim=-1).argmin()
            pred_verts_closest = smplx_out.vertices[closest_idx_t]
            final_joints = smplx_out.joints[closest_idx_t].detach().cpu().numpy()
            closest_idx = int(closest_idx_t.item())

            scene_points = outputs["pts3d"][0].reshape(-1, 3)
            pts3d_np = scene_points.detach().cpu().numpy().astype(np.float64)
            scene_normals = get_scene_normals(pts3d_np, device=scene_points.device)

            plane_points, plane_normal, _ = extend_floor_plane(
                scene_points, scene_normals, f.floor_mask_np
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
                pred_verts_closest,
                return_vecs=True,
                interpenetration_threshold=ctx.interpenetration_threshold,
                threshold=ctx.contact_threshold,
            )

        cache_key = f.frame_path
        if cache_key in ctx.gt_cache:
            gt_contact_mask, gt_vecs = ctx.gt_cache[cache_key]
        else:
            gt_contact_mask, gt_vecs = get_contact_mask(
                f.gt_scene_mesh,
                mosh_verts,
                threshold=ctx.contact_threshold,
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

        if return_per_frame:
            per_frame_log.append(
                {
                    "frame": Path(f.frame_path).name,
                    "camera": f.camera_id,
                    "n_pred": int(smplx_data["betas"].shape[1]),
                    "closest_idx": closest_idx,
                    "pa_mpjpe": float(_re(final_joints[None], mosh_joints[None])[0] * 1000),
                    "mpjpe": float(
                        np.linalg.norm(final_joints - mosh_joints, axis=-1).mean() * 1000
                    ),
                }
            )

    if not os.path.exists(ctx.gt_cache_path):
        with open(ctx.gt_cache_path, "wb") as fh:
            pickle.dump(ctx.gt_cache, fh)

    if not all_final_joints:
        out = {"frames": 0, "skipped": skipped}
        if return_per_frame:
            out["per_frame"] = per_frame_log
        return out

    pa = pa_mpjpe(np.stack(all_final_joints), np.stack(all_mosh_joints))
    contact = contact_prf1(all_gt_contacts, all_pred_contacts)
    v2s_l2 = weighted_v2s_l2(
        all_pred_vecs, all_gt_vecs, ctx.contact_vertices_mask, ctx.contact_vertices_weights
    )
    v2s_angle = weighted_v2s_angle_deg(
        all_pred_vecs, all_gt_vecs, ctx.contact_vertices_mask, ctx.contact_vertices_weights
    )

    if dump_path:
        with open(dump_path, "wb") as _fh:
            pickle.dump(
                {
                    "eval": "rich",
                    "frame_keys": all_frame_keys,
                    "pred_contacts": np.stack(all_pred_contacts).astype(bool),
                    "gt_contacts": np.stack(all_gt_contacts).astype(bool),
                    # RICH scores vecs raw (no scene-frame rotation); store as-scored.
                    "pred_vecs": np.stack(all_pred_vecs).astype(np.float32),
                    "gt_vecs": np.stack(all_gt_vecs).astype(np.float32),
                    "contact_vertices_mask": np.asarray(ctx.contact_vertices_mask),
                    "contact_vertices_weights": np.asarray(ctx.contact_vertices_weights),
                    "interpenetration_threshold": ctx.interpenetration_threshold,
                    "contact_threshold": ctx.contact_threshold,
                },
                _fh,
            )

    out = {
        "frames": len(all_final_joints),
        "skipped": skipped,
        "pa_mpjpe_mm": float(pa),
        "v2s_l2_mm": float(v2s_l2 * 1000),
        "v2s_angle_deg": float(v2s_angle),
        "precision": float(contact["precision"]),
        "recall": float(contact["recall"]),
        "f1": float(contact["f1"]),
    }
    if return_per_frame:
        out["per_frame"] = per_frame_log
    return out
