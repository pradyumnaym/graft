"""fast SMPL-X body fitting for the NLF stage.

Replaces NLF's `detect_parametric_batched` / `estimate_parametric_batched` (which run
the embedded body-fitter, ~80-155 ms) with `detect_poses_batched` / `estimate_poses_batched`
(the non-parametric NLF net) + an external, torch.compiled, batched-SVD body fit on NLF's
own 1024-vertex SMPL-X subset. Output matches NLF's parametric path (mean-centered fit,
NLF's own per-point fit weighting, num_iter=3, beta_reg=10, final_adjust_rots), so
downstream is unchanged; only ambiguous finger joints can differ.

All state is built once and cached on the model (`model._ext[...]`), so `GRAFTModel`
gains only a one-line call where it would otherwise invoke NLF's parametric method.
"""
from __future__ import annotations

import numpy as np
import torch
from smplfitter.pt import BodyModel, BodyFitter

from .. import paths

# SMPL-X compile needs inductor's auto-cudagraphs OFF (the fit's linalg.svd/cholesky
# sync and break CUDA-graph capture); plain inductor fusion still gives the speedup.
import torch._inductor.config as _ind
_ind.triton.cudagraphs = False

_NLF_N_VERTS, _NLF_N_JOINTS = 1024, 55   # NLF's SMPL-X subset: cano_all['smplx'] = 1079 pts
_FIT_PAD = 16   # pad the fit's human/batch dim to this fixed size so the compiled
                # fit never recompiles on a new person count (one compile, no 40-50s stalls)


class BatchedBodyFitter(BodyFitter):
    """BodyFitter whose initial global-rotation fit batches the per-joint kabsch SVDs
    into ONE batched SVD (num_joints launches -> 1). ~5x faster on H100; body-exact,
    only rank-deficient finger joints (which WiLoR refines anyway) can differ."""

    def _fit_global_rotations(self, target_vertices, target_joints, reference_vertices,
                              reference_joints, vertex_weights, joint_weights):
        device = target_vertices.device
        mw = torch.full((), 1e-6, device=device, dtype=torch.float32)
        jw_s = 1.0 - mw
        if target_joints is None or reference_joints is None:
            target_joints = self.body_model.J_regressor_post_lbs @ target_vertices
            reference_joints = self.body_model.J_regressor_post_lbs @ reference_vertices
        A = []
        for i in range(self.body_model.num_joints):
            sel = self.part_vertex_selectors[i]
            js = self.children_and_self[i]
            dbp = reference_vertices[:, sel]; ebp = target_vertices[:, sel]
            wbp = (vertex_weights[:, sel].unsqueeze(-1) * mw) if vertex_weights is not None else mw
            dj = reference_joints[:, js]; ej = target_joints[:, js]
            wj = (joint_weights[:, js].unsqueeze(-1) * jw_s) if joint_weights is not None else jw_s
            mref = dj.mean(1, keepdim=True); mtgt = ej.mean(1, keepdim=True)
            dpts = torch.cat([(dbp - mref) * wbp, (dj - mref) * wj], 1)
            epts = torch.cat([(ebp - mtgt), (ej - mtgt)], 1)
            A.append(epts.mT @ dpts)                                   # (B,3,3)
        A = torch.stack(A, 1)                                          # (B,J,3,3)
        U, _, Vh = torch.linalg.svd(A)                                 # ONE batched SVD
        T = U @ Vh
        refl = (torch.det(T) < 0).unsqueeze(-1).unsqueeze(-1)
        R = torch.where(refl, T - 2 * U[..., -1:] @ Vh[..., -1:, :], T)
        if self.is_smpl_family:                                        # toes copy feet
            R = R.clone(); R[:, 10] = R[:, 7].clone(); R[:, 11] = R[:, 8].clone()
        return R


def _require_fast_nlf(model, device):
    """Build (once) the NLF-subset BodyFitter + compiled fit fn + NLF weights, cached
    on the model. Mirrors NLF's internal `_estimate_parametric_batched` setup."""
    if model._ext.get("fast_fit_fn") is not None:
        return
    nlf = model._ext["model_nlf"]
    nV, nJ = _NLF_N_VERTS, _NLF_N_JOINTS
    cano = nlf.cano_all["smplx"].to(device)                            # (1079, 3)
    assert cano.shape == (nV + nJ, 3), f"unexpected cano_all shape {tuple(cano.shape)}"

    # Recover NLF's exact 1024-vertex subset straight from the checkpoint: its body
    # fitter carries the subset's T-pose template and its SMPL-X model carries the
    # full one, and the subset is a strict subset of the full mesh. Matching T-pose
    # to T-pose therefore recovers the indices exactly, using only tensors from this
    # same checkpoint -- so there is nothing to download and nothing that can drift.
    from scipy.spatial import cKDTree
    full_T = nlf.body_models.smplx.v_template.detach().cpu().numpy().astype(np.float32)
    sub_T = nlf.fitters.smplx.body_model.v_template.detach().cpu().numpy().astype(np.float32)
    d, idx = cKDTree(full_T).query(sub_T, k=1)
    if d.max() > 1e-4 or len(set(idx)) != nV:
        raise RuntimeError(f"fast_nlf subset recovery failed (max NN {d.max():.2e}m, "
                           f"unique {len(set(idx))}/{nV})")

    bm = BodyModel("smplx", "neutral", num_betas=10,
                   model_root=str(paths.body_model_dir()), vertex_subset=list(idx)).to(device)
    fitter = BatchedBodyFitter(bm).to(device)
    # children_and_self ships as python lists -> indexing a cuda tensor with them is an
    # H2D copy each call; pre-convert to gpu index tensors (also keeps the graph clean).
    fitter.children_and_self = [torch.as_tensor(c, device=device, dtype=torch.long)
                                for c in fitter.children_and_self]
    with torch.inference_mode():
        weights = nlf.get_weights_for_canonical_points(cano)

    def _fit(tv, tj, vw, jw):
        return fitter.fit(tv, tj, vw, jw, num_iter=3, beta_regularizer=10.0,
                          beta_regularizer2=0.0, final_adjust_rots=True,
                          requested_keys=["pose_rotvecs", "shape_betas", "trans"])

    model._ext["fast_fit_fn"] = torch.compile(_fit)
    model._ext["fast_nlf_weights"] = weights


def _assemble(model, poses_out, device):
    """Mean-center, fit (compiled batched), and split into the smplx_params dict list
    that `nlf_preds_to_smplx_params` would have produced (so callers are unchanged)."""
    fit_fn = model._ext["fast_fit_fn"]
    nV, nJ = _NLF_N_VERTS, _NLF_N_JOINTS
    boxes_list = poses_out["boxes"]
    if sum(int(b.shape[0]) for b in boxes_list) == 0:
        raise RuntimeError("fast_nlf: no humans detected in any image")

    out = []
    for i in range(len(poses_out["poses3d"])):
        p3d = poses_out["poses3d"][i]                 # (n, 1079, 3) mm
        unc = poses_out["uncertainties"][i]           # (n, 1079)
        boxes = boxes_list[i]
        n = p3d.shape[0]
        if n == 0:
            z = lambda *d: torch.zeros(0, *d, device=device)
            out.append({"boxes": boxes, "transl": z(3), "betas": z(10), "global_orient": z(3),
                        "body_pose": z(21, 3), "left_hand_pose": z(15, 3), "right_hand_pose": z(15, 3)})
            continue
        mean = p3d.mean(-2, keepdim=True)             # mean-center (matches NLF internal)
        p3dc = p3d - mean
        tv = (p3dc[:, :nV] / 1000.0).float(); tj = (p3dc[:, nV:] / 1000.0).float()
        uv = unc[:, :nV].float(); uj = unc[:, nV:].float()
        vw = uv.pow(-1.5); vw = vw / vw.mean(-1, keepdim=True)
        jw = uj.pow(-1.5); jw = jw / jw.mean(-1, keepdim=True)
        # Pad the human (batch) dim to a fixed size so the compiled fit never recompiles on a
        # new person count (torch.compile keys on shape; a new count = a ~40-50s inductor recompile).
        # n>PAD pads up to the next multiple (never truncates). NOTE: this changes the batch the
        # batched SVD/cholesky see, so it is metric-equivalent (verified) but NOT bit-identical
        # (batch-FP, same class as the MA-batch effect); fingers also re-roll the SVD ambiguity.
        pad = _FIT_PAD * ((n + _FIT_PAD - 1) // _FIT_PAD)
        if pad > n:
            fill = lambda x: torch.cat([x, x[:1].expand(pad - n, *x.shape[1:])], 0)
            tv, tj, vw, jw = fill(tv), fill(tj), fill(vw), fill(jw)
        with torch.inference_mode():
            fit = fit_fn(tv, tj, vw, jw)
        pr = fit["pose_rotvecs"][:n]
        trans = fit["trans"][:n] + mean.squeeze(-2) / 1000.0
        out.append({
            "boxes": boxes,
            "transl": trans,
            "betas": fit["shape_betas"][:n],
            "global_orient": pr[:, 0:3],
            "body_pose": pr[:, 3:66].reshape(n, 21, 3),
            "left_hand_pose": pr[:, 75:120].reshape(n, 15, 3),
            "right_hand_pose": pr[:, 120:165].reshape(n, 15, 3),
        })
    return out


def detect_parametric_fast(model, image, intrinsic_matrix):
    """Inference path: detect_poses_batched (non-param) + compiled batched fit. Returns
    the smplx_params list (already in nlf_preds_to_smplx_params output format)."""
    _require_fast_nlf(model, image.device)
    nlf = model._ext["model_nlf"]
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        poses_out = nlf.detect_poses_batched(
            image, weights=model._ext["fast_nlf_weights"],
            intrinsic_matrix=intrinsic_matrix, suppress_implausible_poses=True)
    return _assemble(model, poses_out, image.device)


def estimate_parametric_fast(model, image, boxes, intrinsic_matrix):
    """Train path: estimate_poses_batched (non-param, GT boxes) + compiled batched fit."""
    _require_fast_nlf(model, image.device)
    nlf = model._ext["model_nlf"]
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        poses_out = nlf.estimate_poses_batched(
            image, boxes=boxes, weights=model._ext["fast_nlf_weights"],
            intrinsic_matrix=intrinsic_matrix)
    if "boxes" not in poses_out:
        poses_out["boxes"] = boxes
    return _assemble(model, poses_out, image.device)
