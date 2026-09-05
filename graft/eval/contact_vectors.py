"""Weighted vertex-to-scene error, the V2S half of the reported metrics.

Vendored verbatim (by AST source range) from the research tree's
`utils/utils.py`.
"""
import numpy as np


def weighted_contact_vector_l2(pred_vecs_list, gt_vecs_list, mask, weights):
    per_frame = []
    for pred_vecs, gt_vecs in zip(pred_vecs_list, gt_vecs_list):
        pred = np.asarray(pred_vecs)[mask]  # contact vertices only
        gt = np.asarray(gt_vecs)[mask]      # contact vertices only
        l2_per_vertex = np.linalg.norm(pred - gt, axis=-1)  # (num_contact_verts,)
        per_frame.append(np.sum(l2_per_vertex * weights))   # weighted frame error
    return float(np.mean(per_frame))                        # dataset mean


def weighted_contact_vector_angle_deg(pred_vecs_list, gt_vecs_list, mask, weights, eps=1e-8):
    """
    Weighted mean angular error (in degrees) between pred and GT contact vectors.
    Mirrors weighted_contact_vector_l2 usage.
    """
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / (w.sum() + 1e-12)  # normalize once

    per_frame = []
    for pred_vecs, gt_vecs in zip(pred_vecs_list, gt_vecs_list):
        pred = np.asarray(pred_vecs)[mask]   # (Nc, 3)
        gt   = np.asarray(gt_vecs)[mask]     # (Nc, 3)

        pred_n = np.linalg.norm(pred, axis=-1)
        gt_n   = np.linalg.norm(gt, axis=-1)
        denom  = np.clip(pred_n * gt_n, eps, None)

        cosang = np.sum(pred * gt, axis=-1) / denom
        cosang = np.clip(cosang, -1.0, 1.0)

        ang_deg = np.degrees(np.arccos(cosang))   # (Nc,)
        per_frame.append(np.sum(ang_deg * w))     # weighted frame error

    return float(np.mean(per_frame))
