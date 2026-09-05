from typing import List, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .alignment import reconstruction_error
from .contact_vectors import (
    weighted_contact_vector_angle_deg,
    weighted_contact_vector_l2,
)


def pa_mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
    """Procrustes-aligned MPJPE in millimetres. pred/gt: (N, J, 3) in metres."""
    return float(reconstruction_error(pred_joints, gt_joints)[0] * 1000)


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-point L2 error in millimetres. pred/gt: (..., 3)."""
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000)


def contact_prf1(
    gt_contacts: Sequence[np.ndarray],
    pred_contacts: Sequence[np.ndarray],
    threshold: float = 0.5,
) -> dict:
    """Per-vertex contact precision/recall/F1, concatenated across frames."""
    gt = np.concatenate(gt_contacts, axis=0)
    pred = np.concatenate(pred_contacts, axis=0) > threshold
    return {
        "precision": float(precision_score(gt, pred)),
        "recall": float(recall_score(gt, pred)),
        "f1": float(f1_score(gt, pred)),
    }


def weighted_v2s_l2(
    pred_vecs: List[np.ndarray],
    gt_vecs: List[np.ndarray],
    contact_mask: np.ndarray,
    contact_weights: np.ndarray,
) -> float:
    """L2 error between pred vs GT vertex-to-scene vectors, weighted by per-vertex weights."""
    return float(
        weighted_contact_vector_l2(pred_vecs, gt_vecs, contact_mask, contact_weights)
    )


def weighted_v2s_angle_deg(
    pred_vecs: List[np.ndarray],
    gt_vecs: List[np.ndarray],
    contact_mask: np.ndarray,
    contact_weights: np.ndarray,
) -> float:
    """Angular error (degrees) between pred vs GT V2S vectors, weighted by per-vertex weights."""
    return float(
        weighted_contact_vector_angle_deg(
            pred_vecs, gt_vecs, contact_mask, contact_weights
        )
    )
