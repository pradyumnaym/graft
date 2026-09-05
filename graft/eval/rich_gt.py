"""GT SMPL-X recovery for RICH: fit SMPL-X to a per-frame ground-truth mesh."""
from __future__ import annotations

import numpy as np
import smplx
import torch
import trimesh

from ..ops.geometry import aa_to_rotmat


class RichGtFitter:
    """Wraps smplfitter.BodyFitter + smplx.SMPLXLayer to recover GT joints/verts."""

    def __init__(self, smplx_layer: smplx.SMPLXLayer, device: str = "cuda"):
        from smplfitter.pt import BodyFitter, BodyModel

        bodymodel = BodyModel("smplx", "neutral", num_betas=10).to(device)
        self.fitter = BodyFitter(bodymodel).to(device)
        self.smplx_layer = smplx_layer
        self.device = device

    def fit(self, gt_mesh: trimesh.Trimesh, num_iter: int = 3, beta_regularizer: float = 1.0):
        verts = (
            torch.from_numpy(np.array(gt_mesh.vertices))
            .to(self.device)
            .float()
            .unsqueeze(0)
        )
        out = self.fitter.fit(verts, num_iter=num_iter, beta_regularizer=beta_regularizer)
        posevecs = out["pose_rotvecs"]
        global_orient = aa_to_rotmat(posevecs[:, :3])
        body_pose = aa_to_rotmat(posevecs[:, 1 * 3 : 22 * 3].reshape(-1, 3)).reshape(
            1, 21, 3, 3
        )
        lhand_pose = aa_to_rotmat(posevecs[:, 25 * 3 : 40 * 3].reshape(-1, 3)).reshape(
            1, 15, 3, 3
        )
        rhand_pose = aa_to_rotmat(posevecs[:, 40 * 3 : 55 * 3].reshape(-1, 3)).reshape(
            1, 15, 3, 3
        )

        return self.smplx_layer(
            betas=out["shape_betas"][0].unsqueeze(0),
            global_orient=global_orient[0].unsqueeze(0),
            body_pose=body_pose[0].unsqueeze(0),
            left_hand_pose=lhand_pose[0].unsqueeze(0),
            right_hand_pose=rhand_pose[0].unsqueeze(0),
            transl=out["trans"][0].unsqueeze(0),
        )
