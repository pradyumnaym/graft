from typing import Dict

import numpy as np
import smplx
import torch
from ._assets import asset


def make_smplx_models(
    model_dir: str = asset("body_models", "smplx"),
    device: str = "cuda",
    flat_hand_mean: bool = True,
):
    """Build the SMPL-X model variants used in evaluation.

    - aa_neutral: SMPLX(NEUTRAL, use_pca=False) — for decoding model output (axis-angle).
      `flat_hand_mean` must match the checkpoint's training convention.
    - male_pca: SMPLX(MALE, num_pca_comps=12) — for decoding PROX MoSh GT fits.
      Always uses the SMPL-X default (mean added); PROX MoSh PCA coefficients
      are mean-relative, so the internal `pca @ comp + mean` is required.
    """
    aa_neutral = smplx.SMPLX(
        model_path=f"{model_dir}/SMPLX_NEUTRAL.npz",
        num_betas=10,
        use_pca=False,
        flat_hand_mean=flat_hand_mean,
    ).to(device)
    male_pca = smplx.SMPLX(
        model_path=f"{model_dir}/SMPLX_MALE.npz",
        num_betas=10,
        num_pca_comps=12,
    ).to(device)
    return aa_neutral, male_pca


def decode_graft_smplx(
    smplx_data: Dict[str, torch.Tensor],
    aa_model: smplx.SMPLX,
    batch_idx: int = 0,
    human_idx: int = 0,
):
    """Decode a `GRAFTModel.infer` output dict (shape: (B, N_humans, ...)) into joints/verts."""
    p = {k: v[batch_idx, human_idx] for k, v in smplx_data.items()}
    return aa_model(
        betas=p["betas"].unsqueeze(0),
        transl=p["transl"].unsqueeze(0),
        global_orient=p["global_orient"].view(1, 3),
        body_pose=p["body_pose"].view(1, 21, 3),
        left_hand_pose=p["left_hand_pose"].view(1, 15, 3),
        right_hand_pose=p["right_hand_pose"].view(1, 15, 3),
        jaw_pose=p["jaw_pose"].view(1, 3),
        expression=p["expression"].unsqueeze(0),
    )


def decode_mosh_smplx(mosh: dict, male_pca_model: smplx.SMPLX, device: str = "cuda"):
    """Run the male-PCA model on PROX MoSh fit dict (numpy/list values)."""
    return male_pca_model(
        **{k: torch.tensor(v).float().to(device) for k, v in mosh.items()}
    )
