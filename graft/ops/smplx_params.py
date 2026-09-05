"""SMPL-X parameter conversions (axis-angle <-> 6D <-> rotation matrix) and the
tensor-tree helpers the forward pass uses.

Vendored verbatim (by AST source range, see tools/extract.py) from the research
tree's `utils/utils.py`.
"""
import pickle

import torch

from .. import paths
from .geometry import aa_to_rotmat, rot6d_to_rotmat, rotmat_to_rot6d, rotmat_to_aa

# Per-beta offset about which SMPL-X shape is rescaled. `scale_smplx_params`
# declares this `global` and moves it to the caller's device on first use.
with open(paths.ASSETS_DIR / 'scale_c.pkl', 'rb') as _f:
    c_scale = pickle.load(_f)


def recursive_detach(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, dict):
        return {k: recursive_detach(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_detach(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_detach(v) for v in obj)
    elif hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            setattr(obj, k, recursive_detach(v))
        return obj
    else:
        return obj


def recursive_materialize(obj):
    """Clone inference-mode tensors to normal tensors, recursing through the same nested
    containers as recursive_detach. Inference-mode tensors (produced by the frozen backbone
    under torch.inference_mode) cannot be saved for backward, so they must be materialized
    before being consumed by learnable layers in the TRAINING graph. No-op when grad is
    disabled (inference path) so the fast inference tensors are left untouched."""
    if not torch.is_grad_enabled():
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.clone() if obj.is_inference() else obj
    elif isinstance(obj, dict):
        return {k: recursive_materialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_materialize(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_materialize(v) for v in obj)
    else:
        return obj


def convert_smplx_params_to_6d(smplx_data):
    smplx_data_6d = {}
    for k, v in smplx_data.items():
        if k in ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose']:
            smplx_data_6d[k] = rotmat_to_rot6d(aa_to_rotmat(v.reshape(-1, 3))).reshape(v.shape[:-1] + (6,))
        else:
            smplx_data_6d[k] = v
    return smplx_data_6d


def convert_smplx_params_6d_to_aa(smplx_data_6d):
    # Batch all rotation keys into ONE rot6d_to_rotmat + rotmat_to_aa (concat -> convert ->
    # split). The per-key loop was launch-bound (5 calls x the tiny roma/kornia ops = ~535
    # kernels for ~53 indep 3x3 conversions); every op in the chain is per-row independent,
    # so this is bit-identical (verified torch.equal at N_h=1,2,5,17) but ~5x fewer launches:
    # the conversion drops ~25 -> ~6 ms in-pipeline.
    rot_keys = ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose']
    smplx_data = {}
    flats, meta = [], []
    for k in rot_keys:
        if k in smplx_data_6d:
            v = smplx_data_6d[k]
            f = v.reshape(-1, 6)
            flats.append(f)
            meta.append((k, f.shape[0], v.shape))
    if flats:
        big = torch.cat(flats, dim=0)
        aa_big = torch.nan_to_num(rotmat_to_aa(rot6d_to_rotmat(big)), nan=0.0)
        off = 0
        for k, n, shp in meta:
            smplx_data[k] = aa_big[off:off + n].reshape(shp[:-1] + (3,))
            off += n
    for k, v in smplx_data_6d.items():
        if k not in smplx_data:
            smplx_data[k] = v
    return smplx_data


def convert_smplx_params_6d_to_rotmat(smplx_data_6d):
    smplx_data = {}
    for k, v in smplx_data_6d.items():
        if k in ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose']:
            smplx_data[k] = rot6d_to_rotmat(v.reshape(-1, 6)).reshape(v.shape[:-1] + (3, 3))
        else:
            smplx_data[k] = v
    return smplx_data


def convert_smplx_params_aa_to_6d(smplx_data_aa):
    smplx_data_6d = {}
    for k, v in smplx_data_aa.items():
        if k in ['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose']:
            smplx_data_6d[k] = rotmat_to_rot6d(aa_to_rotmat(v.reshape(-1, 3))).reshape(v.shape[:-1] + (6,))
        else:
            smplx_data_6d[k] = v
    return smplx_data_6d


def scale_smplx_params(smplx_data, scale):
    """
    Scale SMPL-X translation parameters according to the given scale.

    Args:
        smplx_data (dict): Dictionary containing SMPL-X parameters. (N_h, ...)
        scale (torch.Tensor): Scaling factors of shape (N_h, 1).

    Returns:
        dict: Scaled SMPL-X parameters. (N_h, ...)
    """
    global c_scale

    assert len(smplx_data['transl'].shape) == 2, "Translation tensor must be of shape (N_h, 3)"
    assert scale.dim() == 2 and scale.shape[1] == 1, "Scale tensor must be of shape (N_h, 1)"

    if c_scale.device != scale.device:
        c_scale = c_scale.to(scale.device)

    updated_data = {k : v.clone() for k, v in smplx_data.items()}
    betas_out = scale * (smplx_data['betas'] + c_scale) - c_scale
    transl_out = smplx_data['transl'] * scale
    updated_data.update({'betas': betas_out, 'transl': transl_out})
    return updated_data
