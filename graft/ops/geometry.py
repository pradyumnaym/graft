"""Geometry helpers used by the GRAFT forward pass.

Vendored verbatim (by AST source range, see tools/extract.py) from the research
tree's `utils/geometry.py`, so the released model computes exactly what the
published numbers were produced with. Only the imports below are new: the
research module pulls in open3d / trimesh / pytorch3d for functions GRAFT's
inference path never touches, and those dependencies are dropped here.
"""
from typing import Optional, Tuple

import roma
import torch
import torch.nn.functional as F


def get_nearest_points_and_normals(
    pts3d,              # B x H x W x 3  (structured)  OR  B x M x 3 (flat pool)
    query_points,       # B x N x 3
    normals,            # same shape as pts3d
    cull_margin=0.5,    # kept for API compat; unused (cdist makes culling unnecessary)
) -> tuple[torch.Tensor, torch.Tensor]:   # nearest_points: B x N x 3, nearest_normals: B x N x 3
    # K=1 nearest neighbour via torch.cdist + argmin. Benchmarked vs pytorch3d
    # knn_points on the real ~200k-point pool (B=5, N=58): 0.66 ms vs 24 ms (36x),
    # bit-identical NN (parity-verified). pytorch3d's K=1 path was the entire KNN
    # bottleneck; culling it only added argsort overhead, and random subsampling both
    # lost to cdist AND corrupted the tail NN. cdist materialises an (B,N,P) distance
    # matrix (~0.3 GB here); for very large pools, bbox-cull or chunk.
    B = pts3d.shape[0]
    pts = pts3d.reshape(B, -1, 3)
    norm = normals.reshape(B, -1, 3)
    idx = torch.cdist(query_points, pts).argmin(-1)            # (B, N) exact NN
    batch_idx = torch.arange(B, device=pts.device).unsqueeze(-1)
    return pts[batch_idx, idx], norm[batch_idx, idx]


def rotmat_to_rot6d(R):
    """
    Converts a batch of 3x3 rotation matrices into a 6D continuous rotation representation.
    This function takes as input a tensor of rotation matrices and returns a tensor
    where each 3x3 rotation matrix is represented by its first two columns, concatenated
    to form a 6-dimensional vector. This 6D representation is often used to facilitate
    learning in neural networks by avoiding discontinuities present in other representations
    of rotations.
    Parameters:
        R (torch.Tensor): A tensor of shape (..., 3, 3) containing rotation matrices.
                          The tensor can have additional dimensions for batching (e.g., (B, N, 3, 3)).
    Returns:
        torch.Tensor: A tensor of shape (..., 6) where each 6-dimensional vector is constructed
                      by concatenating the first two columns of the corresponding 3x3 rotation matrix.
    """

    return torch.concatenate([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot6d_to_rotmat(rots: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Args:
        x (torch.Tensor): (B,6) or (B, N, 6) Batch of 6-D rotation representations.
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B,3,3).
    """
    x = rots.view(-1, 6)
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    rotmats = roma.special_gramschmidt(x, epsilon=1e-6)
    # x = x.reshape(-1, 2, 3).contiguous()
    # a1 = x[:, 0, :]
    # a2 = x[:, 1, :]
    # b1 = F.normalize(a1)
    # b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    # b3 = torch.cross(b1, b2, dim=-1)
    # rotmats = torch.stack((b1, b2, b3), dim=-1)

    if rots.dim() == 3:
        rotmats = rotmats.view(rots.shape[0], rots.shape[1], 3, 3)

    return rotmats


def rotmat_to_aa(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def project_points_torch_batch(x, K):
    """
    This function computes the perspective projection of a set of points assuming the extrinsinc params have already been applied
    Args:
        - x [bs,N,3]: 3D points
        - K [bs,3,3]: Camera instrincs params
    """
    # Apply perspective distortion
    y = x / x[:, :, -1].unsqueeze(-1)  # (bs, N, 3)

    # Apply camera intrinsics
    y = torch.einsum('bij,bkj->bki', K, y)  # (bs, N, 3)

    return y[:, :, :2]


def pointmap_to_intrinsics_batch(pts3d: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Batched variant of ``pointmap_to_intrinsics``.

    Parameters:
        pts3d (torch.Tensor): Tensor of shape (B, H, W, 3) containing 3D points in camera coordinates.
        mask (torch.Tensor, optional): Optional boolean mask of shape (B, H, W) indicating valid points.

    Returns:
        torch.Tensor: Tensor of shape (B, 3, 3) with per-sample intrinsic matrices.
    """

    if pts3d.ndim != 4 or pts3d.shape[-1] != 3:
        raise ValueError(
            f"Expected pts3d to have shape (B, H, W, 3); got {tuple(pts3d.shape)}"
        )

    B, H, W, _ = pts3d.shape
    device, dtype = pts3d.device, pts3d.dtype

    u = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
    v = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)

    cx = torch.tensor((W - 1) / 2.0, device=device, dtype=dtype)
    cy = torch.tensor((H - 1) / 2.0, device=device, dtype=dtype)

    x = pts3d[..., 0]
    y = pts3d[..., 1]
    z = pts3d[..., 2]

    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z)
    if mask is not None:
        if mask.shape != (B, H, W):
            raise ValueError(
                f"Mask must have shape (B, H, W); got {tuple(mask.shape)}"
            )
        valid = valid & mask.bool()

    eps = 1e-6
    diff_u = u - cx
    diff_v = v - cy

    mask_x = (torch.abs(diff_u) > eps) & (torch.abs(x) > eps) & valid
    mask_y = (torch.abs(diff_v) > eps) & (torch.abs(y) > eps) & valid

    denom_x = torch.where(mask_x, x, torch.ones_like(x))
    denom_y = torch.where(mask_y, y, torch.ones_like(y))

    fx_values = torch.abs(diff_u * z / denom_x)
    fy_values = torch.abs(diff_v * z / denom_y)

    nan_tensor = torch.full_like(fx_values, torch.nan)
    fx_values = torch.where(mask_x, fx_values, nan_tensor)
    fy_values = torch.where(mask_y, fy_values, nan_tensor)

    fx_flat = fx_values.view(B, -1)
    fy_flat = fy_values.view(B, -1)
    has_fx = mask_x.view(B, -1).any(dim=1)
    has_fy = mask_y.view(B, -1).any(dim=1)

    fx_median = torch.nanmedian(fx_flat, dim=1).values
    fy_median = torch.nanmedian(fy_flat, dim=1).values

    f = torch.full((B,), 1.0, device=device, dtype=dtype)

    both = has_fx & has_fy
    only_fx = has_fx & ~has_fy
    only_fy = has_fy & ~has_fx

    f[both] = 0.5 * (fx_median[both] + fy_median[both])
    f[only_fx] = fx_median[only_fx]
    f[only_fy] = fy_median[only_fy]

    intrinsics = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    intrinsics[:, 0, 0] = f
    intrinsics[:, 1, 1] = f
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    intrinsics[:, 2, 2] = 1.0

    return intrinsics


def compute_point_normals(pts3d):
    """
    Compute per-point normals for a 3D point grid.
    This function computes the normal vector at each point in a 2D grid of 3D points.
    It uses neighboring points to estimate the surface normal by taking cross-products
    of vectors formed between the point of interest and its adjacent neighbors. Four
    cross-products are computed using different pairs of immediate neighbors (top, right,
    bottom, left), normalized, and then averaged to obtain a robust normal estimate.
    Parameters:
        pts3d (torch.Tensor): A tensor of shape (H, W, 3) or (B, H, W, 3) containing 3D coordinates.
                              It can be on CPU or GPU.
    Returns:
        torch.Tensor: A tensor of shape (H, W, 3) or (B, H, W, 3) containing normalized normals.
                      For points where a valid normal cannot be computed (e.g., near boundaries or
                      where the geometry is degenerate), the normal vector is set to NaN.
    Notes:
        - The grid is padded with NaNs to facilitate easy neighbor access.
        - Normals are calculated from cross products of vectors defined between the central point
          and its four neighbors.
        - A small epsilon (1e-6) is used to avoid division by zero during normalization.
    
    """

    if pts3d.ndim == 3:
        pts3d = pts3d.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    B, H, W, _ = pts3d.shape
    eps = 1e-6
    device = pts3d.device if hasattr(pts3d, "device") else None
    normals = torch.zeros((B, H, W, 3), dtype=pts3d.dtype, device=device)

    padded = torch.full((B, H + 2, W + 2, 3), float('nan'), dtype=pts3d.dtype, device=device)
    padded[:, 1:-1, 1:-1] = pts3d

    top    = padded[:, :-2, 1:-1]
    right  = padded[:, 1:-1, 2:]
    bottom = padded[:, 2:, 1:-1]
    left   = padded[:, 1:-1, :-2]

    p = pts3d

    v_top    = top    - p
    v_right  = right  - p
    v_bottom = bottom - p
    v_left   = left   - p

    cp0 = torch.cross(v_right,    v_top, dim=3)
    cp1 = torch.cross(v_bottom,   v_right, dim=3)
    cp2 = torch.cross(v_left,     v_bottom, dim=3)
    cp3 = torch.cross(v_top,      v_left, dim=3)

    norm0 = torch.norm(cp0, dim=3)
    norm1 = torch.norm(cp1, dim=3)
    norm2 = torch.norm(cp2, dim=3)
    norm3 = torch.norm(cp3, dim=3)

    def normalize(cp, norm):
        valid = norm > eps
        cp_norm = cp.clone()
        cp_norm[valid] = cp[valid] / norm[valid].unsqueeze(-1)
        cp_norm[~valid] = float('nan')
        return cp_norm

    cp0_n = normalize(cp0, norm0)
    cp1_n = normalize(cp1, norm1)
    cp2_n = normalize(cp2, norm2)
    cp3_n = normalize(cp3, norm3)

    cp_stack = torch.stack([cp0_n, cp1_n, cp2_n, cp3_n], dim=3)
    valid_mask = ~torch.isnan(cp_stack[..., 0])
    valid_count = torch.sum(valid_mask, dim=3, keepdim=True).float()

    cp_stack[torch.isnan(cp_stack)] = 0
    avg_norm = torch.sum(cp_stack, dim=3) / torch.where(valid_count == 0, torch.tensor(1.0, device=device), valid_count)

    norm_avg = torch.norm(avg_norm, dim=3)
    valid = norm_avg > eps
    normals[valid] = avg_norm[valid] / norm_avg[valid].unsqueeze(-1)

    if squeeze_out:
        normals = normals.squeeze(0)
    return normals


def align_points_robust_batched(pred_points, target_points, mask, return_transform=False):
    """
    Batched version of ``align_points_robust``.
    Aligns 3D pointmaps with uniform scale and z-axis shift per batch element.

    Args:
        pred_points: torch.Tensor of shape [B, ..., 3]
        target_points: torch.Tensor of shape [B, ..., 3]
        mask: bool tensor of shape [B, ...] for valid points

    Returns:
        aligned_points: torch.Tensor of shape [B, ..., 3]
    """
    if pred_points.shape != target_points.shape:
        raise ValueError(f"pred_points and target_points must have the same shape, got {pred_points.shape} and {target_points.shape}")
    if pred_points.ndim < 3 or pred_points.shape[-1] != 3:
        raise ValueError(f"Expected pred_points to have shape [B, ..., 3], got {pred_points.shape}")
    if mask.shape != pred_points.shape[:-1]:
        raise ValueError(f"mask must have shape {pred_points.shape[:-1]}, got {mask.shape}")

    B = pred_points.shape[0]
    pred_flat = pred_points.reshape(B, -1, 3)
    target_flat = target_points.reshape(B, -1, 3)
    mask_flat = mask.reshape(B, -1).bool()

    finite_mask = torch.isfinite(pred_flat).all(dim=-1) & torch.isfinite(target_flat).all(dim=-1)
    valid_mask = mask_flat & finite_mask

    masked_pred = pred_flat.masked_fill(~valid_mask[..., None], float('nan'))
    masked_target = target_flat.masked_fill(~valid_mask[..., None], float('nan'))

    # Robust center per axis
    t_pr = torch.nanmedian(masked_pred, dim=1).values  # [B, 3]
    t_tar = torch.nanmedian(masked_target, dim=1).values  # [B, 3]

    # Robust uniform scale using all dimensions
    scale_pr = torch.nanmean((masked_pred - t_pr[:, None, :]).abs(), dim=(1, 2))  # [B]
    scale_tar = torch.nanmean((masked_target - t_tar[:, None, :]).abs(), dim=(1, 2))  # [B]
    scale = scale_tar / (scale_pr + 1e-8)

    valid_batch = valid_mask.any(dim=1) & torch.isfinite(scale)
    scale = torch.where(valid_batch, scale, torch.ones_like(scale))

    # z-shift only
    z_shift = t_tar[:, 2] - scale * t_pr[:, 2]
    z_shift = torch.where(valid_batch & torch.isfinite(z_shift), z_shift, torch.zeros_like(z_shift))

    scale_view = scale.view(B, *([1] * (pred_points.ndim - 1)))
    z_shift_view = z_shift.view(B, *([1] * (pred_points.ndim - 2)))

    aligned_points = pred_points * scale_view
    aligned_points[..., 2] = aligned_points[..., 2] + z_shift_view

    if return_transform:
        return aligned_points, scale, z_shift
    else:
        return aligned_points
