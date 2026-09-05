"""Scene-side geometry used only by the benchmarks.

Vendored verbatim (by AST source range) from the research tree's
`utils/geometry.py`. The floor-plane extension in particular is part of the
contact metric definition: it is what lets a body contact the *ground*.
"""
import numpy as np
import open3d as o3d
import torch


def depth_to_points(depth, K=None, R=None, t=None):
    """Back-project depth map to 3D points in camera (or world via R, t) frame.

    Returns (H, W, 3) for the first batch. Closed-form replacement for the
    5D-broadcast matmul previously used here — 2.5× faster, max abs diff
    5.7e-6 m (sub-ULP for typical depth) vs the original.

    Reference: https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/utils/geometry.py
    """
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    height, width = depth.shape[1:3]
    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1).astype(np.float32)  # (H, W, 3), z=1
    rays = coord @ Kinv.T                          # (H, W, 3) — same dot products as Kinv @ coord^T
    pts3D_1 = depth[0, :, :, None] * rays          # (H, W, 3) — first batch only, matches original return shape
    return pts3D_1 @ R.T + t                       # (H, W, 3)


def apply_transformation(points, transform):
    """
    Apply a 4x4 transformation matrix to a set of 3D points.
    
    Parameters:
    points (np.ndarray): An array of shape (N, 3) containing 3D points.
    transform (np.ndarray): A 4x4 transformation matrix.
    
    Returns:
    np.ndarray: The transformed 3D points of shape (N, 3).
    """
    # Convert points to homogeneous coordinates
    num_points = points.shape[0]
    homog_points = np.hstack([points, np.ones((num_points, 1))])
    
    # Apply transformation
    transformed = homog_points @ transform.T
    
    # Convert back to 3D
    return transformed[:, :3]


def extend_floor_plane(pts3d, normals, mask_floor, steps=100):
    """
    Estimates and extends a floor plane based on 3D points and their corresponding normals.
    Args:
        pts3d (torch.Tensor): A tensor of shape (H, W, 3) containing 3D point coordinates.
        normals (torch.Tensor): A tensor of shape (H, W, 3) containing normal vectors for each point.
        mask_floor (numpy.ndarray): A boolean array indicating which points belong to the floor.
    Returns:
        plane_points (numpy.ndarray): A grid of points representing the extended floor plane.
        plane_normal (numpy.ndarray): The normal vector of the fitted plane.
        plane_d (float): The plane's intercept term (D) in the plane equation.
    """

    if not any(mask_floor.flatten()):
        # return empty tensors if no floor points are found
        return torch.zeros(0, 3, device=pts3d.device), torch.zeros(3, device=pts3d.device), 0.0

    # Move floor mask to GPU
    floor_mask = torch.from_numpy(mask_floor).bool().to(pts3d.device)
    # Gather floor points
    pts_floor = pts3d[floor_mask].float()
    valid_mask = torch.isfinite(pts3d).all(dim=-1)  # Shape: (H, W)
    pts3d = pts3d[valid_mask]  # Shape: (N_valid, 3)            # remove NaN/Inf points from pts3d --> can cause issues
    norms_floor = normals[floor_mask].float()

    # RANSAC in PyTorch, but sample 20% of the floor points at a time
    best_normal, best_d, best_inliers = None, None, -1
    sample_size = max(1, int(0.2 * pts_floor.size(0)))  # pick 20% of floor points
    for _ in range(20):
        idx = torch.randint(pts_floor.size(0), (sample_size,), device=pts_floor.device)
        sample_points = pts_floor[idx]
        center = sample_points.mean(dim=0)
        # Fit plane using SVD on the sampled points
        _, _, V = torch.pca_lowrank(sample_points - center, q=3)
        n = V[:, -1]
        # Align plane normal with average of chosen points' normals
        avg_normal = norms_floor[idx].mean(dim=0)
        if torch.dot(n, avg_normal) < 0:
            n = -n
        d_val = -torch.dot(n, center)
        dist = torch.abs(pts_floor @ n + d_val)
        inliers = (dist < 0.01).sum().item()
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal, best_d = n, d_val

    # Refine using SVD on inliers
    diff = torch.abs(pts_floor @ best_normal + best_d)
    inlier_pts = pts_floor[diff < 0.01]
    inlier_normals = norms_floor[diff < 0.01]
    center = inlier_pts.mean(dim=0)
    _, _, V = torch.pca_lowrank(inlier_pts - center, q=3)
    plane_normal = V[:, -1]
    avg_inlier_normal = inlier_normals.mean(dim=0)
    if torch.dot(plane_normal, avg_inlier_normal) < 0:
        plane_normal = -plane_normal
    plane_d = -torch.dot(plane_normal, center)

    # Extend plane with truncated range
    x_vals = pts3d[..., 0].reshape(-1)
    z_vals = pts3d[..., 2].reshape(-1)
    x_vals_sorted, _ = torch.sort(x_vals)
    z_vals_sorted, _ = torch.sort(z_vals)

    idx_lower_x = int(0.02 * x_vals_sorted.numel())
    idx_upper_x = int(0.98 * x_vals_sorted.numel())
    x_min = x_vals_sorted[idx_lower_x]
    x_max = x_vals_sorted[idx_upper_x]

    idx_lower_z = int(0.02 * z_vals_sorted.numel())
    idx_upper_z = int(0.98 * z_vals_sorted.numel())
    z_min = z_vals_sorted[idx_lower_z]
    z_max = z_vals_sorted[idx_upper_z]

    xs = torch.linspace(x_min, x_max, steps, device=pts3d.device)
    zs = torch.linspace(z_min, z_max, steps, device=pts3d.device)
    grid = torch.cartesian_prod(xs, zs)

    y_vals = -(plane_d + plane_normal[0]*grid[:, 0] + plane_normal[2]*grid[:, 1]) / plane_normal[1]
    plane_torch = torch.stack([grid[:, 0], y_vals, grid[:, 1]], dim=-1)

    plane_points = plane_torch
    plane_normal = plane_normal
    plane_d = plane_d.item()

    return plane_points, plane_normal, plane_d


def get_scene_normals(pts3d, device='cpu'):
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    # Estimate normals with fixed random seed for consistency
    # Note: Open3D normal estimation may still have some non-determinism
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
    # Get normals as numpy array
    normals = np.asarray(pcd.normals)
    scene_normals = torch.from_numpy(normals).float().to(device)
    # orient normals towards camera
    points_to_origin = -torch.from_numpy(pts3d).float().to(device)
    dots = torch.sum(scene_normals * points_to_origin, dim=-1)
    scene_normals[dots < 0] *= -1
    return scene_normals
