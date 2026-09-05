"""Body-scene contact classification.

Vendored verbatim (by AST source range) from the research tree's `human.py`.
`get_contact_mask` raycasts against a scene *mesh* (PROX/RICH ground truth);
`get_contact_mask_from_pointcloud_pt` does the same against a predicted point
cloud. Both define the contact metric, so neither is modified.
"""
import numpy as np
import open3d as o3d
import torch

from .knn import knn1


def get_contact_mask(scene_mesh, human_vertices, 
    threshold=0.02, interpenetration_threshold=0.15, return_dists=False,
    return_vecs=False
    ):
    """
    Given a scene mesh and human vertices, compute a boolean mask indicating
    whether each vertex is within the threshold distance from the mesh surface.
    This approach accounts for large faces by measuring distance to the surface,
    not just the nearest vertices.
    """
    scene = o3d.t.geometry.RaycastingScene()
    scene_o3d = o3d.geometry.TriangleMesh()
    scene_o3d.vertices = o3d.utility.Vector3dVector(scene_mesh.vertices)
    scene_o3d.triangles = o3d.utility.Vector3iVector(scene_mesh.faces)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene_o3d))
    ret_dict = scene.compute_closest_points(o3d.core.Tensor.from_numpy(human_vertices.astype(np.float32)))
    points = ret_dict['points'].numpy()
    distances = np.linalg.norm(points - human_vertices, axis=1)

    normals = ret_dict['primitive_normals'].numpy()  # shape (N, 3)
    directions = human_vertices - points  # vector from the closest point to the human vertex
    directions_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    # avoid division by zero
    directions_normalized = directions / (directions_norm + 1e-6)
    dot_products = np.sum(directions_normalized * normals, axis=1)
    contact_from_dot = dot_products < 0
    # mark as contact if within threshold or if the normal check indicates contact
    mask = (distances <= threshold) | (contact_from_dot & (distances < interpenetration_threshold))

    if return_dists:
        assert not return_vecs, "Cannot return both distances and vectors."
        return mask, distances
    
    if return_vecs:
        assert not return_dists, "Cannot return both distances and vectors."
        return mask, directions
    return mask


def get_contact_mask_from_pointcloud_pt(
    scene_points, scene_normals, human_vertices, threshold=0.02, interpenetration_threshold=0.15, 
    fg_mask=None, return_dists=False, return_vecs=False):
    
    # Convert numpy arrays to torch tensors on GPU without a batch dimension.
    device = torch.device("cuda")
    if isinstance(scene_points, np.ndarray):
        scene_points_t = torch.tensor(scene_points, dtype=torch.float32, device=device)  # (N, 3)
        scene_normals_t = torch.tensor(scene_normals, dtype=torch.float32, device=device)  # (N, 3)
        human_vertices_t = torch.tensor(human_vertices, dtype=torch.float32, device=device)  # (M, 3)
    else:
        scene_points_t = scene_points.clone().detach()
        scene_normals_t = scene_normals.clone().detach()
        human_vertices_t = human_vertices.clone().detach()

    fg_mask_binary = torch.tensor(fg_mask > 0, dtype=torch.bool, device=device) if fg_mask is not None else None

    # K=1 nearest scene point per body vertex. Bit-identical to the pytorch3d
    # knn_points(K=1) this replaces; see graft/eval/knn.py for why it was replaced.
    nn_sq_dists, indices = knn1(human_vertices_t, scene_points_t)
    dists = nn_sq_dists.sqrt()                      # (M,)
    # Gather nearest scene points and corresponding normals.
    nearest_pts = scene_points_t[indices]  # (M, 3)
    nearest_normals = scene_normals_t[indices]  # (M, 3)

    # Compute directions from the nearest scene points to human vertices.
    directions = human_vertices_t - nearest_pts  # (M, 3)
    norms = torch.norm(directions, dim=1, keepdim=True) + 1e-6
    direction_normalized = directions / norms
    dot_products = (direction_normalized * nearest_normals).sum(dim=1)
    contact_from_dot = (dot_products < 0) & (dists < interpenetration_threshold)

    mask = (dists <= threshold) | contact_from_dot
    mask = mask.cpu().numpy()


    if fg_mask_binary is not None:
        assert scene_points_t.shape[0] == fg_mask_binary.shape[0], \
            "Scene points and foreground mask must have the same number of points."
        
        nn_in_fg_mask = fg_mask_binary[indices]        # is the nearest point of this vertex in the foreground mask?
        contact_nn = nn_in_fg_mask[mask]               # get a mask of FG mask for each contact vertex
        return mask, torch.any(contact_nn).cpu().numpy()     # return true if any of the contact points are in the foreground mask

    if return_dists:
        assert not return_vecs, "Cannot return both distances and vectors."
        dists = dists.cpu().numpy()
        return mask, dists
    
    if return_vecs:
        assert not return_dists, "Cannot return both distances and vectors."
        directions = directions.cpu().numpy()
        return mask, directions
    return mask
