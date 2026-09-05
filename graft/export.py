"""Export a GRAFT result as a single glTF scene.

The bodies and the scene point map share one coordinate frame, so one ``.glb``
opens in any viewer -- or a browser -- showing the people sitting in the
reconstructed scene. This is how the project page's interactive results are made.

    from graft.export import write_glb
    write_glb("out.glb", out.vertices, model.faces, out.points, "empty_room.jpg")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from .ops.images import ImageLike, to_pil


def pointmap_to_mesh_data(pts3d_hw, colors_hw, max_edge_length: Optional[float] = None):
    """Turn an HxW point map and its colors into triangle-mesh arrays.

    Triangulates the pixel grid, then drops any face whose longest edge exceeds
    ``max_edge_length`` -- eight times the median edge by default. Without that
    cut, the grid stretches sheets of geometry across depth discontinuities and
    the scene reads as a smear rather than separate surfaces.
    """
    h, w = pts3d_hw.shape[:2]
    verts = pts3d_hw.reshape(-1, 3)
    colors = colors_hw.reshape(-1, 3)

    if max_edge_length is None:
        dx = pts3d_hw[:, 1:] - pts3d_hw[:, :-1]
        dy = pts3d_hw[1:] - pts3d_hw[:-1]
        d = np.concatenate([np.linalg.norm(dx.reshape(-1, 3), axis=1),
                            np.linalg.norm(dy.reshape(-1, 3), axis=1)])
        d = d[np.isfinite(d)]
        max_edge_length = np.median(d) * 8.0 if d.size > 0 else np.inf

    yy, xx = np.meshgrid(np.arange(h - 1), np.arange(w - 1), indexing="ij")
    v0 = yy * w + xx
    v1, v2 = v0 + 1, v0 + w
    v3 = v2 + 1
    faces = np.stack([v0, v2, v1, v1, v2, v3], axis=-1).reshape(-1, 3).astype(np.int64)

    valid_faces = np.isfinite(verts).all(axis=1)[faces].all(axis=1)
    p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    edge_max = np.maximum.reduce([np.linalg.norm(p0 - p1, axis=1),
                                  np.linalg.norm(p1 - p2, axis=1),
                                  np.linalg.norm(p2 - p0, axis=1)])
    valid_faces &= edge_max < max_edge_length
    return verts, faces[valid_faces], colors


def scene_mesh(points, image: ImageLike, max_edge_length: Optional[float] = None
               ) -> trimesh.Trimesh:
    """Vertex-colored mesh of the scene point map, colored by ``image``."""
    pts = np.nan_to_num(np.asarray(points, dtype=np.float32),
                        nan=0.0, posinf=0.0, neginf=0.0)
    h, w = pts.shape[:2]
    cols = np.asarray(to_pil(image).convert("RGB").resize((w, h)), np.float32) / 255.0
    v, f, c = pointmap_to_mesh_data(pts, cols, max_edge_length)
    return trimesh.Trimesh(v, f, process=False,
                           vertex_colors=(c * 255).astype(np.uint8))


def write_glb(path, vertices, faces, points, image: ImageLike,
              max_edge_length: Optional[float] = None) -> Path:
    """Write the posed bodies and the scene to one ``.glb``.

    Args:
        path: destination ``.glb``.
        vertices: ``(n_people, V, 3)`` posed SMPL-X vertices.
        faces: the SMPL-X face array (``GRAFTModel.faces``).
        points: ``(H, W, 3)`` scene point map.
        image: the view the scene colors come from -- the scene image if you have
            one, else the input photo.
    """
    bodies = [trimesh.Trimesh(vertices=np.asarray(v), faces=faces, process=False)
              for v in np.asarray(vertices)]
    trimesh.Scene(bodies + [scene_mesh(points, image, max_edge_length)]).export(path)
    return Path(path)
