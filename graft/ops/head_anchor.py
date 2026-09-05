"""analytic head-surface depth anchor.

Drop-in replacement for the full-mesh depth *rasterization* that is used only to
read the head-pixel surface depth (for the NLF<->depth scale/transl alignment).
Instead of rendering the whole SMPL-X mesh and sampling one pixel, we intersect a
single ray (camera origin -> SMPL-X head joint) with the head+neck faces and take
the frontmost hit. Pure torch, no pytorch3d, no tunable threshold, fully batched
over humans.

Validated vs the pytorch3d full render on real humans:
  |Δz| median 2.7 mm, p95 1.1 cm, 0% miss; ~0.3 ms vs ~68 ms (~200x).

Usage in the model's `_process_input` (replaces the render + depth_to_points
+ head-pixel sample block) -- a single line, no per-class init:

    nlf_points = head_anchor_points(self, nlf_out.vertices, nlf_out.joints[:, 15])

`nlf_points` is the (N, 3) camera-space head-surface anchor; `nlf_points[:, 2]` is
the surface depth. The downstream scale/transl alignment is unchanged.
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F

# Head+neck SMPL-X vertex ids (union of regions['head'|'neck'] from
# vlms/smplx_body_parts.json), dumped into the repo so this is self-contained.
_HEAD_NECK_IDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "smplx_head_neck_ids.npy",
)


def load_head_neck_ids() -> np.ndarray:
    """The 4976 head+neck SMPL-X vertex ids used to subset the mesh faces."""
    return np.load(_HEAD_NECK_IDS_PATH)


def build_head_faces(faces, head_vertex_ids) -> torch.Tensor:
    """faces: (F,3) long triangle indices. head_vertex_ids: iterable of vertex ids.
    Returns (Fh,3): the faces whose three vertices are all in the head set."""
    faces = torch.as_tensor(faces, dtype=torch.long)
    nv = int(faces.max()) + 1
    keep = torch.zeros(nv, dtype=torch.bool)
    keep[torch.as_tensor(sorted({int(i) for i in head_vertex_ids}))] = True
    return faces[keep[faces].all(dim=-1)].contiguous()


def ray_triangle_depth(vertices, ray_dir, faces, eps: float = 1e-9):
    """Frontmost ray-triangle intersection (Moller-Trumbore), batched over humans.

    vertices : (N, V, 3)  camera-space mesh vertices (camera centre at the origin)
    ray_dir  : (N, 3)     per-human ray direction (need not be normalised)
    faces    : (Fh, 3)    triangle vertex indices (head+neck subset)
    Returns (points (N,3) camera-space anchor, hit (N,) bool). On miss falls back
    to the head vertex nearest the ray."""
    u = F.normalize(ray_dir, dim=-1)                         # (N,3) unit ray dir
    tri = vertices[:, faces]                                 # (N,Fh,3,3)
    v0, v1, v2 = tri.unbind(dim=2)                           # each (N,Fh,3)
    D = u[:, None].expand(-1, faces.shape[0], -1)            # (N,Fh,3)
    e1, e2 = v1 - v0, v2 - v0
    pv = torch.cross(D, e2, dim=-1)
    det = (e1 * pv).sum(-1)                                  # (N,Fh)
    ok = det.abs() > eps
    inv = torch.where(ok, 1.0 / det.masked_fill(~ok, 1.0), torch.zeros_like(det))
    tvec = -v0                                               # camera origin - v0
    bu = (tvec * pv).sum(-1) * inv
    qv = torch.cross(tvec, e1, dim=-1)
    bv = (D * qv).sum(-1) * inv
    t = (e2 * qv).sum(-1) * inv                              # distance along ray
    hit = ok & (bu >= 0) & (bv >= 0) & (bu + bv <= 1) & (t > eps)
    t_front = t.masked_fill(~hit, float("inf")).min(dim=1).values   # (N,) nearest hit
    miss = torch.isinf(t_front)
    if miss.any():                                          # fallback: vertex nearest the ray
        hv = vertices[:, faces.reshape(-1).unique()]        # (N, Vh, 3) head verts
        tv = (hv * u[:, None]).sum(-1)                       # along-ray dist
        perp2 = (hv ** 2).sum(-1) - tv ** 2                  # squared perp dist to ray
        t_near = tv.gather(1, perp2.argmin(1, keepdim=True)).squeeze(1)
        t_front = torch.where(miss, t_near, t_front)
    return t_front[:, None] * u, ~miss                      # point = t * u (origin at 0)


class HeadRayAnchor:
    """Holds the static head-face subset; call per-forward for the anchor point."""

    def __init__(self, faces, head_vertex_ids):
        self.head_faces = build_head_faces(faces, head_vertex_ids)

    def to(self, device):
        self.head_faces = self.head_faces.to(device)
        return self

    @torch.no_grad()
    def __call__(self, vertices, head_joints) -> torch.Tensor:
        """vertices: (N,V,3) camera-space NLF mesh. head_joints: (N,3) SMPL-X head joint.
        Returns (N,3) camera-space head-surface anchor point (`[:,2]` is the depth)."""
        hf = self.head_faces.to(vertices.device)
        points, _ = ray_triangle_depth(vertices, head_joints, hf)
        return points


def head_anchor_points(model, vertices, head_joints) -> torch.Tensor:
    """Lazily build (once) and apply the model's head anchor. Keeps the model classes
    free of any extra init -- they call this one line where they used to render."""
    anchor = getattr(model, "_head_anchor", None)
    if anchor is None:
        anchor = HeadRayAnchor(model._ext["smplx_aa"].faces, load_head_neck_ids())
        model._head_anchor = anchor
    return anchor(vertices, head_joints)
