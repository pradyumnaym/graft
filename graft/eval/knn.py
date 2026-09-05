"""K=1 nearest neighbour, without pytorch3d.

The contact metric's only use of pytorch3d was a single `knn_points(..., K=1)`.
pytorch3d has to be compiled, so it is replaced here by an explicit
squared-distance argmin -- verified against it on real data as bit-identical in
both indices and distances.
"""
from __future__ import annotations

import torch

#: Cap the (chunk x points x 3) temporary. 256 MB keeps this comfortably inside
#: whatever the backbones have left, while staying large enough to be fast.
_MAX_TEMP_BYTES = 256 << 20


def knn1(query: torch.Tensor, points: torch.Tensor):
    """Nearest neighbour of each query among `points`.

    Args:
        query: (M, 3)
        points: (N, 3)

    Returns:
        (sq_dists, indices), each (M,) -- squared distances, matching what
        pytorch3d returns before its caller takes the square root.
    """
    if query.ndim != 2 or points.ndim != 2:
        raise ValueError(f"expected (M,3) and (N,3), got {tuple(query.shape)} "
                         f"and {tuple(points.shape)}")
    n, dim = points.shape
    per_row = max(1, n * dim * points.element_size())
    chunk = max(1, min(query.shape[0], _MAX_TEMP_BYTES // per_row))

    sq = torch.empty(query.shape[0], dtype=points.dtype, device=points.device)
    idx = torch.empty(query.shape[0], dtype=torch.long, device=points.device)
    for s in range(0, query.shape[0], chunk):
        e = min(s + chunk, query.shape[0])
        diff = query[s:e, None, :] - points[None, :, :]
        d2 = (diff * diff).sum(-1)
        m, a = d2.min(dim=1)
        sq[s:e] = m
        idx[s:e] = a
    return sq, idx
