"""PROX-quantitative dataset loaders and per-frame iteration helpers.

This module owns everything PROX-specific: filesystem layout, GT MoSh
fits, world transforms, depth/intrinsics, scene mesh, and per-frame
inputs. Metric computation lives in `evaluation.metrics`; the orchestrator
lives in `evaluate.py`.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from .scene_geometry import depth_to_points
from ..ops.images import prepare_image
from ._assets import asset


@dataclass
class ProxPaths:
    benchmark_root: str  # root of the data root containing PROX/

    @property
    def quant_root(self) -> str:
        return f"{self.benchmark_root}/PROX/quantitative"

    @property
    def recordings_glob(self) -> str:
        return f"{self.quant_root}/recordings/*/Color/*.jpg"

    @property
    def calib_color(self) -> str:
        return f"{self.benchmark_root}/PROX/proxd_full/calibration/Color.json"

    @property
    def cam2world_json(self) -> str:
        return f"{self.quant_root}/cam2world/vicon.json"

    @property
    def vicon2scene_json(self) -> str:
        return f"{self.quant_root}/vicon2scene.json"

    @property
    def scene_ply(self) -> str:
        return f"{self.quant_root}/scenes/vicon.ply"

    def mosh_pkl(self, recording: str, frame_stem: str) -> str:
        return f"{self.quant_root}/fittings/mosh/{recording}/results/{frame_stem}/000.pkl"


@dataclass
class ProxScene:
    """Static (frame-independent) PROX evaluation context."""
    c2w: np.ndarray  # (4, 4)
    w2s: np.ndarray  # (4, 4)
    K: np.ndarray  # (3, 3) original intrinsics
    K_resized: np.ndarray  # (3, 3) intrinsics for the 518-resized GT depth
    depth_gt: np.ndarray  # (294, 518) padded GT depth
    depth_mask: np.ndarray  # (294, 518) bool, depth>0
    pts3d_gt: np.ndarray  # (294, 518, 3)
    scene_mesh: trimesh.Trimesh
    scene_image: torch.Tensor  # prepared scene image (1, 3, H, W) on GPU
    floor_mask: np.ndarray  # (H, W) bool, floor mask in scene_image coords


def load_prox_scene(
    paths: ProxPaths,
    scene_image_path: str,
    sam_model,
    depth_pkl: str = asset("prox_quantitative_depth.pkl"),
    target_short_side: int = 294,
    target_long_side: int = 518,
) -> ProxScene:
    """Build the static eval context (transforms, GT depth, scene image + floor mask)."""
    c2w = np.array(json.load(open(paths.cam2world_json)))
    w2s = np.array(json.load(open(paths.vicon2scene_json)))

    with open(paths.calib_color) as f:
        K = np.array(json.load(f)["camera_mtx"])

    with open(depth_pkl, "rb") as f:
        depth_full = pickle.load(f)

    larger_dim = max(depth_full.shape)
    scale = float(target_long_side) / float(larger_dim)
    new_w, new_h = int(depth_full.shape[1] * scale), int(depth_full.shape[0] * scale)

    depth = cv2.resize(depth_full, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    depth_padded = np.zeros((target_short_side, target_long_side), dtype=depth.dtype)
    pad_top = (target_short_side - new_h) // 2
    pad_left = (target_long_side - new_w) // 2
    depth_padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = depth

    K_resized = K.copy()
    K_resized[[0, 1, 0, 1], [0, 1, 2, 2]] *= scale
    K_resized[0, 2] += pad_left
    K_resized[1, 2] += pad_top

    pts3d_gt = depth_to_points(depth_padded[None], K_resized)
    depth_mask = pts3d_gt[:, :, 2] > 0

    scene_mesh = trimesh.load(paths.scene_ply)
    scene_mesh.visual.material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )

    scene_image_pil = Image.open(scene_image_path).convert("RGB")
    scene_image = prepare_image(scene_image_pil)

    scene_image_for_sam = Image.fromarray(
        (scene_image[0].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        results_floor = sam_model.predict(
            [scene_image_for_sam], ["ground, floor, carpet."]
        )
    floor_mask = np.stack(results_floor[0]["masks"], axis=0).any(axis=0)

    return ProxScene(
        c2w=c2w,
        w2s=w2s,
        K=K,
        K_resized=K_resized,
        depth_gt=depth_padded,
        depth_mask=depth_mask,
        pts3d_gt=pts3d_gt,
        scene_mesh=scene_mesh,
        scene_image=scene_image,
        floor_mask=floor_mask,
    )


@dataclass
class ProxFrame:
    recording: str
    frame: str  # filename including .jpg
    image_pil: Image.Image  # already FLIP_LEFT_RIGHT'd
    image: torch.Tensor  # prepare_image output, (1, 3, H, W)
    mosh: dict  # raw MoSh fit dict


def iter_prox_frames(
    paths: ProxPaths,
    limit: Optional[int] = None,
) -> Iterator[ProxFrame]:
    """Yield per-frame inputs for PROX-quant evaluation.

    Order matches glob.glob() (filesystem-determined) — same as the source notebook,
    so per-frame evaluation results stack in identical order.
    """
    for i, frame_path in enumerate(glob.glob(paths.recordings_glob)):
        if limit is not None and i >= limit:
            break
        path = Path(frame_path)
        recording = path.parts[-3]
        frame = path.parts[-1]

        with open(paths.mosh_pkl(recording, frame[:-4]), "rb") as f:
            mosh = pickle.load(f, encoding="latin1")

        image_pil = Image.open(path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
        image = prepare_image(image_pil)

        yield ProxFrame(
            recording=recording,
            frame=frame,
            image_pil=image_pil,
            image=image,
            mosh=mosh,
        )


def load_contact_vertex_weights(
    path: str = asset("contact_vertice_weights.pkl"),
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (contact_vertices_mask, contact_vertices_weights) from the cached pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["contact_vertices_mask"], data["contact_vertices_weights"]


def load_or_init_gt_contact_cache(interpenetration_threshold: float) -> tuple[dict, str]:
    """Load the cached GT contact mask/vector dict; returns (cache_dict, cache_path)."""
    cache_path = (
        asset(f"example_to_gt_contacts_prox_{interpenetration_threshold}.pkl")
    )
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    return cache, cache_path
