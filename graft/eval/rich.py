"""RICH-sample100 dataset loaders and per-frame iteration helpers.

Mirrors `evaluation.prox` but for the RICH dataset:
- Per-frame `CalibratedCamera` (XML) gives world->camera transform.
- Per-(scene, camera) pre-rendered "scene_image" (input to model.infer).
- Per-(scene, camera) floor mask (computed once, cached by scene+cam id).
- Per-activity GT scene mesh (loaded once per activity).
- GT comes as a mesh PLY per frame; SMPL-X is fit to it via BodyFitter.
"""
from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image

from ..ops.images import prepare_image, prepare_mask
from ._assets import asset


ACTIVITY_TO_SCENE = {
    "cooking1": "scan_table_camcoord.ply",
    "dips1": "scan_camcoord.ply",
    "dips2": "scan_camcoord.ply",
    "lunge1": "scan_camcoord.ply",
    "lunge2": "scan_camcoord.ply",
    "mountainclimber1": "scan_camcoord.ply",
    "mountainclimber2": "scan_camcoord.ply",
    "pushup1": "scan_camcoord.ply",
    "pushup2": "scan_camcoord.ply",
    "plankjack1": "scan_yoga_scene_camcoord.ply",
    "sidebalancerun1": "scan_yoga_scene_camcoord.ply",
}


@dataclass
class RichPaths:
    benchmark_root: str

    @property
    def root(self) -> str:
        return f"{self.benchmark_root}/PROX/rich-sample100"

    @property
    def images_glob(self) -> str:
        return f"{self.root}/sample/*.jpeg"

    def calibration_xml(self, scene_name: str, camera_id: str) -> str:
        return f"{self.root}/calibration/{scene_name}/calibration/0{camera_id}.xml"

    def gt_mesh_ply(self, frame_stem_no_camid: str) -> str:
        return f"{self.root}/sample/{frame_stem_no_camid}_gt.ply"

    def scene_image_png(self, scene_full: str, camera_id: str) -> str:
        return asset("imgs", "rich", "scene_imgs", f"{scene_full}_{camera_id}.png")

    def gt_scene_mesh(self, scene_name: str, activity: str) -> str:
        ply = ACTIVITY_TO_SCENE[activity].replace(".ply", "_simp.ply")
        return f"{self.root}/scene/scan_calibration/{scene_name}/{ply}"


@dataclass
class RichSceneCache:
    """Caches scene_image / floor_mask / scene_mesh keyed by their natural index."""
    sam_model: object
    scene_image_cache: Dict[Tuple[str, str], torch.Tensor] = field(default_factory=dict)
    floor_mask_cache: Dict[Tuple[str, str], torch.Tensor] = field(default_factory=dict)
    floor_mask_np_cache: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    scene_mesh_cache: Dict[Tuple[str, str], trimesh.Trimesh] = field(default_factory=dict)

    def get_scene_inputs(
        self, paths: RichPaths, scene_full: str, camera_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        key = (scene_full, camera_id)
        if key in self.scene_image_cache:
            return (
                self.scene_image_cache[key],
                self.floor_mask_cache[key],
                self.floor_mask_np_cache[key],
            )

        scene_image_pil = Image.open(paths.scene_image_png(scene_full, camera_id))
        scene_image = prepare_image(scene_image_pil)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            results_floor = self.sam_model.predict(
                [scene_image_pil], ["ground, floor, carpet."]
            )
        mask_floor_arr = np.stack(results_floor[0]["masks"], axis=0).any(axis=0)
        floor_mask = prepare_mask(Image.fromarray(mask_floor_arr))
        floor_mask_np = floor_mask.reshape(-1).detach().cpu().numpy()

        self.scene_image_cache[key] = scene_image
        self.floor_mask_cache[key] = floor_mask
        self.floor_mask_np_cache[key] = floor_mask_np
        return scene_image, floor_mask, floor_mask_np

    def get_scene_mesh(
        self, paths: RichPaths, scene_name: str, activity: str, transform: np.ndarray
    ) -> trimesh.Trimesh:
        key = (scene_name, activity)
        if key in self.scene_mesh_cache:
            mesh = self.scene_mesh_cache[key].copy()
            mesh.apply_transform(transform)
            return mesh
        mesh = trimesh.load(paths.gt_scene_mesh(scene_name, activity))
        self.scene_mesh_cache[key] = mesh.copy()  # store untransformed
        mesh.apply_transform(transform)
        return mesh


@dataclass
class RichFrame:
    frame_path: str
    scene_name: str          # e.g. "Gym" — used for calibration / GT-scene-mesh paths
    scene_full: str          # e.g. "Gym_010_cooking1" — used for scene_image path
    camera_id: str           # e.g. "01"
    activity: str            # e.g. "cooking1"
    transform: np.ndarray    # 4x4 world->camera (from CalibratedCamera)
    image: torch.Tensor      # prepare_image
    person_mask: torch.Tensor  # prepare_mask (1,1,H,W)
    scene_image: torch.Tensor  # cached per (scene_full, cam)
    floor_mask: torch.Tensor   # cached per (scene_full, cam)
    floor_mask_np: np.ndarray  # cached flat numpy view of floor_mask
    gt_mesh: trimesh.Trimesh   # per-frame GT human mesh
    gt_scene_mesh: trimesh.Trimesh  # per-activity scene mesh, transformed to camera


def _parse_frame_name(name: str) -> Tuple[str, str, str, str, str]:
    """Returns (scene_name, scene_full, camera_id, activity, stem_no_camera).

    Filename format: `{scene}_{subj}_{activity}+{frame_idx}_{camera_id}.jpeg`
    e.g. `Gym_010_cooking1+00324_01.jpeg`

    The notebook uses two different parses:
      - `name.split('+')[0]` -> scene_full ("Gym_010_cooking1")  (for scene_image lookup)
      - `stem.split('_')[0]` -> scene_name ("Gym")               (for calibration XML)
    """
    stem = name[: -len(".jpeg")] if name.endswith(".jpeg") else name
    scene_name = stem.split("_")[0]
    camera_id = stem.split("_")[-1]
    activity = stem.split("_")[2].split("+")[0]
    scene_full = name.split("+")[0]
    stem_no_cam = stem[: -len(f"_{camera_id}")]
    return scene_name, scene_full, camera_id, activity, stem_no_cam


def _build_world_to_camera(cam) -> np.ndarray:
    """CalibratedCamera -> 4x4 world->camera transform."""
    T = np.eye(4)
    T[:3, :3] = cam.rotation.squeeze().cpu().numpy()
    T[:3, 3] = cam.translation.squeeze().cpu().numpy()
    return T


def iter_rich_frames(
    paths: RichPaths,
    sam_model,
    scene_cache: RichSceneCache,
    limit: Optional[int] = None,
) -> Iterator[RichFrame]:
    """Yield RichFrame for each .jpeg in the rich-sample100 sample dir."""
    from .alignment import CalibratedCamera

    images = glob.glob(paths.images_glob)
    for i, frame_path in enumerate(images):
        if limit is not None and i >= limit:
            break
        path = Path(frame_path)
        scene_name, scene_full, camera_id, activity, stem_no_cam = _parse_frame_name(
            path.name
        )

        cam = CalibratedCamera(paths.calibration_xml(scene_name, camera_id))
        transform = _build_world_to_camera(cam)

        gt_mesh = trimesh.load(paths.gt_mesh_ply(stem_no_cam))

        image_pil = Image.open(frame_path)
        image = prepare_image(image_pil)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            results = sam_model.predict([image_pil], ["person."])
        mask_arr = np.stack(results[0]["masks"], axis=0).any(axis=0)
        person_mask = prepare_mask(Image.fromarray((mask_arr * 255).astype(np.uint8)))

        scene_image, floor_mask, floor_mask_np = scene_cache.get_scene_inputs(
            paths, scene_full, camera_id
        )
        gt_scene_mesh = scene_cache.get_scene_mesh(paths, scene_name, activity, transform)

        yield RichFrame(
            frame_path=frame_path,
            scene_name=scene_name,
            scene_full=scene_full,
            camera_id=camera_id,
            activity=activity,
            transform=transform,
            image=image,
            person_mask=person_mask,
            scene_image=scene_image,
            floor_mask=floor_mask,
            floor_mask_np=floor_mask_np,
            gt_mesh=gt_mesh,
            gt_scene_mesh=gt_scene_mesh,
        )


def load_or_init_gt_contact_cache(
    interpenetration_threshold: float, contact_threshold: float
) -> tuple[dict, str]:
    cache_path = (
        asset(f"example_to_gt_contacts_rich_"
              f"{interpenetration_threshold}_{contact_threshold}.pkl")
    )
    cache: dict = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    return cache, cache_path
