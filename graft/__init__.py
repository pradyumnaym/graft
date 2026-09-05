"""GRAFT: Geometric Refinement and Fitting Transformer for Human Scene Reconstruction.

A learned human-scene-interaction prior that predicts iterative,
geometry-grounded refinements to reconstruct humans in scenes from a single image.

    from graft import GRAFTModel

    model = GRAFTModel.from_pretrained("graft").cuda().eval()
    out = model.predict("photo.jpg", scene_image="empty_room.jpg")
    out.vertices        # (n_people, 10475, 3) in the scene's camera frame, metres
"""
from .model import (PRETRAINED, GRAFTConfig, GRAFTModel, GRAFTOutput,
                    NoPeopleDetected)
from .paths import data_root

__all__ = ["GRAFTModel", "GRAFTConfig", "GRAFTOutput", "NoPeopleDetected",
           "PRETRAINED", "data_root"]
__version__ = "1.0.0"
