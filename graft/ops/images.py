"""Image preparation for GRAFT.

`prepare_image` is vendored verbatim (by AST source range, see tools/extract.py)
from the research tree's `utils/utils.py` -- it defines the exact geometry the
published numbers were measured under: longest side resized to 518 with LANCZOS,
then padded with white to a multiple of 14 (DINOv2's patch size). The only change
is that the destination device is a parameter instead of a hard-coded `.cuda()`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image, ImageOps

ImageLike = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]


def prepare_image(image: Image.Image, pad_to_square=False, device='cuda'):
    largest_dim = max(image.size)
    scale = 518 / largest_dim
    new_w = int(image.size[0] * scale)
    new_h = int(image.size[1] * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    image = ImageOps.pad(image, ((new_w + 13)//14*14, (new_h + 13)//14*14), color=(255, 255, 255))
    if pad_to_square:
        image = ImageOps.pad(image, (518, 518), color=(255, 255, 255))
    image = torch.from_numpy(np.array(image)).float() / 255.0
    image = image.permute(2, 0, 1)[None, :, :, :].to(device)
    return image


def prepare_mask(mask: Image.Image, pad_to_square=False, device='cuda'):
    """
    Prepare a 2D mask for processing.
    
    Args:
        mask: PIL Image with 2D mask (H, W)
        pad_to_square: whether to pad to 518x518
    
    Returns:
        torch.Tensor on cuda with shape (1, H, W)
    """
    largest_dim = max(mask.size)
    scale = 518 / largest_dim
    new_w = int(mask.size[0] * scale)
    new_h = int(mask.size[1] * scale)
    mask = mask.resize((new_w, new_h), Image.NEAREST)
    mask = ImageOps.pad(mask, ((new_w + 13)//14*14, (new_h + 13)//14*14), color=0)
    if pad_to_square:
        mask = ImageOps.pad(mask, (518, 518), color=0)
    mask = torch.from_numpy(np.array(mask)).float()
    mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
    mask = mask.to(device)
    return mask


def to_pil(image: ImageLike) -> Image.Image:
    """Accept the shapes a caller is likely to have and return RGB PIL.

    Understands a path, a PIL image, an HWC array (uint8, or float in [0, 1]),
    and a CHW / 1CHW float tensor in [0, 1].
    """
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        arr = image.detach().cpu()
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise ValueError(f"expected a single image, got a batch of {arr.shape[0]}")
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3):   # CHW
            arr = arr.permute(1, 2, 0)
        image = arr.numpy()
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        hi = float(np.nanmax(image)) if image.size else 1.0
        image = (image * 255.0 if hi <= 1.0 + 1e-6 else image).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)   # HWC uint8 -> RGB
