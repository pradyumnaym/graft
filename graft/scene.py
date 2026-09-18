"""Produce the empty-scene image GRAFT wants, when you don't already have one.

GRAFT is given two views of the same camera: the photo containing the people, and
a view of the same scene *without* them, which is where the geometry it places
bodies against comes from. Both published benchmarks supply a real photograph for
the second one. In the wild you usually do not have it, so this module makes one:
segment the people, then inpaint them out.

    from graft.scene import person_mask, remove_people

    mask = person_mask("photo.jpg")
    scene = remove_people("photo.jpg", mask)
    out = model.predict("photo.jpg", scene_image=scene, person_mask=mask)

The dependencies here are heavier than GRAFT's own (a segmentation model plus a
diffusion inpainter) and are therefore optional: `pip install -e ".[demo]"`.
Nothing in `graft.model` imports this module.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from ._optional import optional
from .ops.images import ImageLike, to_pil

_THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"

_sam = None
_pipe = None
_pipe_key = None
_auto_strategy: dict = {}

#: Prompt used to segment the people out of the photo.
PERSON_PROMPT = "person."
#: OmniEraser is conditioned to remove, not to invent; this is its removal prompt.
INPAINT_PROMPT = "There is nothing here."


#: Peak GPU memory, in GiB, each strategy needs for one 1024x1024 bf16 call.
#: FLUX.1-dev is 31.4 GiB of bf16 weights by itself (transformer 22.2, T5-XXL 8.9,
#: CLIP + VAE 0.4). "park" holds all of it at once, "model" only the largest
#: component at a time, "sequential" streams submodules and so fits anywhere.
#: The figures are the weights plus headroom for activations. A card near a
#: boundary is worth measuring rather than trusting these.
_PEAK_GIB = {"park": 35.0, "model": 26.0}

#: Most- to least-preferred. The first whose _PEAK_GIB fits wins; the last has no
#: minimum and is the fallback.
_STRATEGIES = ("park", "model", "sequential")
assert set(_PEAK_GIB) == set(_STRATEGIES[:-1]), "every strategy but the last needs a minimum"


def _pick_strategy(device: str) -> str:
    """Choose an offload strategy from the memory actually free on `device`.

    Reads *free* rather than total memory: GRAFT is normally already resident on
    the same GPU when this is called. The answer is cached per device, so a long
    run cannot flip strategy between images as the allocator's cache grows and
    force an expensive pipeline rebuild.
    """
    key = str(device)
    if key in _auto_strategy:
        return _auto_strategy[key]
    # torch's caching allocator holds on to freed blocks, and mem_get_info counts
    # those as used. Release them first, or this systematically under-reports.
    torch.cuda.empty_cache()
    try:
        free_gib = torch.cuda.mem_get_info(torch.device(device))[0] / 2**30
    except Exception as e:
        raise RuntimeError(
            f"graft.scene cannot size the inpainter for device={device!r}: {e}\n"
            f"    Inpainting needs a CUDA device. Pass an explicit offload= "
            f"strategy to skip this check."
        ) from e
    chosen = _STRATEGIES[-1]
    for name in _STRATEGIES[:-1]:
        if free_gib >= _PEAK_GIB[name]:
            chosen = name
            break
    _auto_strategy[key] = chosen
    return chosen


def _resolve_offload(offload: Union[str, bool], device: str) -> str:
    """Normalise the `offload` argument of :func:`remove_people` to a strategy."""
    if isinstance(offload, bool) or type(offload).__name__ == "bool_":
        return "park" if offload else "resident"
    if offload == "auto":
        return _pick_strategy(device)
    if offload not in (*_STRATEGIES, "resident"):
        raise ValueError(
            f"offload={offload!r}; expected 'auto', 'resident', or one of "
            + ", ".join(repr(s) for s in _STRATEGIES))
    return offload


def _load_sam(sam_type: str = "sam2.1_hiera_large"):
    global _sam
    if _sam is None:
        with optional("demo", "Segmenting people out of the scene"):
            from lang_sam import LangSAM
        _sam = LangSAM(sam_type=sam_type)
    return _sam


def _load_inpainter(device: str = "cuda", strategy: str = "park"):
    """Build the FLUX-based OmniEraser pipeline (once per strategy)."""
    global _pipe, _pipe_key
    key = (str(device), strategy)
    if _pipe is not None and _pipe_key == key:
        return _pipe
    if _pipe is not None:
        # A rebuild rather than remove_all_hooks() + re-enable: cheap insurance,
        # since the pipeline is only ever built once per process in practice.
        # accelerate's hooks make reference cycles (the wrapped forward closes
        # over the module), so dropping the name is not enough to free it -- the
        # cyclic collector has to run before empty_cache() can do anything.
        _pipe, _pipe_key = None, None
        gc.collect()
        torch.cuda.empty_cache()
    if str(_THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(_THIRD_PARTY))
    with optional("demo", "Inpainting people out of the scene"):
        from diffusers import FluxTransformer2DModel
        from omnieraser import FluxControlRemovalPipeline

    base = "black-forest-labs/FLUX.1-dev"
    transformer = FluxTransformer2DModel.from_pretrained(
        base, subfolder="transformer", torch_dtype=torch.bfloat16)

    # OmniEraser conditions on image + mask alongside the latent, so the patch
    # embedder takes 4x the channels. The extra columns start at zero, which makes
    # the widened layer exactly equivalent to the original at initialisation.
    with torch.no_grad():
        in_channels = transformer.config.in_channels
        wide = torch.nn.Linear(
            transformer.x_embedder.in_features * 4,
            transformer.x_embedder.out_features,
            bias=transformer.x_embedder.bias is not None,
            dtype=transformer.dtype, device=transformer.device,
        )
        wide.weight.zero_()
        wide.weight[:, :in_channels].copy_(transformer.x_embedder.weight)
        if transformer.x_embedder.bias is not None:
            wide.bias.copy_(transformer.x_embedder.bias)
        transformer.x_embedder = wide
        transformer.register_to_config(in_channels=in_channels * 4)

    pipe = FluxControlRemovalPipeline.from_pretrained(
        base, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.transformer.to(torch.bfloat16)
    pipe.load_lora_weights("theSure/Omnieraser",
                           weight_name="pytorch_lora_weights.safetensors")
    if strategy == "model":
        pipe.enable_model_cpu_offload(device=device)
    elif strategy == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        # "park" is moved on and off the GPU per call by remove_people;
        # "resident" simply stays there.
        pipe.to("cpu" if strategy == "park" else device)
    _pipe, _pipe_key = pipe, key
    return _pipe


def person_mask(image: ImageLike, prompt: str = PERSON_PROMPT) -> Image.Image:
    """Binary mask (mode ``L``, 255 = person) covering everyone in the image."""
    pil = to_pil(image)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        results = _load_sam().predict([pil], [prompt])
    masks = results[0]["masks"]
    if len(masks) == 0:
        return Image.fromarray(np.zeros(pil.size[::-1], np.uint8), mode="L")
    merged = np.stack(masks, axis=0).any(axis=0)
    return Image.fromarray((merged * 255).astype(np.uint8), mode="L")


def remove_people(
    image: ImageLike,
    mask: Optional[Image.Image] = None,
    *,
    device: str = "cuda",
    offload: Union[str, bool] = "auto",
    steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int = 42,
) -> Image.Image:
    """Return `image` with the people inpainted away, ready to pass as ``scene_image``.

    Args:
        image: the photo containing the people.
        mask: person mask; computed with :func:`person_mask` if omitted.
        offload: how to fit the pipeline into VRAM. "auto" (the default) reads
            the free memory on `device` and picks the fastest that fits:

            ============  =========  ============================================
            "park"        ~35 GiB    whole pipeline on the GPU for the call,
                                     parked on the CPU between calls
            "model"       ~26 GiB    one component on the GPU at a time
            "sequential"  a few GiB  streams submodules; fits anywhere, slow
            ============  =========  ============================================

            "resident" keeps the whole pipeline on the GPU permanently, which is
            only worth it when nothing else shares the card. `True` and `False`
            are accepted as "park" and "resident".
        steps, guidance_scale, seed: inpainting settings. The defaults are the
            ones used for the paper's qualitative results.
    """
    pil = to_pil(image)
    mask = mask if mask is not None else person_mask(pil)
    strategy = _resolve_offload(offload, device)
    pipe = _load_inpainter(device=device, strategy=strategy)
    try:
        if strategy == "park":
            pipe.to(device)
        # OmniEraser is trained at 1024x1024; go there and come back.
        size = pil.size
        result = pipe(
            prompt=INPAINT_PROMPT,
            control_image=pil.convert("RGB").resize((1024, 1024), Image.LANCZOS),
            control_mask=mask.convert("RGB").resize((1024, 1024), Image.LANCZOS),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=device).manual_seed(seed),
            max_sequence_length=512,
            height=1024, width=1024,
        ).images[0]
    finally:
        # "model"/"sequential" leave residency to accelerate's hooks: .to()
        # raises on a sequentially-offloaded pipeline and warns on a
        # model-offloaded one, so only "park" is moved by hand.
        if strategy == "park":
            pipe.to("cpu")
            torch.cuda.empty_cache()
    return result.resize(size, Image.LANCZOS)
