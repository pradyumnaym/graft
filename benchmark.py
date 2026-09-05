"""Measure GRAFT's inference latency, with a per-stage breakdown.

    python benchmark.py --image photo.jpg --scene-image empty_room.jpg
    python benchmark.py --image photo.jpg --runs 50 --budget 200

Latency is measured from prepared input tensors onwards: the images are decoded
and resized once, before the timed loop. That is the model's cost, and it is what
you can compare against another method or another GPU. File decoding is I/O whose
cost depends on the resolution of whatever you happened to hand it, not on GRAFT.

Exits non-zero if the median exceeds `--budget` milliseconds, so it can gate CI.

`--warmup` covers the first-call costs: the batched body fit is compiled on first
use, cuDNN autotunes, and the backbones are built on the first forward. Every
measurement is bracketed by a device synchronise. Stage times sum to slightly less
than the total; the remainder is Python glue.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import click


class Stages:
    """Accumulates wall-clock time per named stage across one call."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.times: dict[str, float] = {}

    @contextmanager
    def record(self, name: str):
        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - t0) * 1e3


def _timed(stages: Stages, name: str, fn, *a, **k):
    with stages.record(name):
        return fn(*a, **k)


def instrument(model, stages: Stages):
    """Wrap the model's stages in place. Returns a hook to call after warm-up,
    for the backbones that are only constructed on the first forward."""
    import graft.model as gm

    ma = model._ext["model_ma"]
    ma_infer = ma.infer
    ma.infer = lambda *a, **k: _timed(stages, "geometry backbone (MapAnything)", ma_infer, *a, **k)

    detect = gm.detect_parametric_fast
    gm.detect_parametric_fast = lambda *a, **k: _timed(
        stages, "pose init (NLF + body fit)", detect, *a, **k)

    refine = model._refine_once
    model._refine_once = lambda *a, **k: _timed(stages, "refinement", refine, *a, **k)

    pose = model.smplx_forward
    model.smplx_forward = lambda *a, **k: _timed(stages, "mesh posing (SMPL-X)", pose, *a, **k)

    proc = model._process_input
    model._process_input = lambda *a, **k: _timed(stages, "_input_total", proc, *a, **k)

    def wrap_moge():
        moge = model._ext.get("model_moge")
        if moge is not None and not getattr(moge, "_graft_wrapped", False):
            inner = moge.infer
            moge.infer = lambda *a, **k: _timed(stages, "metric scale (MoGe)", inner, *a, **k)
            moge._graft_wrapped = True
    return wrap_moge


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--image", type=click.Path(exists=True), required=True,
              help="A photo containing at least one person.")
@click.option("--scene-image", type=click.Path(exists=True), default=None,
              help="The empty-scene view. Defaults to reusing --image, which is the "
                   "same amount of work and so does not change the timing.")
@click.option("--model", "model_name", default="graft", show_default=True)
@click.option("--runs", default=30, show_default=True, help="Timed iterations.")
@click.option("--warmup", default=5, show_default=True,
              help="Untimed iterations. Must be >=2: the body fit is compiled on first use.")
@click.option("--num-iters", default=3, show_default=True, help="Refinement iterations.")
@click.option("--refine-hands", is_flag=True, help="Include WiLoR hand refinement.")
@click.option("--budget", default=200.0, show_default=True,
              help="Fail if the median exceeds this many milliseconds.")
@click.option("--data-root", type=click.Path(), default=None, help="Overrides $GRAFT_DATA_ROOT.")
@click.option("--device", default="cuda", show_default=True)
def main(image, scene_image, model_name, runs, warmup, num_iters, refine_hands,
         budget, data_root, device):
    if data_root:
        os.environ["GRAFT_DATA_ROOT"] = str(Path(data_root).expanduser().resolve())

    import torch
    from PIL import Image
    from graft import GRAFTModel
    from graft.ops.images import prepare_image

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device; latency on CPU is not meaningful here")

    photo = Image.open(image).convert("RGB")
    scene = Image.open(scene_image).convert("RGB") if scene_image else photo

    model = GRAFTModel.from_pretrained(model_name).to(device).eval()

    # Prepared once, outside the timed loop: decoding and resizing the file is not
    # part of the model's cost.
    image_t = prepare_image(photo, device=device)
    scene_t = prepare_image(scene, device=device)
    batch = {"image": image_t, "scene_image": scene_t}
    infer_kwargs = dict(num_iters=num_iters, use_moge=False, use_moge_scale=True,
                        use_wilor_hands=refine_hands)

    print(f"device : {torch.cuda.get_device_name(0)}")
    print(f"input  : {Path(image).name}  {photo.size[0]}x{photo.size[1]}"
          f"  ->  {tuple(image_t.shape[2:])} prepared")
    print(f"model  : {model_name}   refinement iterations: {num_iters}"
          f"{'   hand refinement: on' if refine_hands else ''}\n")

    stages = Stages()
    wrap_moge = instrument(model, stages)

    def run_once():
        out = model.infer(dict(batch), **infer_kwargs)
        valid = out["valid_humans"][0]
        model.smplx_forward({k: v[0][valid] for k, v in out["smplx_data"].items()})
        return int(valid.sum())

    print(f"warming up ({warmup} runs; the first compiles the body fit)…", flush=True)
    for i in range(warmup):
        n_people = run_once()
        if i == 0:
            wrap_moge()
    print(f"people detected: {n_people}\n")

    totals, per_stage = [], {}
    for _ in range(runs):
        stages.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        totals.append((time.perf_counter() - t0) * 1e3)
        for k, v in stages.times.items():
            per_stage.setdefault(k, []).append(v)

    totals.sort()
    median = statistics.median(totals)
    p90 = totals[int(0.9 * (len(totals) - 1))]

    med = {k: statistics.median(v) for k, v in per_stage.items()}
    # `_process_input` contains the backbone, MoGe and pose-init calls; what is left
    # of it is the alignment work (normals, head anchor, projection, SMPL-X forward).
    inner = sum(med.get(k, 0.0) for k in ("geometry backbone (MapAnything)",
                                          "metric scale (MoGe)", "pose init (NLF + body fit)"))
    med["scene alignment"] = max(med.get("_input_total", 0.0) - inner, 0.0)

    print(f"{'stage':34s} {'median ms':>10s} {'share':>7s}")
    print("-" * 53)
    accounted = 0.0
    for name in ("geometry backbone (MapAnything)", "metric scale (MoGe)",
                 "pose init (NLF + body fit)", "scene alignment", "refinement",
                 "mesh posing (SMPL-X)"):
        if name not in med:
            continue
        accounted += med[name]
        print(f"{name:34s} {med[name]:10.1f} {100*med[name]/median:6.1f}%")
    other = median - accounted
    print(f"{'other (Python glue)':34s} {other:10.1f} {100*other/median:6.1f}%")
    print("-" * 53)
    print(f"{'TOTAL':34s} {median:10.1f} {100.0:6.1f}%")

    print(f"\nlatency: median {median:.1f} ms   p90 {p90:.1f} ms   "
          f"min {totals[0]:.1f} ms   max {totals[-1]:.1f} ms   over {runs} runs")
    print(f"throughput: {1000/median:.1f} images/s")

    if median > budget:
        print(f"\nFAIL: median {median:.1f} ms exceeds the {budget:.0f} ms budget")
        return 1
    print(f"\nOK: median {median:.1f} ms is within the {budget:.0f} ms budget")
    return 0


if __name__ == "__main__":
    sys.exit(main(standalone_mode=False) or 0)
