"""Reproduce GRAFT's published PROX and RICH numbers.

    python evaluate.py                       # both checkpoints, both benchmarks
    python evaluate.py --model graft
    python evaluate.py --benchmark prox --limit 10      # quick check

Results are written to ``outputs/<model>_<benchmark>.json`` and printed as a
table. Six metrics per benchmark: PA-MPJPE, vertex-to-scene L2 and angle, and
contact precision / recall / F1.

Needs two roots:

* ``$GRAFT_DATA_ROOT``      -- body models, NLF, checkpoints and the benchmark
  support files (``scripts/download_assets.py`` populates it).
* ``$GRAFT_BENCHMARK_ROOT`` -- the datasets themselves, i.e. a directory holding
  ``PROX/quantitative``, ``PROX/proxd_full`` and ``PROX/rich-sample100``. These
  are distributed by their own authors under their own licences; see the README.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--model", "models", multiple=True,
              help="Checkpoint name or path. Repeatable. Default: both published checkpoints.")
@click.option("--benchmark", "benchmarks", multiple=True,
              type=click.Choice(["prox", "rich"]),
              help="Repeatable. Default: both.")
@click.option("--data-root", type=click.Path(), default=None,
              help="Overrides $GRAFT_DATA_ROOT.")
@click.option("--benchmark-root", type=click.Path(), default=None,
              help="Overrides $GRAFT_BENCHMARK_ROOT.")
@click.option("--out-dir", type=click.Path(), default="outputs", show_default=True)
@click.option("--limit", type=int, default=None,
              help="Evaluate only the first N frames (smoke test; not comparable to the paper).")
@click.option("--device", default="cuda", show_default=True)
def main(models, benchmarks, data_root, benchmark_root, out_dir, limit, device):
    # The data root must be set before `graft` is imported: several benchmark
    # asset paths are default argument values, resolved at import time.
    if data_root:
        os.environ["GRAFT_DATA_ROOT"] = str(Path(data_root).expanduser().resolve())
    bench_root = benchmark_root or os.environ.get("GRAFT_BENCHMARK_ROOT")
    if not bench_root:
        raise SystemExit(
            "Set $GRAFT_BENCHMARK_ROOT (or pass --benchmark-root) to the directory "
            "holding PROX/quantitative, PROX/proxd_full and PROX/rich-sample100."
        )

    from graft.eval import set_deterministic
    set_deterministic(seed=42, single_thread=False)

    import torch
    from graft import PRETRAINED, GRAFTModel
    from graft.eval import (build_prox_context, build_rich_context,
                            default_infer_kwargs, run_prox_eval, run_rich_eval)

    models = list(models) or list(PRETRAINED)
    benchmarks = list(benchmarks) or ["prox", "rich"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # SAM is shared: PROX uses it once for the floor mask, RICH once per scene.
    print("Loading LangSAM…", flush=True)
    from lang_sam import LangSAM
    sam = LangSAM(sam_type="sam2.1_hiera_large")

    print("Building benchmark contexts…", flush=True)
    prox_ctx = build_prox_context(benchmark_root=bench_root, sam_model=sam) if "prox" in benchmarks else None
    rich_ctx = build_rich_context(benchmark_root=bench_root, sam_model=sam) if "rich" in benchmarks else None

    rows = []
    for name in models:
        print(f"\n=== {name} ===", flush=True)
        model = GRAFTModel.from_pretrained(name).to(device).eval()
        kwargs = default_infer_kwargs(model)
        for bench in benchmarks:
            runner, ctx = (run_prox_eval, prox_ctx) if bench == "prox" else (run_rich_eval, rich_ctx)
            metrics = runner(model, ctx, infer_kwargs=kwargs, limit=limit)
            metrics["model"] = name
            metrics["benchmark"] = bench
            (out / f"{Path(name).stem}_{bench}.json").write_text(json.dumps(metrics, indent=2))
            rows.append(metrics)
            print(f"  {bench}: PA {metrics['pa_mpjpe_mm']:.2f} mm | "
                  f"V2S {metrics['v2s_l2_mm']:.2f} mm / {metrics['v2s_angle_deg']:.2f}° | "
                  f"F1 {metrics['f1']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    print()
    header = f"{'model':18s} {'bench':6s} {'PA-MPJPE':>9s} {'V2S L2':>8s} {'V2S deg':>8s} " \
             f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s}  frames"
    print(header)
    print("-" * len(header))
    for m in rows:
        print(f"{m['model']:18s} {m['benchmark']:6s} {m['pa_mpjpe_mm']:9.2f} "
              f"{m['v2s_l2_mm']:8.2f} {m['v2s_angle_deg']:8.2f} {m['precision']:6.3f} "
              f"{m['recall']:6.3f} {m['f1']:6.3f}  {m['frames']}")
    print(f"\nWrote {len(rows)} result file(s) to {out}/")


if __name__ == "__main__":
    main()
