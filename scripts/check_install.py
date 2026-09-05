"""Check that this environment can actually run GRAFT.

    python scripts/check_install.py           # core + data root
    python scripts/check_install.py --all     # also the eval and demo extras

Imports each dependency and reports what is missing per install extra, then
checks the data root. Exits non-zero if anything the core needs is absent, so it
answers "is my install complete?" without waiting for a crash mid-inference.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# (import name, what it is for). Grouped by the extra that provides it.
GROUPS = {
    "core": [
        ("torch", "everything"),
        ("torchvision", "image ops"),
        ("numpy", "arrays"),
        ("PIL", "image loading"),
        ("cv2", "image ops"),
        ("scipy", "the NLF vertex-subset lookup"),
        ("einops", "the refinement transformer"),
        ("trimesh", "mesh export in demo.py"),
        ("click", "the command-line scripts"),
        ("roma", "rotation conversions"),
        ("smplx", "the SMPL-X body model"),
        ("smplfitter", "the batched body fit"),
        ("transformers", "the geometry backbone's encoder"),
        ("huggingface_hub", "fetching backbone weights"),
        ("uniception", "the geometry backbone's transformer blocks"),
        ("moge", "metric scale"),
        # Reached through graft._vendor, which puts third_party/ on sys.path.
        ("graft._vendor", "the geometry backbone (vendored under third_party/)"),
    ],
    "eval": [
        ("open3d", "closest-point queries in the contact metric"),
        ("sklearn", "precision / recall / F1"),
        ("lang_sam", "floor and person segmentation"),
        ("tqdm", "progress bars"),
    ],
    "demo": [
        ("wilor_mini", "hand refinement (refine_hands=True)"),
        ("skimage", "the hand-crop resampling"),
        ("diffusers", "scene inpainting"),
        ("accelerate", "the inpainter's CPU-offload strategies"),
        ("peft", "the inpainter's LoRA weights"),
        ("sentencepiece", "the inpainter's text encoder"),
        ("lang_sam", "person segmentation"),
    ],
}


def check_group(name: str) -> list[str]:
    how = "pip install -e ." if name == "core" else 'pip install -e ".[%s]"' % name
    print(f"\n[{name}]  pyproject.toml  ({how})")
    missing = []
    for mod, why in GROUPS[name]:
        try:
            importlib.import_module(mod)
            print(f"  ok       {mod:18s} {why}")
        except Exception as e:                      # ImportError, but also broken builds
            kind = type(e).__name__
            print(f"  MISSING  {mod:18s} {why}   ({kind}: {e})")
            missing.append(mod)
    return missing


#: SMPL-X v1.1 ships 400 bases in `shapedirs` (300 shape + 100 expression);
#: v1.0 ships 20. The published numbers were produced with v1.1, and the body
#: model is on the inference path, so a v1.0 file changes results silently.
SMPLX_V11_SHAPEDIRS = 400


def check_body_model(path: Path) -> str | None:
    """Confirm the SMPL-X file is the version the numbers were produced with.

    Returns a warning string, or None if it looks right. Note the `version.txt`
    that ships alongside these models is not reliable -- read the arrays instead.
    """
    try:
        import numpy as np
        with np.load(path, allow_pickle=True) as d:
            if "shapedirs" not in d.files:
                return f"{path.name} has no 'shapedirs'; is it a SMPL-X model?"
            n = d["shapedirs"].shape[-1]
            v = d["v_template"].shape if "v_template" in d.files else None
        if v != (10475, 3):
            return f"{path.name} has v_template {v}, expected (10475, 3)"
        if n != SMPLX_V11_SHAPEDIRS:
            return (f"{path.name} has {n} shape bases, expected "
                    f"{SMPLX_V11_SHAPEDIRS} (SMPL-X v1.1). This looks like a "
                    f"different SMPL-X version; results will not match the paper.")
        return None
    except Exception as e:
        return f"could not read {path.name}: {type(e).__name__}: {e}"


def check_data(want_eval: bool) -> list[str]:
    from graft import paths
    print(f"\n[data]  {paths.data_root()}")
    items = [
        (paths.smplx_neutral(), "SMPL-X neutral body model"),
        (paths.body_model_dir() / "SMPLX_MALE.npz", "SMPL-X male body model"),
        # smplfitter loads this for every smpl-family model; without it the first
        # BodyModel(...) raises and inference never starts.
        (paths.body_model_dir() / "kid_template.npy", "SMPL-X kid template"),
        (paths.nlf_torchscript(), "NLF pose detector"),
        (paths.checkpoint_dir() / "graft.pt", "GRAFT checkpoint"),
        (paths.checkpoint_dir() / "graft_no_visual.pt", "ablation checkpoint"),
    ]
    if want_eval:
        d = paths.data_root()
        items += [(d / n, "benchmark support file") for n in (
            "scene_image.png", "prox_quantitative_depth.pkl", "contact_vertice_weights.pkl")]
    missing = []
    for path, what in items:
        if path.exists():
            note = ""
            if path.suffix == ".npz" and "body model" in what:
                warn = check_body_model(path)
                note = f"\n             WARNING: {warn}" if warn else "   (SMPL-X v1.1)"
            print(f"  ok       {what}{note}")
        else:
            print(f"  MISSING  {what}\n             {path}")
            missing.append(str(path))
    return missing


def check_versions() -> None:
    """Report where this environment differs from the one the numbers came from.

    Not a failure: GRAFT will run on other versions. But if a benchmark number
    does not reproduce, this is the first place to look.
    """
    import importlib.metadata as md
    lock = Path(__file__).resolve().parent.parent / "requirements-lock.txt"
    if not lock.exists():
        return
    drift = []
    for line in lock.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, want = line.split("==")
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            continue
        if got != want:
            drift.append((name, want, got))
    print("\n[versions]  vs requirements-lock.txt")
    if not drift:
        print("  identical to the environment the published numbers were produced in")
        return
    for name, want, got in drift:
        print(f"  differs  {name:18s} verified {want}  installed {got}")
    print("  (not an error; consult this first if a number does not reproduce)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Also check the eval and demo extras.")
    args = ap.parse_args()

    core_missing = check_group("core")
    extra_missing = {}
    if args.all:
        for g in ("eval", "demo"):
            m = check_group(g)
            if m:
                extra_missing[g] = m

    if not core_missing:
        check_versions()

    data_missing = []
    if not core_missing:
        try:
            data_missing = check_data(args.all)
        except Exception as e:
            print(f"\n[data]  could not resolve the data root: {e}")
            data_missing = ["<unresolved>"]

    print()
    if core_missing:
        print(f"core install INCOMPLETE — missing: {', '.join(core_missing)}")
        if core_missing == ["graft._vendor"]:
            print("  third_party/mapanything is missing or broken — re-clone the repository.")
        else:
            print("  bash install.sh")
    else:
        print("core install OK")
    for g, mods in extra_missing.items():
        print(f"'{g}' extra incomplete — missing: {', '.join(mods)}")
        print(f'  pip install -e ".[{g}]"')
    if data_missing:
        print(f"data root incomplete — {len(data_missing)} file(s) missing")
        print("  python scripts/download_assets.py" + (" --eval" if args.all else ""))
    elif not core_missing:
        print("data root OK")

    ready = not core_missing and not data_missing
    print("\nGRAFT is ready to run." if ready else "\nGRAFT is NOT ready to run.")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
