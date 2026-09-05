"""Rebuild `rich-sample100` from an official RICH download.

    python scripts/build_rich_sample100.py --rich-root /path/to/RICH --out <dir>

RICH is licence-gated (register at https://rich.is.tue.mpg.de), so this
repository ships only the 100-frame *manifest*, not the frames. This script
assembles the benchmark directory from your own copy.

What it produces, and where each piece comes from:

    <out>/sample/{seq}+{frame}_{cam}.jpeg   <- the RICH multi-view image, re-encoded
    <out>/sample/{seq}+{frame}_gt.pkl       <- {split}_body/{seq}/{frame}/{subj}.pkl  (copied verbatim)
    <out>/sample/{seq}+{frame}_gt.ply       <- {split}_body/{seq}/{frame}/{subj}.ply  (copied verbatim)
    <out>/calibration/{scene}/calibration/{cam:03d}.xml   <- scan_calibration (copied verbatim)
    <out>/scene/scan_calibration/{scene}/*_simp.ply       <- scene scans, decimated to 5%
    <out>/multicam2world/{scene}_multicam2world.json      <- copied verbatim

The empty-scene images the model actually consumes are NOT built here: they are
ours, not RICH's, and ship with the release
(`python scripts/download_assets.py --eval`).

See docs/EVALUATION.md for the whole picture, including the caveats.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "docs" / "rich_sample100_manifest.json"

#: The benchmark's frames are byte-identical copies of RICH's JPEG release
#: (`JPG_images/{split}.tar.gz`) -- verified for all 100. So the correct operation
#: is a copy, and re-encoding is what breaks reproduction. These settings are only
#: used under --allow-reencode, for someone who has the full-resolution BMP/PNG
#: release instead; the result will NOT match the published numbers.
JPEG_QUALITY = 90
JPEG_SUBSAMPLING = 2   # 4:2:0

#: Activity -> the scene scan it is evaluated against. Mirrors
#: graft/eval/rich.py::ACTIVITY_TO_SCENE, which is the authority.
ACTIVITY_TO_SCENE = {
    "cooking1": "scan_table_camcoord.ply",
    "dips1": "scan_camcoord.ply", "dips2": "scan_camcoord.ply",
    "lunge1": "scan_camcoord.ply", "lunge2": "scan_camcoord.ply",
    "mountainclimber1": "scan_camcoord.ply", "mountainclimber2": "scan_camcoord.ply",
    "pushup1": "scan_camcoord.ply", "pushup2": "scan_camcoord.ply",
    "plankjack1": "scan_yoga_scene_camcoord.ply",
    "sidebalancerun1": "scan_yoga_scene_camcoord.ply",
}

#: Keep-ratio for the scene scans. Verified against the reference subset: at 0.05
#: the decimated Gym scans match it exactly in both vertex and face count
#: (8,788,693 -> 439,434 faces and 8,843,334 -> 442,165).
DECIMATION_RATIO = 0.05


#: Where the frames sit inside a RICH download, and what they are called there.
#: RICH ships the frames as `JPG_images/{split}.tar.gz`, which extracts to
#:     images/{split}/{sequence}/cam_{NN}/{frame}_{NN}.jpeg
#: e.g. images/test/Gym_010_cooking1/cam_03/00032_03.jpeg
#: Override with --image-dir / --image-glob if your extraction differs.
DEFAULT_IMAGE_DIR = "images/{split}/{seq}/cam_{cam}"
DEFAULT_IMAGE_GLOB = "{frame}_{cam}.jpeg"


def find_first(root: Path, candidates) -> Path | None:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    return None


def locate_image(images_root: Path, split: str, seq: str, frame: str, cam: str,
                 image_dir: str, image_glob: str):
    """Resolve one frame's image. Returns (path, tried) -- never searches the tree.

    Looking through a RICH download frame by frame is what makes this slow and
    fragile, so this only ever lists ONE directory.
    """
    d = images_root / image_dir.format(split=split, seq=seq, cam=cam, frame=frame)
    if not d.is_dir():
        return None, str(d)
    hits = sorted(d.glob(image_glob.format(frame=frame, cam=cam, seq=seq)))
    return (hits[0] if hits else None), str(d / image_glob.format(frame=frame, cam=cam, seq=seq))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rich-root", required=True, type=Path,
                    help="Your RICH download (contains test_body/ and scan_calibration/).")
    ap.add_argument("--out", type=Path, default=Path("rich-sample100"))
    ap.add_argument("--images-root", type=Path, default=None,
                    help="Directory holding the extracted RICH images. Defaults to "
                         "--rich-root. Omit --images-root and pass --skip-images to "
                         "build everything except the frames.")
    ap.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR,
                    help=f"Per-camera directory template (default: {DEFAULT_IMAGE_DIR}).")
    ap.add_argument("--image-glob", default=DEFAULT_IMAGE_GLOB,
                    help=f"Filename glob within that directory (default: {DEFAULT_IMAGE_GLOB}).")
    ap.add_argument("--allow-reencode", action="store_true",
                    help="Permit building frames from the full-resolution BMP/PNG "
                         "release. This does NOT reproduce the published numbers; the "
                         "benchmark frames are copies of RICH's JPEG release.")
    ap.add_argument("--skip-images", action="store_true",
                    help="Do not copy frames; build the ground truth and calibration only.")
    ap.add_argument("--skip-meshes", action="store_true",
                    help="Skip decimating the scene scans. The result CANNOT be evaluated: "
                         "the loader opens the scan for every frame. For staged builds only.")
    args = ap.parse_args()

    spec = json.loads(MANIFEST.read_text())
    frames = spec["frames"]
    split = spec["split"]
    rich, out = args.rich_root.expanduser(), args.out.expanduser()
    images_root = (args.images_root or rich).expanduser()

    body_root = find_first(rich, [f"{split}_body", f"data/{split}_body"])
    calib_root = find_first(rich, ["scan_calibration", "data/scan_calibration"])
    if body_root is None or calib_root is None:
        raise SystemExit(
            f"could not find {split}_body/ and scan_calibration/ under {rich}.\n"
            f"Point --rich-root at the directory that contains them.")

    (out / "sample").mkdir(parents=True, exist_ok=True)
    from PIL import Image

    missing = []
    for i, f in enumerate(frames, 1):
        seq, frame, cam, subj = f["sequence"], f["frame"], f["camera"], f["subject"]
        stem = f"{seq}+{frame}"

        for ext in ("pkl", "ply"):
            src = body_root / seq / frame / f"{subj}.{ext}"
            if not src.exists():
                missing.append(str(src)); continue
            shutil.copyfile(src, out / "sample" / f"{stem}_gt.{ext}")

        dst_img = out / "sample" / f"{stem}_{cam}.jpeg"
        if not args.skip_images and not dst_img.exists():
            src_img, tried = locate_image(images_root, split, seq, frame, cam,
                                          args.image_dir, args.image_glob)
            if src_img is None:
                missing.append(f"image {seq} {frame} cam {cam}  (looked for {tried})")
                continue
            if src_img.suffix.lower() in (".jpg", ".jpeg"):
                shutil.copyfile(src_img, dst_img)     # the published frames are this, verbatim
            elif args.allow_reencode:
                Image.open(src_img).convert("RGB").save(
                    dst_img, "JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
            else:
                missing.append(
                    f"{src_img} is not a JPEG. The benchmark frames are byte-identical "
                    f"copies of RICH's JPG_images release; use that, or pass "
                    f"--allow-reencode to accept different pixels.")
                continue
        if i % 20 == 0:
            print(f"  {i}/{len(frames)} frames", flush=True)

    scenes = sorted({f["scene"] for f in frames})
    for scene in scenes:
        d = out / "calibration" / scene / "calibration"
        d.mkdir(parents=True, exist_ok=True)
        src_dir = calib_root / scene / "calibration"
        if not src_dir.is_dir():
            missing.append(str(src_dir)); continue
        for xml in src_dir.glob("*.xml"):
            shutil.copyfile(xml, d / xml.name)
        for j in (calib_root / scene).glob("*multicam2world*.json"):
            (out / "multicam2world").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(j, out / "multicam2world" / j.name)

    if not args.skip_meshes:
        needed = {(f["scene"], ACTIVITY_TO_SCENE[f["activity"]]) for f in frames}
        for scene, ply in sorted(needed):
            dst_dir = out / "scene" / "scan_calibration" / scene
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / ply.replace(".ply", "_simp.ply")
            if dst.exists():
                continue
            src = calib_root / scene / ply
            if not src.exists():
                missing.append(str(src)); continue
            print(f"  decimating {scene}/{ply} (keep {DECIMATION_RATIO:.0%})…", flush=True)
            try:
                import fast_simplification
                import trimesh
                m = trimesh.load(src, process=False)
                v, fc = fast_simplification.simplify(
                    m.vertices, m.faces, target_reduction=1.0 - DECIMATION_RATIO)
                trimesh.Trimesh(v, fc, process=False).export(dst)
            except ImportError:
                missing.append(f"{dst} (needs `pip install fast-simplification`)")

    print()
    if missing:
        print(f"{len(missing)} item(s) could not be produced:")
        for m in missing[:20]:
            print(f"  {m}")
        if len(missing) > 20:
            print(f"  … and {len(missing) - 20} more")
        print("\nrich-sample100 is INCOMPLETE.")
        return 1
    if args.skip_images or args.skip_meshes:
        skipped = " and ".join(x for x, on in
                               [("frames", args.skip_images), ("scene scans", args.skip_meshes)] if on)
        print(f"INCOMPLETE: built without {skipped}. The evaluation opens both for every "
              f"frame, so re-run without --skip-* before using this.\n")
    print(f"rich-sample100 built at {out}")
    print("Set $GRAFT_BENCHMARK_ROOT so that $GRAFT_BENCHMARK_ROOT/PROX/rich-sample100 "
          "is this directory, then run `python evaluate.py --benchmark rich`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
