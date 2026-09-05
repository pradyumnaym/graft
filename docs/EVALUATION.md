# Evaluation: the PROX and RICH benchmarks

GRAFT is scored on two benchmarks. This document covers both: what to download,
how to lay it out, what we supply, and what to watch for.

| | PROX-quantitative | RICH |
|---|---|---|
| frames iterated / scored | 178 / 174 | 100 / 94 |
| you download | one 1.2 GB archive | three files, 250 GB |
| construction needed | none — used as it extracts | a 100-frame subset, built by a script |
| ground truth | MoSh fits (male) | SMPL-X + per-frame body mesh |

```bash
export GRAFT_BENCHMARK_ROOT=/data     # the parent of PROX/
python evaluate.py                    # both benchmarks, both checkpoints
```

Both need the `eval` extra, which `bash install.sh` installs by default
(`pip install -e ".[eval]"` on its own): LangSAM for segmentation and open3d for
the contact metric.

---

## What we supply

Everything that is ours rather than the datasets' ships with the release and is
fetched by `python scripts/download_assets.py --eval`:

| file | used by | what it is |
|---|---|---|
| `scene_image.png` | PROX | the empty-scene view GRAFT consumes as its second input |
| `imgs/rich/scene_imgs/*.png` | RICH | the same, per (sequence, camera) — 33 files |
| `prox_quantitative_depth.pkl` | PROX | precomputed scene depth |
| `contact_vertice_weights.pkl` | both | per-vertex weights for the V2S metric |

They come from [`ympradyumna/graft`](https://huggingface.co/ympradyumna/graft) on
the Hugging Face Hub.

The empty-scene images are **our artifact, not the datasets'**. Do not try to
regenerate them: a later pipeline produces visibly different images, which changes
the numbers.

Ground-truth contacts are **not** shipped — they are computed from the scene
geometry on the first run and cached.

---

# PROX

Register at <https://prox.is.tue.mpg.de> and take **`quantitative.zip`** (~1.2 GB
extracted). You do *not* need the main `proxd` release. Extract to:

```
$GRAFT_BENCHMARK_ROOT/PROX/quantitative/
├── recordings/vicon_03301_*/Color/*.jpg          178 frames, 18 recordings
├── fittings/mosh/<recording>/results/<frame>/000.pkl    MoSh ground truth
├── cam2world/vicon.json
├── vicon2scene.json
├── scenes/vicon.ply                              the scene mesh
└── calibration/Color.json
```

The loader reads intrinsics from `PROX/proxd_full/calibration/Color.json`, a path
from the full release. That file is byte-identical to the one in the quantitative
archive (md5 `8deefedf1d891adc320d2b018dcf9174`), so copy it rather than
downloading the full dataset:

```bash
mkdir -p $GRAFT_BENCHMARK_ROOT/PROX/proxd_full/calibration
cp $GRAFT_BENCHMARK_ROOT/PROX/quantitative/calibration/Color.json \
   $GRAFT_BENCHMARK_ROOT/PROX/proxd_full/calibration/
```

PROX's ground truth is **gendered** — the MoSh fits are male — so the evaluation
loads `SMPLX_MALE.npz` as well as the neutral model. Both must be in the data root.

Four of the 178 frames yield no single-person prediction and are dropped; the
`skipped` field in the result JSON reports this.

---

# RICH

The RICH benchmark is 100 specific frames, all from the **test** split, all of
subject `010`, across 11 sequences:

| scene | sequences | frames |
|---|---|---|
| `Gym` | cooking1, dips1, dips2, lunge1, lunge2, mountainclimber1, mountainclimber2, pushup1, pushup2 | 60 |
| `LectureHall` | plankjack1, sidebalancerun1 | 40 |

The full list is [`rich_sample100_manifest.json`](rich_sample100_manifest.json).
Six frames are dropped in scoring, so the metrics are over 94.

## What to download

Register at <https://rich.is.tue.mpg.de> and take three files:

| file | size | what it gives |
|---|---|---|
| `JPG_images/test.tar.gz` | **242.6 GB** | the RGB frames |
| `test_body.zip` | 7.3 GB | SMPL-X ground truth |
| `scan_calibration.zip` | 0.9 GB | camera XMLs, scene scans, multicam2world |

You do **not** need the human-scene contact labels (~23 GB): the benchmark
computes ground-truth contacts itself from the scene scan.

The images archive is unavoidably the whole 242.6 GB. It is a single gzip stream,
so there is no way to fetch only the 100 frames from it — gzip cannot be
decompressed from an arbitrary offset, and RICH publishes no per-sequence
archives. Extract it, build the subset (161 MB), then delete the tarball.

Extract so that you have:

```
<RICH>/
├── images/test/{sequence}/cam_{NN}/{frame}_{NN}.jpeg
├── test_body/{sequence}/{frame}/{subject}.{pkl,ply}
└── scan_calibration/{scene}/
    ├── calibration/{NNN}.xml
    └── *.ply
```

Then build the benchmark directory:

```bash
python scripts/build_rich_sample100.py \
    --rich-root /path/to/RICH \
    --out       $GRAFT_BENCHMARK_ROOT/PROX/rich-sample100
```

If your extraction nests things differently, `--images-root`, `--image-dir` and
`--image-glob` adjust the lookup without editing code; the script prints the exact
path it expected whenever a frame is missing.

## The filename is the whole index

Every field the loader needs is in the name, which is why the manifest alone is
enough to rebuild the set. Taking `Gym_010_cooking1+00123_04.jpeg`:

```
Gym _ 010 _ cooking1 + 00123 _ 04 .jpeg
 │     │       │        │      │
 │     │       │        │      └─ camera id, 2 digits
 │     │       │        └──────── frame index, 5 digits
 │     │       └───────────────── activity
 │     └───────────────────────── subject id
 └─────────────────────────────── scene
```

The first three joined by `_` are RICH's sequence name (`Gym_010_cooking1`); the
`+` separates that from the frame. Each field resolves to a real RICH path:

| field | verified by |
|---|---|
| scene `Gym` | `scan_calibration/Gym/` exists |
| subject `010` | RICH stores that frame's ground truth as `test_body/Gym_010_cooking1/00123/`**`010`**`.pkl`, and our `_gt.pkl` is byte-identical to it |
| activity `cooking1` | selects the scene scan via `ACTIVITY_TO_SCENE` in `graft/eval/rich.py` |
| frame `00123` | `test_body/Gym_010_cooking1/` has 776 frame directories, one of which is `00123` |
| camera `04` | resolves to `calibration/Gym/calibration/004.xml` |

Two easy mistakes: the subject id sits between scene and activity and is *not*
part of either; and the camera id is **two** digits in the filename but **three**
in the calibration path, so `_04` reads `004.xml`.
`graft/eval/rich.py::_parse_frame_name` is the implementation and the authority.

## The layout the loader expects

```
$GRAFT_BENCHMARK_ROOT/PROX/rich-sample100/
├── sample/
│   ├── {sequence}+{frame}_{cam}.jpeg    the RGB frame
│   ├── {sequence}+{frame}_gt.pkl        SMPL-X ground truth
│   └── {sequence}+{frame}_gt.ply        ground-truth body mesh
├── calibration/{scene}/calibration/{NNN}.xml
└── scene/scan_calibration/{scene}/{scan}_simp.ply
```

## What is verified about the rebuild

`build_rich_sample100.py` was run end to end against a copy of RICH and its output
byte-compared against the reference subset:

| step | status |
|---|---|
| frames copied from RICH's JPEG release | **verified** — 100/100 byte-identical |
| `_gt.pkl` / `_gt.ply` from `test_body/{seq}/{frame}/{subject}.*` | **verified** — 200/200 |
| calibration XMLs from `scan_calibration/{scene}/calibration/` | **verified** — 14/14 |
| **total** | **314/314, zero differing** |
| the layout a fresh tarball extraction produces | evidenced, not tested |
| scene-scan decimation at keep-ratio 0.05 | **verified** — matches the reference exactly in vertex and face count |

Every file the benchmark reads is a verbatim copy of a RICH file, renamed. Nothing
is transformed and nothing is re-encoded.

The rebuild writes 314 files where the reference directory holds 411; the
difference is calibration for the four unused scenes (29), `multicam2world/` (7,
never referenced), a `.DS_Store`, and the scene scans (59) — those last are
required, and were skipped in that particular run only because it was testing the
copy paths.

---

## Two things that will bite you

**1. Use RICH's JPEG release, not the full-resolution one.** RICH ships both
`JPG_images/{split}.tar.gz` and a full-resolution BMP/PNG release. The benchmark's
frames are byte-identical copies of the JPEG release — all 100 verified. Nothing
is re-encoded, and nothing should be: `build_rich_sample100.py` copies the file
through unchanged and refuses a non-JPEG source unless you pass `--allow-reencode`.
Building from the full-resolution release produces different pixels and will not
reproduce the published numbers.

**2. Ground-truth contacts are computed on the first run, not downloaded.** Both
benchmarks cache them afterwards, so only the first run pays the few extra
minutes.

---

## Results

What this repository produces:

| model | benchmark | PA-MPJPE | V2S L2 | V2S angle | Prec | Rec | F1 | frames |
|---|---|---|---|---|---|---|---|---|
| `graft` | PROX | 49.73 | 184.36 | 50.92 | 0.556 | 0.638 | 0.594 | 174 |
| `graft` | RICH | 46.32 | 224.66 | 34.60 | 0.473 | 0.743 | 0.578 | 94 |
| `graft-no-visual` | PROX | 52.56 | 188.47 | 51.92 | 0.534 | 0.510 | 0.522 | 174 |
| `graft-no-visual` | RICH | 45.52 | 223.10 | 34.06 | 0.434 | 0.717 | 0.540 | 94 |

The `graft` rows are the published numbers, reproduced exactly. See the README for
what "exactly" is scoped to, and why the ablation rows differ from the paper's.

## Checking a rebuild

```bash
ls rich-sample100/sample/*.jpeg | wc -l     # 100
ls rich-sample100/sample/*_gt.ply | wc -l   # 100
python evaluate.py --benchmark rich --limit 5
```

A wrong rebuild fails loudly: a missing calibration XML, a GT mesh that will not
load, a frame the script could not find.
