<div align="center">

<img src="assets/graft_wobg.png" alt="GRAFT logo" width="80"/>

# GRAFT: Geometric Refinement and Fitting Transformer for Human Scene Reconstruction

<p>
  <a href="https://pradyumnaym.github.io/" target="_blank">Pradyumna YM</a> ·
  <a href="https://yuxuan-xue.com/" target="_blank">Yuxuan Xue</a> ·
  <a href="https://fanegg.github.io/" target="_blank">Yue Chen</a> ·
  <a href="https://virtualhumans.mpi-inf.mpg.de/people/Kister.html" target="_blank">Nikita Kister</a> ·
  <a href="https://istvansarandi.com/" target="_blank">István Sárándi</a> ·
  <a href="https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html" target="_blank">Gerard Pons-Moll</a>
</p>

<p>
  <a href="https://pradyumnaym.github.io/graft/"><img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-graft-blue?style=flat&logo=googlechrome"></a>
  <a href="https://arxiv.org/abs/2604.19624"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2604.19624-b31b1b?style=flat&logo=arxiv"></a>
  <a href="https://huggingface.co/ympradyumna/graft"><img alt="Weights" src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-graft-yellow?style=flat"></a>
</p>

</div>

**TL;DR** — A learned HSI prior that predicts iterative geometry-grounded refinements
to reconstruct humans in scenes from a single image. Matches optimization-based
contact quality at **~100× lower runtime**; as a plug-in prior it boosts existing
methods by up to **44%** contact F1 — no retraining needed.


https://github.com/user-attachments/assets/0ce2b017-afa6-49c5-a016-d007e1ef7a93


This repository is the code release: the model, the demo, and the harness that
reproduces the paper's PROX and RICH numbers.

```python
from graft import GRAFTModel

from graft.scene import person_mask

model = GRAFTModel.from_pretrained("graft").cuda().eval()
out = model.predict("photo.jpg", scene_image="empty_room.jpg",
                    person_mask=person_mask("photo.jpg"))

out.vertices     # (n_people, 10475, 3) SMPL-X vertices, scene frame, metres
out.smplx        # dict of SMPL-X parameters
out.points       # (H, W, 3) metric scene point map
```

GRAFT refines an existing per-frame pose estimate rather than detecting people
itself, so a photo with nobody in it raises `graft.NoPeopleDetected`.

---

## Roadmap

- [x] Project page
- [x] arXiv paper
- [x] Demo / inference code
- [x] Evaluation code and the PROX / RICH benchmark harness
- [ ] Training code

---

## Install

```bash
conda create -n graft python=3.10 && conda activate graft
bash install.sh                           # everything (--minimal for inference only)
python scripts/download_assets.py --eval  # weights and assets (drop --eval to skip benchmarks)
python scripts/check_install.py --all     # confirm the environment is ready
```

`install.sh` installs torch from the CUDA-matched index (`cu124`; override with
`TORCH_INDEX`), then the project and its extras one at a time, so a failing extra
still leaves a working inference install. `requirements-lock.txt` records the versions
the published numbers were produced with.

The download pulls ≈1.1 GB into `$GRAFT_DATA_ROOT` (default `./data`): our weights and
benchmark assets from [`ympradyumna/graft`](https://huggingface.co/ympradyumna/graft),
the NLF detector from its GitHub release, and the SMPL-X body models from an
account-gated portal — **register at <https://smpl-x.is.tue.mpg.de> first**, the only
account you need. Credentials are used for the download and not stored; `--skip-gated`
opts out. The benchmark *datasets* live separately under `$GRAFT_BENCHMARK_ROOT`, see
[`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Checkpoints

Both load into the same `GRAFTModel` and differ only in configuration.

| Name | What it is | Config difference |
|---|---|---|
| `graft` | the main model | — |
| `graft-no-visual` | GRAFT in **geometry-only mode** | `drop_vis_feats=True` |

`drop_vis_feats` pushes every image sample point out of bounds, so the refinement sees
scene geometry only — the mode the paper uses as a plug-in prior on top of other
methods, refining an existing reconstruction with no retraining and no image features.
The two also use different `flat_hand_mean` conventions, so `from_pretrained` needs an
explicit `GRAFTConfig` for a bare checkpoint path.

---

## The scene image

GRAFT takes **two views of the same camera**: the photo with the people, and a view
of the same scene *without* them, which is where the geometry comes from. Both
benchmarks supply a real one. For your own images, best first:

```python
from graft.scene import person_mask, remove_people
mask = person_mask("photo.jpg")                            # where the people are

model.predict("photo.jpg", scene_image="empty_room.jpg", person_mask=mask)   # a real empty-scene photo
model.predict("photo.jpg", scene_image=remove_people("photo.jpg", mask),     # or inpaint them out
              person_mask=mask)
model.predict("photo.jpg", person_mask=mask)                                 # fallback: no scene view
```

`person_mask` is required. The scene geometry comes from MoGe, which predicts the
two views independently, and the person view is aligned onto the scene view on the
pixels no person covers, so without the mask the two views are not aligned. Any
mask of the people in `photo.jpg` works; `graft.scene.person_mask` segments one
with SAM and needs the `demo` extra.

Inpainting also needs the `demo` extra, plus Hub access to the gated
[`FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) and
[`Omnieraser`](https://huggingface.co/theSure/Omnieraser) (`huggingface-cli login`).
`remove_people` sizes FLUX's 31 GiB to the free VRAM automatically — it fits a 24 GiB
card, slowly; see `offload=` for the strategies. Passing no scene view at all reuses
the photo as its own, so the people occlude the geometry being fitted: a fallback,
not a supported setting.

---

## Evaluation

```bash
export GRAFT_BENCHMARK_ROOT=/path/to/datasets    # holds PROX/
python evaluate.py
```

PROX and RICH are distributed by their own authors and not redistributed here;
**[`docs/EVALUATION.md`](docs/EVALUATION.md)** says what to download and how to lay
it out. Six metrics per benchmark — PA-MPJPE (mm), vertex-to-scene L2 (mm) and angle
(degrees), contact precision / recall / F1:

| model | benchmark | PA-MPJPE | V2S L2 | V2S angle | Prec | Rec | F1 | frames |
|---|---|---|---|---|---|---|---|---|
| `graft` | PROX | 49.73 | 184.36 | 50.92 | 0.556 | 0.638 | 0.594 | 174 |
| `graft` | RICH | 46.32 | 224.66 | 34.60 | 0.473 | 0.743 | 0.578 | 94 |
| `graft-no-visual` | PROX | 52.56 | 188.47 | 51.92 | 0.534 | 0.510 | 0.522 | 174 |
| `graft-no-visual` | RICH | 45.52 | 223.10 | 34.06 | 0.434 | 0.717 | 0.540 | 94 |

The `graft` rows are the published numbers, reproduced exactly. The geometry-only
rows differ from the paper's ablation by ~2 mm PA-MPJPE, which predates the batched
body fit used here; the interaction metrics are essentially unchanged. Scored frames
are fewer than the totals (178 and 100) where no single-person prediction is made.

---

## Performance

```bash
python benchmark.py --image photo.jpg --scene-image empty_room.jpg
```

**Inference latency is 174 ms** (5.7 images/s) on one H100, for a single image
with one person and three refinement iterations:

| stage | ms | share |
|---|---|---|
| geometry backbone (MapAnything) | 44.0 | 25% |
| pose init (NLF + batched body fit) | 43.3 | 25% |
| metric scale (MoGe) | 34.8 | 20% |
| refinement (3 iterations) | 33.1 | 19% |
| scene alignment | 9.9 | 6% |
| mesh posing (SMPL-X) | 2.6 | 1% |
| Python glue | 6.7 | 4% |
| **total** | **174.4** | 100% |

Measured from prepared input tensors, so image decoding is excluded.
`benchmark.py` exits non-zero if the median exceeds `--budget` (200 ms default).

---

## Licence

Our code is **MIT** (see [`LICENSE`](LICENSE)). The released **weights are for
non-commercial research use only** — GRAFT loads MapAnything (CC BY-NC 4.0), NLF
and SMPL-X at run time, all of which are non-commercial.

Vendored third-party code keeps its own licence: `third_party/mapanything/` is
Apache-2.0 (Meta), with our changes recorded in
[`third_party/PATCHES.md`](third_party/PATCHES.md).

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{ym2026graft,
  title        = {GRAFT: Geometric Refinement and Fitting Transformer for Human Scene Reconstruction},
  author       = {Pradyumna YM and Yuxuan Xue and Yue Chen and Nikita Kister and Istv{\'a}n S{\'a}r{\'a}ndi and Gerard Pons-Moll},
  year         = {2026},
  eprint       = {2604.19624},
  archivePrefix = {arXiv},
  url          = {https://arxiv.org/abs/2604.19624},
}
```

## Acknowledgements

GRAFT builds directly on [MapAnything](https://github.com/facebookresearch/map-anything)
(scene geometry), [NLF](https://github.com/isarandi/nlf) (initialisation),
[MoGe](https://github.com/microsoft/MoGe) (metric scale),
[smplfitter](https://github.com/isarandi/smplfitter) (body fitting), and
optionally [WiLoR](https://github.com/rolpotamias/WiLoR) (hands) and
[OmniEraser](https://github.com/PRIS-CV/Omnieraser) (scene inpainting). See
`third_party/PATCHES.md`.
