# Vendored third-party code

## MapAnything

`third_party/mapanything/` is a copy of Meta's
[MapAnything](https://github.com/facebookresearch/map-anything), which GRAFT uses as its
scene-geometry backbone. It is vendored rather than installed because GRAFT needs
a modification the upstream package does not have.

**Why it cannot be the upstream package.** GRAFT does not consume MapAnything's
predictions alone; it samples the *encoder features* at projected body points. The
`early_exit=True` path added to `MapAnything.forward` / `MapAnything.infer` returns
those intermediate features without running the prediction heads. Without it,
GRAFT would have to run the full head stack on every forward and then throw most
of it away.

**The exact delta** is recorded in `third_party/local-changes.patch`, a git diff
against the state at which MapAnything was vendored into this project. In summary,
across the files shipped here:

| File | Change |
|---|---|
| `mapanything/models/mapanything/model.py` | `early_exit` in `forward`/`infer`; `use_register_tokens_from_encoder`; `info_sharing_mlp_layer_str`; encoder returns per-view register tokens |
| `mapanything/utils/inference.py` | preprocessing reworked for the two-view (scene, people) input GRAFT feeds |
| `mapanything/utils/image.py`, `mapanything/utils/geometry.py` | small helpers |
| `mapanything/models/__init__.py`, `mapanything/models/external/da3/` | additional model registration |

**What was removed.** The training-side subtrees (`datasets/`, `train/`, and
MapAnything's own `third_party/`) are not shipped: GRAFT's release is
inference-only and nothing under `models/` or `utils/` imports them.

Model weights are *not* vendored. They are fetched from the Hugging Face Hub at
`facebook/map-anything` on first use, exactly as upstream does.

MapAnything is licensed by Meta under the Apache License 2.0; the original
`LICENSE` and copyright headers are preserved in the vendored tree.

## Other components

These are ordinary dependencies, installed from their own sources by
`install.sh`, and are **not** vendored:

| Component | Role in GRAFT | Source |
|---|---|---|
| NLF | per-frame SMPL-X initialisation | <https://github.com/isarandi/nlf> |
| MoGe | metric scale for the scene point map | <https://github.com/microsoft/MoGe> |
| smplfitter | the batched SMPL-X body fit | <https://github.com/isarandi/smplfitter> |
| SMPL-X | body model | <https://smpl-x.is.tue.mpg.de> |
| WiLoR (optional) | hand refinement | <https://github.com/rolpotamias/WiLoR> |
| UniCeption | MapAnything's transformer blocks | <https://github.com/castacks/UniCeption> |
