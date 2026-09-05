"""Generate the vendored helper modules under `graft/ops/` and `graft/nets/`.

Same contract as the other build scripts: everything numerical is copied by AST
source range from the research tree, so the release computes what the published
numbers were produced with. Only import headers are new -- the research modules
pull in open3d / trimesh / pytorch3d / DINOv2 / `transformers` for code GRAFT's
inference path never reaches, and those dependencies are dropped here.

Run:  python tools/build_ops.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import extract  # noqa: E402
from scrub import scrub  # noqa: E402

ROOT = Path(__file__).parent.parent
SRC = ROOT.parent

MODULES = [
    ("graft/ops/geometry.py", "utils/geometry.py",
     ["project_points_torch_batch", "pointmap_to_intrinsics_batch", "compute_point_normals",
      "get_nearest_points_and_normals", "align_points_robust_batched",
      "aa_to_rotmat", "rot6d_to_rotmat", "rotmat_to_rot6d", "rotmat_to_aa"],
     '"""Geometry helpers used by the GRAFT forward pass.\n\n'
     "Vendored verbatim (by AST source range, see tools/extract.py) from the research\n"
     "tree's `utils/geometry.py`, so the released model computes exactly what the\n"
     "published numbers were produced with. Only the imports below are new: the\n"
     "research module pulls in open3d / trimesh / pytorch3d for functions GRAFT's\n"
     'inference path never touches, and those dependencies are dropped here.\n"""\n'
     "from typing import Optional, Tuple\n\n"
     "import roma\nimport torch\nimport torch.nn.functional as F\n"),

    ("graft/ops/smplx_params.py", "utils/utils.py",
     ["recursive_detach", "recursive_materialize", "convert_smplx_params_to_6d",
      "convert_smplx_params_6d_to_aa", "convert_smplx_params_6d_to_rotmat",
      "convert_smplx_params_aa_to_6d", "scale_smplx_params"],
     '"""SMPL-X parameter conversions (axis-angle <-> 6D <-> rotation matrix) and the\n'
     "tensor-tree helpers the forward pass uses.\n\n"
     "Vendored verbatim (by AST source range, see tools/extract.py) from the research\n"
     'tree\'s `utils/utils.py`.\n"""\n'
     "import pickle\n\nimport torch\n\n"
     "from .. import paths\n"
     "from .geometry import aa_to_rotmat, rot6d_to_rotmat, rotmat_to_rot6d, rotmat_to_aa\n\n"
     "# Per-beta offset about which SMPL-X shape is rescaled. `scale_smplx_params`\n"
     "# declares this `global` and moves it to the caller's device on first use.\n"
     "with open(paths.ASSETS_DIR / 'scale_c.pkl', 'rb') as _f:\n"
     "    c_scale = pickle.load(_f)\n"),

    ("graft/nets/layers.py", "network/modules.py",
     ["LearnableFourierFeatures", "MLP"],
     '"""The two small building blocks GRAFT\'s heads are made of.\n\n'
     "Vendored verbatim (by AST source range, see tools/extract.py) from the research\n"
     "tree's `network/modules.py`, which additionally imports DINOv2 and\n"
     '`transformers` for encoders GRAFT does not use.\n"""\n'
     "import itertools\nfrom typing import Sequence\n\n"
     "import torch\nimport torch.nn as nn\n"),
]

# Modules taken across whole, because they are already self-contained.
VERBATIM_FILES = [
    ("graft/ops/head_anchor.py", "utils/head_anchor.py"),
    ("graft/ops/perf.py", "utils/perf.py"),
    ("graft/ops/joint_names.py", "utils/joint_names.py"),
    ("graft/ops/wilor_batch.py", "utils/wilor_batch.py"),
    ("graft/nets/feature_sampler.py", "network/feature_sampler.py"),
    ("graft/nets/transformer.py", "network/scenefit/pose_transformer_ineff.py"),
    ("graft/nets/t_cond_mlp.py", "network/scenefit/t_cond_mlp.py"),
]

ASSETS = ["smplx_head_neck_ids.npy", "smplx_sampled_points_27.pkl",
          "smplx_sampled_points_64.pkl", "scale_c.pkl"]


def main():
    for out, src, symbols, header in MODULES:
        body = extract.extract(str(SRC / src), symbols)
        (ROOT / out).write_text(scrub(header + "\n\n" + body))
        print(f"  generated {out}  <- {src}")
    for out, src in VERBATIM_FILES:
        (ROOT / out).write_text(scrub((SRC / src).read_text()))
        print(f"  copied    {out}  <- {src}")
    for name in ASSETS:
        (ROOT / "graft" / "assets" / name).write_bytes((SRC / "assets" / name).read_bytes())
        print(f"  copied    graft/assets/{name}")

    # fast_fitter is copied with three surgical edits: the two research-tree paths
    # go through graft.paths, and the docstring points at GRAFTModel.
    ff = (SRC / "utils" / "fast_fitter.py").read_text()
    edits = [
        ("so the model classes\ngain only a one-line call where they used to invoke NLF's parametric method.",
         "so `GRAFTModel`\ngains only a one-line call where it would otherwise invoke NLF's parametric method."),
        ("from smplfitter.pt import BodyModel, BodyFitter",
         "from smplfitter.pt import BodyModel, BodyFitter\n\nfrom .. import paths"),
        # Recover the vertex subset from the detector itself rather than from an
        # external canonical-mesh file. NLF's checkpoint carries both the full SMPL-X
        # T-pose template and its fitter's 1024-vertex subset template, so matching
        # T-pose to T-pose recovers the same indices (verified identical) without a
        # second download -- and cannot drift, since both come from one checkpoint.
        ('    # Recover NLF\'s exact 1024-vertex subset: NN-match its subset canonical verts to the\n'
         '    # full SMPL-X canonical T-pose (the subset is a strict subset of the full mesh).\n'
         '    from scipy.spatial import cKDTree\n'
         '    full_cano = np.load("ckpt/nlf/canonical_verts/smplx.npy").astype(np.float32)\n'
         '    d, idx = cKDTree(full_cano).query(cano[:nV].cpu().numpy(), k=1)',
         '    # Recover NLF\'s exact 1024-vertex subset straight from the checkpoint: its body\n'
         '    # fitter carries the subset\'s T-pose template and its SMPL-X model carries the\n'
         '    # full one, and the subset is a strict subset of the full mesh. Matching T-pose\n'
         '    # to T-pose therefore recovers the indices exactly, using only tensors from this\n'
         '    # same checkpoint -- so there is nothing to download and nothing that can drift.\n'
         '    from scipy.spatial import cKDTree\n'
         '    full_T = nlf.body_models.smplx.v_template.detach().cpu().numpy().astype(np.float32)\n'
         '    sub_T = nlf.fitters.smplx.body_model.v_template.detach().cpu().numpy().astype(np.float32)\n'
         '    d, idx = cKDTree(full_T).query(sub_T, k=1)'),
        ('                   model_root="ckpt/body_models/smplx", vertex_subset=list(idx)).to(device)',
         '                   model_root=str(paths.body_model_dir()), vertex_subset=list(idx)).to(device)'),
        # Drop the per-point uncertainty from the returned dict. It still weights the
        # fit internally (that is NLF's own algorithm); only the surfaced output key,
        # which nothing in this release consumes, is removed.
        ('                        "body_pose": z(21, 3), "left_hand_pose": z(15, 3), "right_hand_pose": z(15, 3),\n'
         '                        "uncertainty": z(nV + nJ)})  # key always present (mirrors vgraft PATCHES.md #4)',
         '                        "body_pose": z(21, 3), "left_hand_pose": z(15, 3), "right_hand_pose": z(15, 3)})'),
        ('            "right_hand_pose": pr[:, 120:165].reshape(n, 15, 3),\n'
         '            "uncertainty": unc[:n],  # per-(human,point) NLF uncertainty (1024 verts + 55 joints);\n'
         '                                     # consumed by the fit weights above and, downstream, by the\n'
         '                                     # uncertainty-conditioned temporal write gate + the vgraft\n'
         '                                     # input-smoother (same surfacing as vgraft PATCHES.md #4).\n'
         '        })',
         '            "right_hand_pose": pr[:, 120:165].reshape(n, 15, 3),\n        })'),
    ]
    for old, new in edits:
        assert old in ff, f"fast_fitter.py changed upstream; missing:\n{old}"
        ff = ff.replace(old, new)
    (ROOT / "graft/ops/fast_fitter.py").write_text(scrub(ff))
    print("  generated graft/ops/fast_fitter.py  <- utils/fast_fitter.py (3 path edits)")

    # images.py: prepare_image/prepare_mask verbatim, with the device made a
    # parameter instead of a hard-coded .cuda(), plus a small input coercion helper.
    body = extract.extract(str(SRC / "utils/utils.py"), ["prepare_image", "prepare_mask"])
    body = body.replace("def prepare_image(image: Image, pad_to_square=False):",
                        "def prepare_image(image: Image.Image, pad_to_square=False, device='cuda'):")
    body = body.replace("def prepare_mask(mask: Image, pad_to_square=False):",
                        "def prepare_mask(mask: Image.Image, pad_to_square=False, device='cuda'):")
    assert body.count("[None, :, :, :].cuda()") == 1 and body.count("mask = mask.cuda()") == 1
    body = body.replace("[None, :, :, :].cuda()", "[None, :, :, :].to(device)")
    body = body.replace("mask = mask.cuda()", "mask = mask.to(device)")
    header = (ROOT / "tools/images_header.py.in").read_text()
    footer = (ROOT / "tools/images_footer.py.in").read_text()
    (ROOT / "graft/ops/images.py").write_text(scrub(header + "\n\n" + body + "\n\n" + footer))
    print("  generated graft/ops/images.py  <- utils/utils.py")


if __name__ == "__main__":
    main()
