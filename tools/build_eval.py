"""Generate `graft/eval/` from the research tree's `evaluation/` package.

The benchmark code is copied VERBATIM and then rewritten only where it reaches
outside itself: imports are repointed at the released package, the `ckpt/`-relative
asset literals are routed through `graft.paths`, and the opaque `wrk` parameter is
renamed to `benchmark_root`. No metric, threshold, or frame list is touched -- that
is the whole point, since these numbers are what the release must reproduce.

Run:  python tools/build_eval.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import extract  # noqa: E402
from scrub import scrub  # noqa: E402

ROOT = Path(__file__).parent.parent
SRC = ROOT.parent
OUT = ROOT / "graft" / "eval"

# Modules copied straight across from evaluation/.
COPY = ["metrics", "prefetch", "determinism", "smplx_utils",
        "prox", "prox_eval", "rich", "rich_gt", "rich_eval"]

# Helpers that live outside evaluation/ in the research tree and are pulled into
# the released eval package: (output module, source file, symbols, header).
GATHER = [
    ("contacts", "human.py",
     ["get_contact_mask", "get_contact_mask_from_pointcloud_pt"],
     '"""Body-scene contact classification.\n\n'
     "Vendored verbatim (by AST source range) from the research tree's `human.py`.\n"
     "`get_contact_mask` raycasts against a scene *mesh* (PROX/RICH ground truth);\n"
     "`get_contact_mask_from_pointcloud_pt` does the same against a predicted point\n"
     'cloud. Both define the contact metric, so neither is modified.\n"""\n'
     "import numpy as np\nimport open3d as o3d\nimport torch\n\n"
     "from .knn import knn1\n"),
    ("scene_geometry", "utils/geometry.py",
     ["apply_transformation", "extend_floor_plane", "get_scene_normals", "depth_to_points"],
     '"""Scene-side geometry used only by the benchmarks.\n\n'
     "Vendored verbatim (by AST source range) from the research tree's\n"
     "`utils/geometry.py`. The floor-plane extension in particular is part of the\n"
     'contact metric definition: it is what lets a body contact the *ground*.\n"""\n'
     "import numpy as np\nimport open3d as o3d\nimport torch\n"),
    ("alignment", "utils/eval_utils.py", ["reconstruction_error", "CalibratedCamera"],
     '"""Procrustes alignment behind PA-MPJPE, and the RICH calibrated camera.\n\n'
     "Vendored verbatim (by AST source range) from the research tree's\n"
     '`utils/eval_utils.py`.\n"""\n'
     "import numpy as np\nimport torch\nimport torch.nn as nn\n"
     "from smplx.lbs import transform_mat\n"
     "from xml.etree import ElementTree as ET\n"),
    ("contact_vectors", "utils/utils.py",
     ["weighted_contact_vector_l2", "weighted_contact_vector_angle_deg"],
     '"""Weighted vertex-to-scene error, the V2S half of the reported metrics.\n\n'
     "Vendored verbatim (by AST source range) from the research tree's\n"
     '`utils/utils.py`.\n"""\n'
     "import numpy as np\n"),
]

# Import rewrites applied to every copied module.
IMPORTS = [
    (r"^(\s*)from evaluation\.(\w+) import", r"\1from .\2 import"),
    (r"^(\s*)from human import", r"\1from .contacts import"),
    (r"^(\s*)from utils\.eval_utils import", r"\1from .alignment import"),
    (r"^from utils\.geometry import (apply_transformation|extend_floor_plane|get_scene_normals|depth_to_points)",
     r"from .scene_geometry import \1"),
    (r"^from utils\.geometry import aa_to_rotmat", "from ..ops.geometry import aa_to_rotmat"),
    (r"^from utils\.utils import prepare_image, prepare_mask",
     "from ..ops.images import prepare_image, prepare_mask"),
    (r"^from utils\.utils import prepare_image", "from ..ops.images import prepare_image"),
]

# The seven `ckpt/`-relative literals, routed through the data root.
ASSETS = [
    ('depth_pkl: str = "ckpt/prox_quantitative_depth.pkl"',
     'depth_pkl: str = asset("prox_quantitative_depth.pkl")'),
    ('path: str = "ckpt/contact_vertice_weights.pkl"',
     'path: str = asset("contact_vertice_weights.pkl")'),
    ('f"ckpt/example_to_gt_contacts_prox_{interpenetration_threshold}.pkl"',
     'asset(f"example_to_gt_contacts_prox_{interpenetration_threshold}.pkl")'),
    ('model_path="ckpt/body_models/smplx/SMPLX_NEUTRAL.npz"',
     'model_path=asset("body_models", "smplx", "SMPLX_NEUTRAL.npz")'),
    ('scene_image_path: str = "ckpt/scene_image.png"',
     'scene_image_path: str = asset("scene_image.png")'),
    ('return f"ckpt/imgs/rich/scene_imgs/{scene_full}_{camera_id}.png"',
     'return asset("imgs", "rich", "scene_imgs", f"{scene_full}_{camera_id}.png")'),
    # implicit two-line string concatenation -- rewrite both lines together
    ('        f"ckpt/example_to_gt_contacts_rich_"\n'
     '        f"{interpenetration_threshold}_{contact_threshold}.pkl"',
     '        asset(f"example_to_gt_contacts_rich_"\n'
     '              f"{interpenetration_threshold}_{contact_threshold}.pkl")'),
    ('model_dir: str = "ckpt/body_models/smplx"',
     'model_dir: str = asset("body_models", "smplx")'),
]

ASSET_IMPORT = "from ._assets import asset\n"


def rewrite(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        for pat, rep in IMPORTS:
            line = re.sub(pat, rep, line)
        out.append(line)
    text = "".join(out)
    n = 0
    for old, new in ASSETS:
        if old in text:
            text = text.replace(old, new)
            n += 1
    if n:
        # `asset` is needed; put its import after the last existing relative import
        text = text.replace("from .metrics import", ASSET_IMPORT + "from .metrics import", 1) \
            if "from .metrics import" in text else text
        if ASSET_IMPORT not in text:
            lines = text.splitlines(keepends=True)
            last = max(i for i, l in enumerate(lines)
                       if l.startswith(("import ", "from ")))
            lines.insert(last + 1, ASSET_IMPORT)
            text = "".join(lines)
    # `wrk` is meaningless outside the research tree.
    text = re.sub(r"\bwrk\b", "benchmark_root", text)
    return scrub(text)


# `default_infer_kwargs` in the research tree selects the inference mode by sniffing
# the class name for "Scale" -- a switch between the scale-headed model and older
# variants that no longer exist here. Renaming the class to GRAFTModel would have
# silently flipped it to the MoGe-geometry path and changed every number, so it is
# replaced with the unconditional setting the published results were measured under.
INFER_KWARGS_OLD = '''def default_infer_kwargs(model) -> dict:
    """Pick `infer(...)` kwargs from the model's class name."""
    is_scale = "Scale" in type(model).__name__
    return {"use_moge": False, "use_moge_scale": True} if is_scale else {"use_moge": True}'''

INFER_KWARGS_NEW = '''def default_infer_kwargs(model=None) -> dict:
    """The `infer(...)` settings both published benchmarks are run under.

    MapAnything supplies the scene geometry and MoGe is used only to put it in
    metric scale. GRAFT predicts a per-person scale on top, which is what makes
    this the right pairing; the alternative (`use_moge=True`, MoGe geometry) is a
    different model family and is not what the reported numbers used.
    """
    return {"use_moge": False, "use_moge_scale": True}'''


KNN_OLD = """    # Create Pointclouds objects.
    scene_points_pc = Pointclouds(scene_points_t.unsqueeze(0))
    query_pts_pc = Pointclouds(human_vertices_t.unsqueeze(0))

    # Perform knn search using pytorch3d's knn_points.
    nn_dists, nn_idxs, _ = knn_points(
         oputil.convert_pointclouds_to_tensor(query_pts_pc)[0],
         oputil.convert_pointclouds_to_tensor(scene_points_pc)[0],
         K=1,
         return_nn=False
    )
    # Compute distances and indices.
    dists = nn_dists.sqrt().squeeze(0).squeeze(-1)  # shape: (M,)
    indices = nn_idxs.squeeze(0).squeeze(-1  # shape: (M,)
)"""

KNN_NEW = """    # K=1 nearest scene point per body vertex. Bit-identical to the pytorch3d
    # knn_points(K=1) this replaces; see graft/eval/knn.py for why it was replaced.
    nn_sq_dists, indices = knn1(human_vertices_t, scene_points_t)
    dists = nn_sq_dists.sqrt()                      # (M,)"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_assets.py").write_text(
        '"""Resolve benchmark support files inside the GRAFT data root.\n\n'
        "Note these are resolved at import time (several are default argument\n"
        "values), so `$GRAFT_DATA_ROOT` must be set before `graft.eval` is imported.\n"
        '"""\n'
        "from .. import paths\n\n\n"
        "def asset(*parts: str) -> str:\n"
        '    """Path to a benchmark support file, e.g. the PROX GT contact cache."""\n'
        "    return str(paths.data_root().joinpath(*parts))\n"
    )
    for name, src_file, symbols, header in GATHER:
        body = extract.extract(str(SRC / src_file), symbols)
        (OUT / f"{name}.py").write_text(scrub(header + "\n\n" + body))
    for name in COPY:
        (OUT / f"{name}.py").write_text(rewrite((SRC / "evaluation" / f"{name}.py").read_text()))

    # Swap the single pytorch3d call for the vendored equivalent.
    (OUT / "knn.py").write_text((ROOT / "tools" / "eval_knn.py.in").read_text())
    c = (OUT / "contacts.py").read_text()
    assert KNN_OLD in c, "the pytorch3d knn block changed upstream -- re-check the swap"
    (OUT / "contacts.py").write_text(c.replace(KNN_OLD, KNN_NEW))
    # Replace the class-name-sniffing inference-mode switch (see above). rich_eval
    # carries a duplicate of it; there it just re-exports prox_eval's single copy.
    pe = (OUT / "prox_eval.py").read_text()
    assert INFER_KWARGS_OLD in pe, "default_infer_kwargs changed upstream -- re-check the rewrite"
    (OUT / "prox_eval.py").write_text(pe.replace(INFER_KWARGS_OLD, INFER_KWARGS_NEW))
    re_ = (OUT / "rich_eval.py").read_text()
    assert INFER_KWARGS_OLD in re_, "rich_eval's default_infer_kwargs changed upstream"
    (OUT / "rich_eval.py").write_text(re_.replace(
        INFER_KWARGS_OLD, "from .prox_eval import default_infer_kwargs  # noqa: F401"))

    # metrics.py additionally reaches into utils.utils for the V2S helpers.
    m = (OUT / "metrics.py").read_text().replace(
        "from utils.utils import (\n    weighted_contact_vector_angle_deg,\n    weighted_contact_vector_l2,\n)",
        "from .contact_vectors import (\n    weighted_contact_vector_angle_deg,\n    weighted_contact_vector_l2,\n)")
    (OUT / "metrics.py").write_text(m)
    (OUT / "__init__.py").write_text(
        '"""GRAFT benchmark harness: PROX-quantitative and RICH.\n\n'
        "Generated by `tools/build_eval.py` from the research implementation, so the\n"
        "released numbers are produced by exactly the code that produced the paper's.\n"
        '"""\n'
        "from .determinism import set_deterministic\n"
        "from .prox_eval import build_prox_context, default_infer_kwargs, run_prox_eval\n"
        "from .rich_eval import build_rich_context, run_rich_eval\n\n"
        '__all__ = [\n    "set_deterministic",\n    "build_prox_context",\n'
        '    "run_prox_eval",\n    "default_infer_kwargs",\n    "build_rich_context",\n'
        '    "run_rich_eval",\n]\n')
    leftovers = []
    for f in sorted(OUT.glob("*.py")):
        t = f.read_text()
        for bad in ("from utils.", "from evaluation.", "from human import", '"ckpt/', 'f"ckpt/'):
            if bad in t:
                leftovers.append(f"{f.name}: {bad}")
    if leftovers:
        raise SystemExit("unrewritten references:\n  " + "\n  ".join(leftovers))
    print(f"wrote {len(list(OUT.glob('*.py')))} modules to {OUT}")


if __name__ == "__main__":
    main()
