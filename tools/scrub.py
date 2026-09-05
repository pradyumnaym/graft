"""Rewrite research-tree names and descriptions out of vendored source.

Applied by every build script to every generated file. The vendored code is
copied verbatim for numerical fidelity, which also copies the comments it was
written with: internal class names, file paths, dataset names, and descriptions
of work that is not part of this release. This pass replaces them.

`tools/guard.py` enforces that nothing slips through.
"""

PERF_DOCSTRING_OLD = '"""perf.py — fast-inference defaults for SceneFit, and a guard against the single\nnastiest performance regression this codebase has hit.\n\n────────────────────────────────────────────────────────────────────────────────\nTHE SAGA (read this before touching anything cuBLAS-determinism related):\n────────────────────────────────────────────────────────────────────────────────\nWiLoR (hand estimator) was added AFTER the paper. Importing it pulls in `ultralytics`,\nwhose `ultralytics/utils/__init__.py` unconditionally runs\n\n    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"\n\n(it wants bitwise-deterministic *training*). That env var forces cuBLAS onto its slow\ndeterministic GEMM kernels for the WHOLE process, which roughly DOUBLES every\nmatmul-heavy step — the NLF body-fitter, the SMPL-X forwards, the transformer, MA.\nNet effect: SceneFitScale.infer jumped from the paper\'s ~400 ms to ~540 ms purely\nbecause a transitive import flipped a global flag. It is silent, global, and survives\nfor the life of the process — exactly the kind of thing that costs a day of panic.\n\nWe do NOT need bitwise determinism at inference, so the fix is simply to pop the env\nvar (and turn deterministic algos off). `enable_fast_gemm()` does that; it is called\n  1. at `network/scenefitter.py` import (top level), and\n  2. inside `require_wilor()` *immediately after* the lazy wilor_mini import — i.e. right\n     after ultralytics gets a chance to plant the var, so the rest of the run stays fast.\nRe-importing wilor_mini later does NOT re-plant it (ultralytics\' __init__ runs once,\nmodule-cached), so those two pops cover the normal flow.\n\n────────────────────────────────────────────────────────────────────────────────\nTHE SAFEGUARD (why this isn\'t fragile anymore):\n────────────────────────────────────────────────────────────────────────────────\nThe two pops above are *order-dependent*. To make a future regression impossible to\nmiss, `SceneFitScale.infer` calls `enable_fast_gemm(warn_if_replanted=True)` on every\ninference: it re-pops the var (self-healing, ~µs) AND emits a one-time RuntimeWarning\nif it found the var set again — so a NEW importer (or a new dependency that plants the\nsame flag) shows up as a loud warning at the call site, not a silent ~1.4x slowdown.\nIf you ever see that warning: find the new ultralytics/deterministic importer and add an\n`enable_fast_gemm()` right after it (grep for the existing call sites).\n"""'

PERF_DOCSTRING_NEW = '''"""Fast-inference defaults, and a guard that keeps them.

`ultralytics` -- pulled in transitively by the optional WiLoR hand refiner -- sets
`CUBLAS_WORKSPACE_CONFIG=":4096:8"` at import, which puts cuBLAS on its slow
deterministic GEMM kernels for the whole process and roughly doubles the cost of
every matmul. GRAFT does not need bitwise determinism at inference, so
`enable_fast_gemm()` removes the variable and turns deterministic algorithms off.

It is called at import, again after the lazy WiLoR import, and once inside
`GRAFTModel.infer` with `warn_if_replanted=True` -- that last call warns if it
finds the flag set again, so a future dependency that plants it shows up as a
warning rather than a silent slowdown.
"""'''

PERF_WARNING_OLD = '            warnings.warn(\n                f"[scenefit.perf] {_DET_ENV} was set again at inference time — some import "\n                "re-planted the deterministic-cuBLAS flag (historically wilor_mini->ultralytics). "\n                "This ~2x\'s every GEMM (the post-paper ~400->540 ms regression). It has been "\n                "auto-popped so this run stays fast, but find the new importer and add an "\n                "enable_fast_gemm() call right after it (see call sites in network/scenefitter.py).",\n'

PERF_WARNING_NEW = '            warnings.warn(\n                f"[graft.perf] {_DET_ENV} was set again at inference time. Some import "\n                "re-planted the deterministic-cuBLAS flag, which roughly doubles the cost "\n                "of every GEMM. It has been popped so this run stays fast; to fix it "\n                "properly, call enable_fast_gemm() immediately after whichever import set it.",\n'

RULES = [
    # --- internal stages marked private: `predict` and `infer` are the public
    #     surface, and these four are only ever called from inside the class.
    ("def process_input(", "def _process_input("),
    ("self.process_input(", "self._process_input("),
    ("def update_parameters(", "def _update_parameters("),
    ("self.update_parameters(", "self._update_parameters("),
    ("def call_update_param(", "def _refine_once("),
    ("self.call_update_param(", "self._refine_once("),
    ("def get_hand_pose(", "def _get_hand_pose("),
    ("self.get_hand_pose(", "self._get_hand_pose("),
    ("`process_input`", "`_process_input`"),
    (PERF_DOCSTRING_OLD, PERF_DOCSTRING_NEW),
    (PERF_WARNING_OLD, PERF_WARNING_NEW),

    # --- imports that still reached into the research tree
    ("from utils.wilor_batch import batched_predict_with_bboxes",
     'with optional("demo", "Hand refinement (refine_hands=True)"):\n'
     "            from .ops.wilor_batch import batched_predict_with_bboxes"),

    # --- internal names -> the released ones
    ("decode_scenefit_smplx", "decode_graft_smplx"),
    ("SceneFitScale.infer", "GRAFTModel.infer"),
    ("SceneFitScale", "GRAFTModel"),
    ("a SceneFit model's", "the model's"),
    ("Decode a SceneFit `outputs[key]` dict", "Decode a `GRAFTModel.infer` output dict"),
    ("depth anchor for SceneFit.", "depth anchor."),
    ("fast-inference defaults for SceneFit, and", "fast-inference defaults, and"),
    ("[scenefit.perf]", "[graft.perf]"),
    ("Batched WiLoR hand-pose for SceneFit.", "Batched WiLoR hand-pose refinement."),

    # --- internal file paths -> the released layout
    ("`network/scenefitter.py`", "`graft/model.py`"),
    ("network/scenefitter.py", "graft/model.py"),
    ("`eval_scenefit_prox.py`", "`evaluate.py`"),
    ("eval_scenefit_prox.py", "evaluate.py"),
    ("head_anchor.py — ", ""),
    ("perf.py — ", ""),
    ("fast_fitter.py — ", ""),

    # --- comments left mid-thought in the research code
    ("        # TODO: remove this hacky code and write an elegant sampler.",
     """        # Pack the two token families into one fixed (n_tokens, 54, 512) context block.
        # Joint and surface tokens attend to their whole sampled window across both
        # views (2 * radius^2 = 18 features); the single whole-body token instead takes
        # only the window CENTRE of each of the 27 surface points, which is also 54.
        # Same width, different meaning -- hence the two separate writes below."""),
    ("        # print(\"Sampled features shape:\", context.shape)  # N_h x N_q x 18 x 512\n", ""),
    ("        # print(\"Final context shape:\", final_context.shape)  # N_h x 24 x 54 x 512\n", ""),
    ("        # print(\"Final mask shape:\", final_mask.shape)        # N_h x 24 x\n", ""),

    # --- stale or misplaced section comments
    ("        # renderer no longer needed: head-pixel depth via head_anchor (ray-triangle)\n", ""),
    ("""        ###################################################
        # NLF estimation
        ####################################################
        self.require_nlf(device=device)""",
     """        ###################################################
        # Scene geometry, and the per-frame pose initialisation
        ###################################################
        self.require_nlf(device=device)"""),

    # --- descriptions of work outside this release
    ("""# Sanitize non-finite NLF init params (~0.5% overall, ~2.8% on BEDLAM2 dense
        # crowds): replace NaN/Inf with 0 so the forward stays finite. We KEEP all
        # humans (no drop) so prediction and GT human counts stay consistent for the
        # loss -- dropping here previously desynced pred (N-1) vs gt_vertices (N) and
        # broke vertex_loss. valid_humans / idxs are left as prepare_gt set them.""",
     """# Sanitise non-finite NLF initialisations (a fraction of a percent overall,
        # more in dense crowds): replace NaN/Inf with 0 so the forward stays finite.
        # Every person is kept -- dropping one here would desynchronise the person
        # count that valid_humans / idxs index with."""),
    ("weights = uncertainty^-1.5, num_iter=3, beta_reg=10, final_adjust_rots) so downstream is\n"
     "unchanged; only ambiguous finger joints (refined downstream by WiLoR) can differ.",
     "NLF's own per-point fit weighting, num_iter=3, beta_reg=10, final_adjust_rots), so\n"
     "downstream is unchanged; only ambiguous finger joints can differ."),
    ("only rank-deficient finger joints (low confidence, refined by WiLoR) can differ.",
     "only rank-deficient finger joints (which WiLoR refines anyway) can differ."),
    ("# the conversion drops ~25 -> ~6 ms in-pipeline (~16 ms off GRAFTModel.infer).",
     "# the conversion drops ~25 -> ~6 ms in-pipeline."),
    ("Net effect: GRAFTModel.infer jumped from the paper's ~400 ms to ~540 ms purely",
     "Net effect: inference jumped from ~400 ms to ~540 ms purely"),
]


def scrub(text: str) -> str:
    for old, new in RULES:
        text = text.replace(old, new)
    return text
