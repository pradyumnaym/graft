"""Regenerate the whole vendored package from the research tree, then check it.

    python tools/build_all.py

Runs, in order: build_ops (helpers and small nets), build_model (GRAFTModel),
build_eval (the benchmark harness), then a static undefined-name pass and the
reference guard over everything generated. Every numerical line in `graft/` comes from here rather
than from hand-porting, which is what lets the release claim it reproduces the
published numbers exactly.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable

STEPS = ["build_ops.py", "build_model.py", "build_eval.py"]
CHECK = ["graft/model.py", "graft/paths.py", "graft/_vendor.py", "graft/_optional.py",
         "graft/scene.py", "evaluate.py", "demo.py", "benchmark.py",
         "scripts/download_assets.py", "scripts/check_install.py"]


def main():
    for step in STEPS:
        print(f"== {step}")
        subprocess.run([PY, str(HERE / step)], check=True)
    root = HERE.parent
    targets = [str(root / p) for p in CHECK if (root / p).exists()]
    targets += [str(p) for g in ("graft/ops", "graft/nets", "graft/eval")
                for p in sorted((root / g).glob("*.py"))]
    print("== check_names.py")
    subprocess.run([PY, str(HERE / "check_names.py"), *targets], check=True)
    print("== check_attrs.py")
    subprocess.run([PY, str(HERE / "check_attrs.py"), *targets], check=True)
    print("== guard.py")
    subprocess.run([PY, str(HERE / "guard.py")], check=True)
    print("== check_requirements.py")
    subprocess.run([PY, str(HERE / "check_requirements.py")], check=True,
                   stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
