"""Check that every third-party module the release imports is declared.

Walks the shipped python files, collects top-level imports, drops the standard
library and the release's own packages, and maps what is left onto the dependency
sets declared in pyproject.toml. Reports anything imported but undeclared, which
is a broken install waiting to happen.

    python tools/check_requirements.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Which files belong to which install extra.
GROUPS = {
    "core": ["graft/__init__.py", "graft/model.py", "graft/paths.py", "graft/_vendor.py",
             "graft/ops", "graft/nets", "demo.py", "benchmark.py",
             "scripts/download_assets.py", "scripts/check_install.py"],
    "eval": ["graft/eval", "evaluate.py", "scripts/build_rich_sample100.py"],
    "demo": ["graft/scene.py"],
}

#: How each group is installed, for the messages.
REQ = {"core": 'pip install -e .', "eval": 'pip install -e ".[eval]"',
       "demo": 'pip install -e ".[demo]"'}

# import name -> distribution name, where they differ.
IMPORT_TO_DIST = {
    "cv2": "opencv-python", "PIL": "Pillow", "sklearn": "scikit-learn",
    "yaml": "PyYAML", "skimage": "scikit-image", "lang_sam": "lang-sam",
    "moge": "MoGe", "uniception": "UniCeption", "wilor_mini": "WiLoR-mini",
    "fast_simplification": "fast-simplification",
    "mapanything": "<vendored: third_party/mapanything>",
    "omnieraser": "<vendored: third_party/omnieraser>",
    "smplfitter": "smplfitter", "smplx": "smplx",
}

OWN = {"graft", "tools", "extract", "methods", "scrub"}

# Imported from a core file but only ever reached through an optional feature, so
# the extra that enables the feature is what has to declare it. Both of these sit
# behind `refine_hands=True` (GRAFTModel.require_wilor / graft.ops.wilor_batch).
OPTIONAL = {"wilor_mini": "demo", "skimage": "demo"}

def imports_of(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.add(n.module.split(".")[0])
    return out


def files(spec: str):
    p = ROOT / spec
    return sorted(p.rglob("*.py")) if p.is_dir() else ([p] if p.exists() else [])


def _name_of(spec: str) -> str:
    """Distribution name from a PEP 508 requirement, direct references included."""
    spec = spec.split("#")[0].strip()
    if "@" in spec:                      # e.g. "moge @ git+https://..."
        return spec.split("@", 1)[0].strip()
    for sep in ("==", ">=", "<=", "~=", ">", "<", "[", ";"):
        spec = spec.split(sep)[0]
    return spec.strip()


def declared(group: str) -> set[str]:
    """Distribution names declared for a group in pyproject.toml.

    The core dependencies are implicitly available to every group, since the
    extras are installed on top of them.
    """
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(ROOT / "pyproject.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    core = {_name_of(d) for d in cfg["project"]["dependencies"]}
    if group == "core":
        return core
    extra = cfg["project"].get("optional-dependencies", {}).get(group, [])
    return core | {_name_of(d) for d in extra}


def main() -> int:
    stdlib = set(sys.stdlib_module_names)
    bad = []
    for group, specs in GROUPS.items():
        used = set()
        for spec in specs:
            for f in files(spec):
                used |= imports_of(f)
        used -= stdlib | OWN | {"__future__"}
        have = {d.lower() for d in declared(group)}
        print(f"\n[{group}]  pyproject.toml  ({REQ[group]})")
        for mod in sorted(used):
            dist = IMPORT_TO_DIST.get(mod, mod)
            if dist.startswith("<"):
                print(f"  ok       {mod:16s} {dist}")
            elif mod in OPTIONAL:
                extra = OPTIONAL[mod]
                if dist.lower() in {d.lower() for d in declared(extra)}:
                    print(f"  ok       {mod:16s} -> {dist}  (optional, '{extra}' extra)")
                else:
                    print(f"  MISSING  {mod:16s} -> {dist}  (optional, expected in the '{extra}' extra)")
                    bad.append((extra, mod, dist))
            elif dist.lower() in have:
                print(f"  ok       {mod:16s} -> {dist}")
            else:
                print(f"  MISSING  {mod:16s} -> {dist}")
                bad.append((group, mod, dist))
    if bad:
        print(f"\n{len(bad)} import(s) not declared:")
        for g, m, d in bad:
            print(f"  pyproject.toml [{g}]: needs {d}  (imported as {m})")
        return 1
    print("\nevery imported third-party module is declared in pyproject.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
