"""Report `self.<attr>` reads that the class never assigns.

`GRAFTModel.__init__` is rewritten by hand while the methods around it are copied
verbatim, so an attribute the original set and the new one forgot would break only
when that method runs -- and if the attribute gates behaviour rather than raising,
it might not break at all. This is a static check for exactly that gap.

    python tools/check_attrs.py <file.py> [...]
"""
import ast
import sys
from pathlib import Path

import torch.nn as nn

# `training` is set by nn.Module.__init__, so it is not in dir(nn.Module).
BUILTIN = set(dir(nn.Module)) | {"training"}

# Only classes whose bases we can fully account for are checked. A class
# inheriting from something else legitimately reads attributes its parent sets,
# and reporting those is noise that would train people to ignore this check.
KNOWN_BASES = {"object", "Module", "nn.Module", "torch.nn.Module"}


def check(path: str) -> dict[str, list[str]]:
    tree = ast.parse(Path(path).read_text())
    bad = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        bases = {ast.unparse(b) for b in cls.bases}
        if bases - KNOWN_BASES:
            continue
        assigned, read, methods = set(), set(), set()
        # Class-level annotations bind attributes on a dataclass.
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                assigned.add(stmt.target.id)
            elif isinstance(stmt, ast.Assign):
                assigned.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        for n in ast.walk(cls):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.add(n.name)
            elif (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                  and n.value.id == "self"):
                (assigned if isinstance(n.ctx, ast.Store) else read).add(n.attr)
        # register_buffer("x", ...) / register_parameter binds an attribute too.
        for n in ast.walk(cls):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in {"register_buffer", "register_parameter"}
                    and n.args and isinstance(n.args[0], ast.Constant)):
                assigned.add(n.args[0].value)
        missing = sorted(read - assigned - methods - BUILTIN)
        if missing:
            bad[cls.name] = missing
    return bad


if __name__ == "__main__":
    failed = False
    for p in sys.argv[1:]:
        for cls, attrs in check(p).items():
            failed = True
            print(f"{p}: {cls} reads but never assigns -> {', '.join(attrs)}")
    if not failed:
        print("no missing attributes")
    sys.exit(1 if failed else 0)
