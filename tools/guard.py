"""Fail the build if the release mentions anything it should not.

Two lists:

* FORBIDDEN -- must not appear anywhere in the released tree. These name
  unpublished work, internal infrastructure, or private repositories.
* FORBIDDEN_IN_PROSE -- allowed in code (they are third-party API names we have
  to call) but never in a comment, docstring, or documentation file, where they
  would describe rather than merely use.

Run:  python tools/guard.py            # scans the release tree
"""
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).parent.parent

FORBIDDEN = [
    "vgraft", "temporal", "video", "scenefit", "SceneFit",
    "write gate", "input-smoother", "workspace/",
    "bedlam", "mamma", "trumans", "occlusion",
    # "HSI" itself is the paper's own term ("a learned HSI prior"), so only the
    # internal dataset/workspace spellings are banned.
    "hsi68", "hsi55", "hsihard", "_hsi", "hsi_",
]

FORBIDDEN_IN_PROSE = ["uncertaint", "confidence"]

# Paths that are not ours to police: vendored code, generated/local state, and the
# build scripts themselves (which must name the research tree in order to read it).
SKIP_DIRS = {"third_party", "data", "scratch", "outputs", "__pycache__", ".git", "tools"}

# Internal files that are gitignored and therefore never published. They are
# allowed to talk about the research tree; that is their whole purpose. Keep this
# in sync with .gitignore.
SKIP_FILES = {"HANDOFF.md"}


def prose_of(path: Path) -> str:
    """Comments plus docstrings of a python file; everything else for other files."""
    if path.suffix != ".py":
        return path.read_text(errors="ignore")
    src = path.read_text(errors="ignore")
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                out.append(tok.string)
            elif tok.type == tokenize.STRING and re.match(r'^[rbuf]*("""|\'\'\')', tok.string):
                out.append(tok.string)
    except tokenize.TokenError:
        return src
    return "\n".join(out)


def walk():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in {".py", ".md", ".txt", ".json", ".sh", ".toml"}:
            continue
        rel = p.relative_to(ROOT)
        if set(rel.parts) & SKIP_DIRS or rel.name in SKIP_FILES:
            continue
        yield p


def main() -> int:
    hits = []
    for p in walk():
        rel = p.relative_to(ROOT)
        text = p.read_text(errors="ignore")
        for term in FORBIDDEN:
            for i, line in enumerate(text.splitlines(), 1):
                if term.lower() in line.lower():
                    hits.append(f"{rel}:{i}: forbidden {term!r}: {line.strip()[:100]}")
        prose = prose_of(p)
        for term in FORBIDDEN_IN_PROSE:
            for i, line in enumerate(prose.splitlines(), 1):
                if term.lower() in line.lower():
                    hits.append(f"{rel}: forbidden in prose {term!r}: {line.strip()[:100]}")
    if hits:
        print(f"{len(hits)} forbidden reference(s):")
        for h in hits:
            print("  " + h)
        return 1
    print("guard: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
