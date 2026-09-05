"""Source extractor used to vendor GRAFT out of the research tree.

Copies named top-level definitions BY EXACT SOURCE RANGE (ast-derived) and pulls in,
transitively, every other top-level definition in the same file that they reference.
Nothing is retyped, so a vendored helper is byte-identical to the code that produced
the published numbers.

    python tools/extract.py <src.py> <name> [<name> ...]        # emit sources
    python tools/extract.py --deps <src.py> <name> [...]        # list the closure only
"""
import ast
import sys


def _spans(tree, lines):
    """name -> (start_line, end_line) for every top-level definition."""
    spans = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            spans[node.name] = (start, node.end_lineno)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    spans[t.id] = (node.lineno, node.end_lineno)
    return spans


def closure(path, names):
    """Requested names plus every same-file top-level definition they reach, in
    source order (so a helper is always defined before its caller)."""
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    spans = _spans(tree, lines)
    by_name = {}
    for node in tree.body:
        n = getattr(node, "name", None)
        if n is None and isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    by_name[t.id] = node
        elif n is not None:
            by_name[n] = node

    wanted, stack = set(), list(names)
    while stack:
        name = stack.pop()
        if name in wanted:
            continue
        if name not in spans:
            raise SystemExit(f"{path}: no top-level definition named {name!r}")
        wanted.add(name)
        for sub in ast.walk(by_name[name]):
            if isinstance(sub, ast.Name) and sub.id in spans and sub.id not in wanted:
                stack.append(sub.id)
    return src, lines, spans, sorted(wanted, key=lambda n: spans[n][0])


def extract(path, names):
    src, lines, spans, order = closure(path, names)
    return "\n\n\n".join(
        "".join(lines[spans[n][0] - 1:spans[n][1]]).rstrip("\n") for n in order
    ) + "\n"


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--deps":
        _, _, _, order = closure(args[1], args[2:])
        print("\n".join(order))
    else:
        sys.stdout.write(extract(args[0], args[1:]))
