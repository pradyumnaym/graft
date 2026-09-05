"""Pull individual methods out of a class by exact AST source range, and perform
the two structural edits GRAFT's release needs (see tools/build_model.py).

Kept separate from extract.py because class members need a different walk.
"""
import ast
import textwrap


def _load(path):
    src = open(path).read()
    return src, src.splitlines(keepends=True), ast.parse(src)


def class_node(path, cls):
    src, lines, tree = _load(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return src, lines, node
    raise SystemExit(f"{path}: no class {cls!r}")


def method_source(path, cls, name):
    """Verbatim source of one method, including decorators and trailing comments
    up to (but not including) the next member."""
    _, lines, node = class_node(path, cls)
    for m in node.body:
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == name:
            start = min([d.lineno for d in m.decorator_list] + [m.lineno])
            return "".join(lines[start - 1:m.end_lineno]).rstrip("\n")
    raise SystemExit(f"{path}:{cls}: no method {name!r}")


def drop_if_keep_else(source, test_substring):
    """Delete an ``if <test>: ... else: ...`` statement, keeping only the else
    body, dedented one level.

    Used to strip the training-time (ground-truth boxes) branch from
    ``process_input``. Asserts the kept text is character-identical to the
    original else body once dedented, so the edit cannot silently reflow code.
    """
    lines = source.splitlines(keepends=True)
    tree = ast.parse(textwrap.dedent(source))
    # line numbers of the dedented parse match the original 1:1
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and test_substring in ast.unparse(node.test) and node.orelse:
            target = node
            break
    if target is None:
        raise SystemExit(f"no if-statement whose test contains {test_substring!r}")

    if_start = target.lineno - 1                      # 0-based, the `if` line
    else_first = target.orelse[0].lineno - 1
    else_last = target.orelse[-1].end_lineno          # exclusive
    # the `else:` line sits between the if-body's end and the first else stmt
    kept = "".join(lines[else_first:else_last])
    dedented = textwrap.dedent(kept) if kept.startswith(" " * 4) else kept
    # re-indent to the `if`'s own indentation
    indent = lines[if_start][: len(lines[if_start]) - len(lines[if_start].lstrip())]
    body_indent = kept[: len(kept) - len(kept.lstrip())]
    shift = len(body_indent) - len(indent)
    out = "".join(
        (ln[shift:] if ln.strip() else ln) for ln in kept.splitlines(keepends=True)
    )
    assert textwrap.indent(out, " " * shift).replace(" \n", "\n") == kept.replace(" \n", "\n"), \
        "dedent changed the else body"
    return "".join(lines[:if_start]) + out + "".join(lines[else_last:])


def drop_if(source, test_substring):
    """Delete an ``if <test>: ...`` statement that has no else branch."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and test_substring in ast.unparse(node.test):
            assert not node.orelse, "use drop_if_keep_else for if/else"
            return "".join(lines[: node.lineno - 1]) + "".join(lines[node.end_lineno:])
    raise SystemExit(f"no if-statement whose test contains {test_substring!r}")
