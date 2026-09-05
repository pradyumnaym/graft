"""Report names the generated modules use but never define.

The vendored files are assembled from source ranges, so a module-level constant
that the research tree defined elsewhere can silently go missing -- and if it is
only ever read inside a function, nothing fails until that function runs. That is
exactly how a numerical constant can disappear from a release.

Uses `symtable`, which resolves scopes the way the interpreter does: every symbol
that resolves to *global* scope must be bound at module level or be a builtin.
A `global x` declaration is not a binding, so `x` still has to be assigned at
module scope somewhere.

    python tools/check_names.py <file.py> [...]
"""
import builtins
import symtable
import sys
from pathlib import Path

BUILTINS = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__", "__builtins__",
    "WindowsError", "__debug__",
}


def module_bindings(table: symtable.SymbolTable) -> set[str]:
    """Names actually assigned/imported/defined at module scope."""
    out = set()
    for sym in table.get_symbols():
        # `global x` inside a function shows up here as a symbol too; only count
        # it when this scope really binds it.
        if sym.is_assigned() or sym.is_imported() or sym.is_namespace():
            out.add(sym.get_name())
    return out


def globals_used(table: symtable.SymbolTable, acc: set[str]) -> None:
    """Every symbol, at any nesting depth, that resolves to global scope."""
    for sym in table.get_symbols():
        if sym.is_global() and sym.is_referenced():
            acc.add(sym.get_name())
    for child in table.get_children():
        globals_used(child, acc)


def check(path: str) -> list[str]:
    src = Path(path).read_text()
    top = symtable.symtable(src, path, "exec")
    bound = module_bindings(top) | BUILTINS
    used: set[str] = set()
    globals_used(top, used)
    return sorted(used - bound)


if __name__ == "__main__":
    bad = False
    for p in sys.argv[1:]:
        missing = check(p)
        if missing:
            bad = True
            print(f"{p}: undefined at module scope -> {', '.join(missing)}")
    if not bad:
        print("no undefined names")
    sys.exit(1 if bad else 0)
