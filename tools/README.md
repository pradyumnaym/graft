# Build tools

`graft/` is generated from the research implementation rather than hand-ported,
so that the released forward pass is the same code that produced the published
numbers — not a re-implementation of it.

```bash
python tools/build_all.py
```

runs everything below, in order, and fails on the first problem.

| Script | What it does |
|---|---|
| `extract.py` | copies named top-level definitions out of a file by exact AST source range, pulling in whatever else in that file they reference |
| `methods.py` | the same for class members, plus the two structural edits (drop an `if/else` keeping the `else`; drop an `if`) |
| `scrub.py` | the single table of prose and identifier rewrites applied to every generated file |
| `build_ops.py` | `graft/ops/`, `graft/nets/`, `graft/assets/` |
| `build_model.py` | `graft/model.py` — `GRAFTModel` |
| `build_eval.py` | `graft/eval/` — the benchmark harness |
| `check_names.py` | scope analysis: fails if a module-level name is used but never bound |
| `check_attrs.py` | fails if a class reads a `self.<attr>` it never assigns |
| `guard.py` | fails if the release mentions internal names, paths, or unpublished work |
| `check_requirements.py` | fails if an imported third-party module is not declared in pyproject.toml |

## Why it is built this way

Extraction by source range means no numerical line is retyped, so a vendored
helper is byte-identical to its original. Everything that *does* change is an
explicit entry in a table, and almost every entry is asserted: if the upstream
text it matches ever changes, the build fails loudly instead of silently
producing a file with the edit missing.

The two checks exist because of failures this actually hit:

* **`check_names.py`** — extraction can drop a module-level constant that the
  original file defined elsewhere. If it is only read inside a function, nothing
  fails until that function runs, and if it happens to be a numerical constant
  the result is wrong rather than broken. It uses `symtable`, so a `global x`
  declaration does not count as a binding; `x` must really be assigned at module
  scope.
* **`check_attrs.py`** -- `GRAFTModel.__init__` is rewritten while the methods
  around it are copied verbatim, so an attribute the original set and the new
  one forgot would surface only when that method runs -- or, if it merely gates
  behaviour, not at all.
* **`guard.py`** — vendored code carries the comments it was written with:
  internal class names, absolute paths, dataset names, and descriptions of work
  that is not part of this release. It scans the whole tree and fails on any of
  them, so nothing leaks in through a comment.

* **`check_requirements.py`** -- vendored code keeps importing whatever the
  research tree happened to have on `sys.path`. It maps every third-party import
  in the shipped files onto pyproject.toml's dependency sets, and knows which
  modules are optional extras or vendored under `third_party/`.

One class of bug is worth calling out because it is invisible: the research code
chose its inference mode by sniffing the model's class name for `"Scale"`.
Renaming the class to `GRAFTModel` would have silently flipped every benchmark to
a different geometry source. `build_eval.py` replaces that switch with the
unconditional setting the published numbers were measured under, and asserts on
the text it is replacing.

## Changing something

Edit the build script, not `graft/`. Then:

```bash
python tools/build_all.py
python evaluate.py --limit 5      # quick sanity check
python evaluate.py                # full reproduction
```
