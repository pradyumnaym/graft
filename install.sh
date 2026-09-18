#!/usr/bin/env bash
# GRAFT installation. Creates nothing: run it inside the environment you want.
#
#   bash install.sh              # everything: inference, benchmarks, demo extras
#   bash install.sh --minimal    # inference only
#
# Dependencies live in pyproject.toml. requirements-lock.txt records the exact
# versions the published numbers were produced with, for when a result does not
# reproduce; it is a record, not an install target.
set -euo pipefail

MINIMAL=0
for a in "$@"; do
  case "$a" in
    --minimal) MINIMAL=1 ;;
    -h|--help) sed -n '2,5p' "$0" | sed 's/^#[ ]\?//'; exit 0 ;;
    eval|demo)
      echo "the extras are installed by default now -- just run: bash install.sh" >&2
      echo "(use --minimal for an inference-only install)" >&2
      exit 1 ;;
    *) echo "unknown option: $a (expected --minimal)" >&2; exit 1 ;;
  esac
done

# torch must come from the CUDA-matched index; PyPI's default build often is not
# the one for your driver. Override with e.g. TORCH_INDEX=.../cu121 if needed.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
echo "==> torch, from $TORCH_INDEX"
pip install torch==2.6.0 torchvision==0.21.0 --index-url "$TORCH_INDEX"

echo
echo "==> graft (inference)"
pip install -e .

if (( MINIMAL )); then
  echo
  echo "Inference only, as asked. The benchmark harness (graft.eval), and the"
  echo "demo's person segmentation, hand refinement and scene inpainting are not"
  echo "installed; add them later with a plain \`bash install.sh\`."
  echo
  echo "Now fetch the weights:   python scripts/download_assets.py"
  echo "Then check the install:  python scripts/check_install.py"
  exit 0
fi

# The extras go in one at a time, and are allowed to fail. They are the fragile
# part of this script: much heavier dependencies (open3d alone is ~450 MB), and
# two of them -- lang-sam and wilor-mini -- install from git rather than PyPI.
# Installing them separately means a failure costs you that one feature instead
# of the whole environment, which a single `pip install -e ".[all]"` would.
failed=()
for extra in eval demo; do
  echo
  echo "==> graft[$extra]"
  if ! pip install -e ".[$extra]"; then
    failed+=("$extra")
    echo "!!! the '$extra' extra failed to install" >&2
  fi
done

echo
if (( ${#failed[@]} )); then
  echo "------------------------------------------------------------------"
  echo "GRAFT itself installed fine, but these extras did not: ${failed[*]}"
  echo
  for extra in "${failed[@]}"; do
    case "$extra" in
      eval) echo "  eval  -- evaluate.py and graft.eval will not run." ;;
      demo) echo "  demo  -- demo.py will not run (it segments the people with SAM)." ;;
    esac
  done
  echo
  echo "Inference works regardless. Retry with:"
  for extra in "${failed[@]}"; do
    echo "  pip install -e \".[$extra]\""
  done
  echo "------------------------------------------------------------------"
else
  echo "installed: graft, with the eval and demo extras."
fi

echo
echo "Now fetch the weights:   python scripts/download_assets.py --eval"
echo "Then check the install:  python scripts/check_install.py --all"
