#!/usr/bin/env bash
# Configure dynamic library paths for gpu4pyscf/cupy in this repo's .venv.
#
# Usage:
#   source setup_cuda.sh
#   uv run python -c "import gpu4pyscf"

# This file is sourced into interactive shells; do not change caller shell options
# (e.g. `set -u`) because that can break tab-completion and normal shell usage.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Please source this script, do not execute it:" >&2
  echo "  source scripts/setup_cuda.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/.git" || -d "$SCRIPT_DIR/.venv" ]]; then
  REPO_ROOT="$SCRIPT_DIR"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
SITE_PACKAGES_BASE="$REPO_ROOT/.venv/lib"

if [[ ! -d "$SITE_PACKAGES_BASE" ]]; then
  echo "Could not find .venv under $REPO_ROOT. Run 'uv sync' first." >&2
  return 1
fi

# Resolve site-packages/nvidia path for whichever Python minor version is used.
NVPKG=""
for d in "$SITE_PACKAGES_BASE"/python*/site-packages/nvidia; do
  if [[ -d "$d" ]]; then
    NVPKG="$d"
    break
  fi
done

if [[ -z "$NVPKG" ]]; then
  echo "Could not find nvidia libs in .venv site-packages." >&2
  echo "Install GPU deps first: uv sync --group gpu4pyscf" >&2
  return 1
fi

_append_path_unique() {
  local add_path="$1"
  local current="${2:-}"
  case ":${current}:" in
    *":${add_path}:"*) printf "%s" "$current" ;;
    *)
      if [[ -n "$current" ]]; then
        printf "%s:%s" "$add_path" "$current"
      else
        printf "%s" "$add_path"
      fi
      ;;
  esac
}

# Prefer libs required by observed failures first, then include any other nvidia/*/lib dirs.
lib_dirs=()
for subdir in cusolver cublas cusparse cuda_runtime cuda_nvrtc nvjitlink; do
  if [[ -d "$NVPKG/$subdir/lib" ]]; then
    lib_dirs+=("$NVPKG/$subdir/lib")
  fi
done
for d in "$NVPKG"/*/lib; do
  if [[ -d "$d" ]]; then
    lib_dirs+=("$d")
  fi
done

for d in "${lib_dirs[@]}"; do
  LD_LIBRARY_PATH="$(_append_path_unique "$d" "${LD_LIBRARY_PATH:-}")"
  LIBRARY_PATH="$(_append_path_unique "$d" "${LIBRARY_PATH:-}")"
done
export LD_LIBRARY_PATH
export LIBRARY_PATH

# find_library('cusolver') can be unreliable; preload helps deterministic resolution.
if [[ -f "$NVPKG/cusolver/lib/libcusolver.so.11" ]]; then
  LD_PRELOAD="$(_append_path_unique "$NVPKG/cusolver/lib/libcusolver.so.11" "${LD_PRELOAD:-}")"
  export LD_PRELOAD
fi

echo "Configured CUDA library paths from:"
echo "  $NVPKG"
