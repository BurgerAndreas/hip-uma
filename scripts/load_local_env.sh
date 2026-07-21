#!/usr/bin/env bash
# Source the repo-local, gitignored environment file into the current shell.

if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

ENV_FILE="${HIP_UMA_ENV_FILE:-$REPO_ROOT/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing local environment file: $ENV_FILE" >&2
  echo "Create it with: cp .env.example .env" >&2
  return 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

export FAIRCHEM_CACHE_DIR="${FAIRCHEM_CACHE_DIR:-$REPO_ROOT/.cache/fairchem}"
export KAGGLEHUB_CACHE="${KAGGLEHUB_CACHE:-$REPO_ROOT/data/kagglehub}"
