#!/usr/bin/env bash
# Pull HMM result trees down from the cluster.
#
# Additive by design: same-name files are overwritten with the cluster copy,
# new files are added, and local-only files/dirs are LEFT ALONE (no --delete).
#
# A run's _cv_checkpoints/ is never pulled: it exists only while that fit is unfinished
# (the Lustig fit CLI removes it once every output is written), so pulling one would copy
# half a job into the local tree.
#
# Usage:
#   ./pull_hmm_from_cluster.sh                 # pull the default trees
#   ./pull_hmm_from_cluster.sh -n              # dry run, show what would change
#   ./pull_hmm_from_cluster.sh results/hmm_fits_full_em_mbtraj   # specific trees
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Account settings come from the environment or from cluster/cluster.env (git-ignored; copy
# cluster/cluster.env.example). Values already in the environment win.
if [[ -f "$LOCAL_ROOT/cluster/cluster.env" ]]; then source "$LOCAL_ROOT/cluster/cluster.env"; fi
: "${REMOTE_HOST:?set REMOTE_HOST (user@login-host), e.g. in cluster/cluster.env}"
: "${REMOTE_ROOT:?set REMOTE_ROOT (the repository path on the cluster), e.g. in cluster/cluster.env}"
CONTROL_PATH="${HOME}/.ssh/cm-%r@%h:%p"
SSH_OPTS=(
  -o ControlMaster=auto
  -o "ControlPath=${CONTROL_PATH}"
  -o ControlPersist=2h
)

DRY=()
if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then
  DRY=(-n)
  shift
fi

# Paths are relative to the project root and mirror 1:1 between the two machines.
PATHS=("$@")
if [[ ${#PATHS[@]} -eq 0 ]]; then
  PATHS=(
    results/buzsaki_hmm_fits
    results/PfeifferFoster2015/hmm_fits
  )
fi

# One master connection up front (single auth/MFA prompt), reused by every rsync.
ssh "${SSH_OPTS[@]}" -MNf "$REMOTE_HOST"
trap 'ssh "${SSH_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true' EXIT

for p in "${PATHS[@]}"; do
  echo "== $p =="
  mkdir -p "$LOCAL_ROOT/$p"
  # ${DRY[@]+...} rather than a bare "${DRY[@]}": macOS ships bash 3.2, where expanding an
  # EMPTY array under `set -u` is an "unbound variable" error -- i.e. every non-dry run.
  rsync -a --exclude='_cv_checkpoints/' ${DRY[@]+"${DRY[@]}"} -e "ssh ${SSH_OPTS[*]}" \
    "$REMOTE_HOST:$REMOTE_ROOT/$p/" "$LOCAL_ROOT/$p/"
done
