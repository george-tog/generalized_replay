#!/usr/bin/env bash
set -euo pipefail

# Pull HMM fit results back from the cluster, one rat directory at a time. The reverse of
# move_hmm_to_cluster.sh, and it borrows that script's ControlMaster trick: one master
# connection up front means one auth/MFA prompt for all seven rats instead of seven.
#
# Usage:
#   ./pull_lustig_hmm_from_cluster.sh                            # every rat, every run tag
#   ./pull_lustig_hmm_from_cluster.sh --tag '*__centerarm'       # only matching tag directories
#   ./pull_lustig_hmm_from_cluster.sh TinyTim Lucky              # only these rats
#   ./pull_lustig_hmm_from_cluster.sh --tag '*__centerarm' Ned   # both
#   DRY_RUN=1 ./pull_lustig_hmm_from_cluster.sh --tag '*__centerarm'
#
# Quote the --tag glob: it is expanded by rsync on the remote side, not by your shell.
#
# Deliberately no --delete: the local tree holds fits, figures and caches that never went to
# the cluster, and a mirror-delete would take them out. This only ever adds and updates.

LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Account settings come from the environment or from cluster/cluster.env (git-ignored; copy
# cluster/cluster.env.example). Values already in the environment win.
if [[ -f "$LOCAL_ROOT/cluster/cluster.env" ]]; then source "$LOCAL_ROOT/cluster/cluster.env"; fi
: "${REMOTE_HOST:?set REMOTE_HOST (user@login-host), e.g. in cluster/cluster.env}"
: "${REMOTE_ROOT:?set REMOTE_ROOT (the repository path on the cluster), e.g. in cluster/cluster.env}"
SUBDIR="results/hmm_fits_full_em_mbtraj"

ALL_RATS=(TinyTim Lucky Ned BB Spot Huey Stinky)

TAG=""
RATS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?--tag needs a pattern}"; shift 2 ;;
    -h|--help) sed -n '3,20p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) RATS+=("$1"); shift ;;
  esac
done
[[ ${#RATS[@]} -eq 0 ]] && RATS=("${ALL_RATS[@]}")

# Unknown rats raise rather than being skipped: a silently empty pull looks exactly like a
# successful one until the figures come out short.
for rat in "${RATS[@]}"; do
  [[ " ${ALL_RATS[*]} " == *" $rat "* ]] || {
    echo "unknown rat: $rat (known: ${ALL_RATS[*]})" >&2; exit 2; }
done

# Paths inside the transfer are <session>/<tag>/<file>, so the tag pattern is anchored one
# level down. `--include='*/'` lets rsync descend to find them and -m prunes the empty
# session directories the exclude leaves behind.
FILTER=()
[[ -n "$TAG" ]] && FILTER=(-m --include='*/' --include="*/$TAG/***" --exclude='*')

# --stats, not --info=stats1: macOS ships openrsync ("rsync version 2.6.9 compatible"),
# and --info= arrived in GNU rsync 3.1. Everything else here (-a -z --partial
# --prune-empty-dirs, --include/--exclude) is supported by both.
RSYNC_OPTS=(-az --partial --stats)
# A run's _cv_checkpoints/ exists only while its fit is unfinished (the Lustig fit CLI removes it
# once every output is written), so pulling it would copy half a job into the local tree. In
# RSYNC_OPTS rather than FILTER because rsync takes the first matching rule, and FILTER's
# --include='*/' would otherwise let the directory through.
RSYNC_OPTS+=(--exclude='_cv_checkpoints/')
[[ -n "${DRY_RUN:-}" ]] && RSYNC_OPTS+=(--dry-run -v)

CONTROL_PATH="${HOME}/.ssh/cm-%r@%h:%p"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${CONTROL_PATH}" -o ControlPersist=2h)

ssh "${SSH_OPTS[@]}" -MNf "$REMOTE_HOST"
trap 'ssh "${SSH_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true' EXIT

for rat in "${RATS[@]}"; do
  echo "== $rat =="
  mkdir -p "$LOCAL_ROOT/$SUBDIR/$rat"
  # Trailing slash on the source: copy the CONTENTS of the remote rat dir into the local one,
  # rather than nesting it as <rat>/<rat>.
  # ${FILTER[@]+...} rather than a bare "${FILTER[@]}": macOS ships bash 3.2, where expanding
  # an EMPTY array under `set -u` is an "unbound variable" error -- i.e. every run without --tag.
  rsync "${RSYNC_OPTS[@]}" ${FILTER[@]+"${FILTER[@]}"} \
    -e "ssh ${SSH_OPTS[*]}" \
    "$REMOTE_HOST:$REMOTE_ROOT/$SUBDIR/$rat/" \
    "$LOCAL_ROOT/$SUBDIR/$rat/"
done
