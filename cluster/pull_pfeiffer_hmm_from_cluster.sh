#!/usr/bin/env bash
set -euo pipefail

# Pull the Pfeiffer & Foster (2015) HMM fit tree back from the cluster.
#
# The Pfeiffer counterpart to pull_hmm_from_cluster.sh, shaped the way
# pull_lustig_hmm_from_cluster.sh handles the Lee-lab fits: one rsync per session behind a single
# ControlMaster connection, so all nine linear sessions cost one auth/MFA prompt, not nine.
#
# Usage:
#   ./pull_pfeiffer_hmm_from_cluster.sh                         # all nine linear sessions
#   ./pull_pfeiffer_hmm_from_cluster.sh -n                      # dry run, show what would change
#   ./pull_pfeiffer_hmm_from_cluster.sh Rat2                    # one rat, all its sessions
#   ./pull_pfeiffer_hmm_from_cluster.sh Rat1/Linear3 Rat5       # mixed rat and rat/session selectors
#   ./pull_pfeiffer_hmm_from_cluster.sh --tag '*__tradreplay'   # only matching run tag directories
#   ./pull_pfeiffer_hmm_from_cluster.sh --figures Rat1          # also pull that rat's figure tree
#   DRY_RUN=1 ./pull_pfeiffer_hmm_from_cluster.sh --tag '*__tradreplay'
#
# Quote the --tag glob: it is expanded by rsync on the remote side, not by your shell.
#
# Additive by design, deliberately no --delete: same-name files are overwritten with the
# cluster copy, new files are added, local-only files and dirs are LEFT ALONE. Two things
# here depend on that. results/PfeifferFoster2015/hmm_fits/jump_distribution/ and the
# top-level PNGs in the figures tree are both built locally, from the pulled per-session
# fits, and never go to the cluster; iterating over rat/session keeps this pull out of them.

LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Account settings come from the environment or from cluster/cluster.env (git-ignored; copy
# cluster/cluster.env.example). Values already in the environment win.
if [[ -f "$LOCAL_ROOT/cluster/cluster.env" ]]; then source "$LOCAL_ROOT/cluster/cluster.env"; fi
: "${REMOTE_HOST:?set REMOTE_HOST (user@login-host), e.g. in cluster/cluster.env}"
: "${REMOTE_ROOT:?set REMOTE_ROOT (the repository path on the cluster), e.g. in cluster/cluster.env}"
# Both mirror 1:1 between the two machines, and both are <rat>/<session>/<run_tag> below
# this point -- see DEFAULT_RESULTS_ROOT / DEFAULT_FIG_ROOT in src/replay_hmm/datasets/pfeiffer/constants.py.
RESULTS_SUBDIR="results/PfeifferFoster2015/hmm_fits"
FIGURES_SUBDIR="figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em"

# The nine linear-track sessions, mirroring PFEIFFER_SESSIONS in
# src/replay_hmm/datasets/registry.py. Rat3 and Rat4 recorded only open-field sessions, and the
# eight Open* sessions are 2-D, i.e. a different pipeline with nothing to pull here.
# Flat strings rather than an associative array: macOS ships bash 3.2, which has no declare -A.
ALL_RATS=(Rat1 Rat2 Rat5)
SESSIONS_Rat1="Linear1 Linear2 Linear3 Linear4 Linear5 Linear6"
SESSIONS_Rat2="Linear1 Linear2"
SESSIONS_Rat5="Linear1"

sessions_for() {
  local var="SESSIONS_$1"
  echo "${!var:-}"
}

TAG=""
WANT_FIGURES=0
DRY=0
[[ -n "${DRY_RUN:-}" ]] && DRY=1
SELECTORS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?--tag needs a pattern}"; shift 2 ;;
    --figures) WANT_FIGURES=1; shift ;;
    -n|--dry-run) DRY=1; shift ;;
    -h|--help) sed -n '4,25p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) SELECTORS+=("$1"); shift ;;
  esac
done

# Unknown rats and sessions raise rather than being skipped: a silently empty pull looks
# exactly like a successful one until the figures come out short.
JOBS=()
if [[ ${#SELECTORS[@]} -eq 0 ]]; then
  for rat in "${ALL_RATS[@]}"; do
    for sess in $(sessions_for "$rat"); do JOBS+=("$rat/$sess"); done
  done
else
  for sel in "${SELECTORS[@]}"; do
    rat="${sel%%/*}"
    known="$(sessions_for "$rat")"
    [[ -n "$known" ]] || {
      echo "unknown rat: $rat (known: ${ALL_RATS[*]})" >&2; exit 2; }
    if [[ "$sel" == */* ]]; then
      sess="${sel#*/}"
      [[ " $known " == *" $sess "* ]] || {
        echo "unknown session: $sel (known for $rat: $known)" >&2; exit 2; }
      JOBS+=("$rat/$sess")
    else
      for sess in $known; do JOBS+=("$rat/$sess"); done
    fi
  done
fi

# --stats, not --info=stats1: macOS ships openrsync ("rsync version 2.6.9 compatible"),
# and --info= arrived in GNU rsync 3.1. Everything else here (-a -z --partial, -m,
# --include/--exclude) is supported by both.
RSYNC_OPTS=(-az --partial --stats)
# A run's _cv_checkpoints/ exists only while its fit is unfinished, so pulling it would copy
# half a job into the local tree. Only the Lustig fit CLI writes one; it is excluded here too
# so every pull script follows the same rule. In
# RSYNC_OPTS rather than FILTER because rsync takes the first matching rule, and FILTER's
# --include='<tag>/***' would otherwise let the directory through.
RSYNC_OPTS+=(--exclude='_cv_checkpoints/')
[[ "$DRY" -eq 1 ]] && RSYNC_OPTS+=(--dry-run -v)

# One rsync per session, so paths inside the transfer are <run_tag>/<file> and the tag
# pattern is anchored at the top of it. `<tag>/***` matches the tag directory and everything
# under it; -m prunes the empty directories the exclude would otherwise leave behind.
FILTER=()
[[ -n "$TAG" ]] && FILTER=(-m --include="$TAG/***" --exclude='*')

CONTROL_PATH="${HOME}/.ssh/cm-%r@%h:%p"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${CONTROL_PATH}" -o ControlPersist=2h)

ssh "${SSH_OPTS[@]}" -MNf "$REMOTE_HOST"
trap 'ssh "${SSH_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true' EXIT

# ${FILTER[@]+"${FILTER[@]}"} rather than a bare "${FILTER[@]}": under bash 3.2 expanding an
# EMPTY array with `set -u` in force is an "unbound variable" error, so the no-tag case would
# otherwise die before the first transfer.
pull_tree() {
  local subdir="$1" job="$2"
  # Not under -n: a dry run that still creates local directories is not a dry run.
  [[ "$DRY" -eq 1 ]] || mkdir -p "$LOCAL_ROOT/$subdir/$job"
  # Trailing slash on the source: copy the CONTENTS of the remote session dir into the local
  # one, rather than nesting it as <session>/<session>.
  rsync "${RSYNC_OPTS[@]}" ${FILTER[@]+"${FILTER[@]}"} \
    -e "ssh ${SSH_OPTS[*]}" \
    "$REMOTE_HOST:$REMOTE_ROOT/$subdir/$job/" \
    "$LOCAL_ROOT/$subdir/$job/"
}

# A session that was never fit on the cluster fails its rsync. Collect those and carry on --
# aborting the loop would leave the remaining sessions unpulled over one gap -- then exit
# non-zero at the end so the gap is not mistaken for a clean run.
FAILED=()
for job in "${JOBS[@]}"; do
  echo "== $job =="
  pull_tree "$RESULTS_SUBDIR" "$job" || FAILED+=("$RESULTS_SUBDIR/$job")
  if [[ "$WANT_FIGURES" -eq 1 ]]; then
    echo "-- figures --"
    pull_tree "$FIGURES_SUBDIR" "$job" || FAILED+=("$FIGURES_SUBDIR/$job")
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo >&2
  echo "rsync failed for ${#FAILED[@]} tree(s) -- nothing on the cluster, or a transfer error:" >&2
  for f in "${FAILED[@]}"; do echo "  $f" >&2; done
  exit 1
fi
