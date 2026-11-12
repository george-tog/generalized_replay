#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Account settings come from the environment or from cluster/cluster.env (git-ignored; copy
# cluster/cluster.env.example). Values already in the environment win.
if [[ -f "$LOCAL_ROOT/cluster/cluster.env" ]]; then source "$LOCAL_ROOT/cluster/cluster.env"; fi
: "${REMOTE_HOST:?set REMOTE_HOST (user@login-host), e.g. in cluster/cluster.env}"
: "${REMOTE_ROOT:?set REMOTE_ROOT (the repository path on the cluster), e.g. in cluster/cluster.env}"
# Every bin width to ship. A `--bin-width-cm` run resolves per session -- 8 px on the
# TinyTim/Lucky/Ned/BB camera and 10 px on Spot/Huey/Stinky -- so shipping only one width
# leaves half the sessions without the archive the fit will ask for. Widths that do not exist
# for a session are skipped, not an error.
BWS=("8" "10")
CONTROL_PATH="${HOME}/.ssh/cm-%r@%h:%p"
SSH_OPTS=(
  -o ControlMaster=auto
  -o "ControlPath=${CONTROL_PATH}"
  -o ControlPersist=2h
)

JOBS=(
  "TinyTim rec01" "TinyTim rec02" "TinyTim rec03" "TinyTim rec04" "TinyTim rec05" "TinyTim rec06"
  "Lucky rec01"   "Lucky rec02"   "Lucky rec03"   "Lucky rec04"   "Lucky rec05"
  "Ned rec01"     "Ned rec02"     "Ned rec03"     "Ned rec04"
  # These four use the per-session directory layout and are handled by the two-layout
  # resolution below rather than the <Rat>_<session>.mat filename above.
  "BB rec01"      "Spot rec01"    "Huey rec01"    "Stinky rec01"
)

# Auxiliary tracking that `replay_hmm.datasets.lustig.maps.session.load_session` PREFERS over
# Track.xPix when present. BB is one of the sessions whose usable position data lives here;
# omitting it does not fail, it silently linearizes BB from uncleaned tracking and the
# cluster's fits diverge from everything computed locally.
AUX_FILES=("cleaned2dtracking.mat" "mutliwheelrunmarker.mat")

# Open a single master connection up front (one auth/MFA prompt), reuse for all ssh/scp calls.
ssh "${SSH_OPTS[@]}" -MNf "$REMOTE_HOST"
trap 'ssh "${SSH_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true' EXIT

for job in "${JOBS[@]}"; do
  RAT="${job%% *}"
  SESSION="${job#* }"
  echo "== $RAT / $SESSION =="

  # Two layouts. The first 15 sessions are <rat>/<rat>_<session>.mat; BB, Spot, Huey and Stinky
  # are <rat>/<session>/rec01_BehavElectrDataLFP.mat, mirroring what
  # replay_hmm.datasets.lustig.sessions._resolve_animal_session_main_file does on the Python side.
  FLAT_MAT="$LOCAL_ROOT/data/lee_lab/$RAT/${RAT}_${SESSION}.mat"
  DIR_MAT="$LOCAL_ROOT/data/lee_lab/$RAT/$SESSION/rec01_BehavElectrDataLFP.mat"
  if [[ -f "$FLAT_MAT" ]]; then
    SRC_MAT="$FLAT_MAT"
    REMOTE_SESSION_DIR="$REMOTE_ROOT/data/lee_lab/$RAT"
  elif [[ -f "$DIR_MAT" ]]; then
    SRC_MAT="$DIR_MAT"
    REMOTE_SESSION_DIR="$REMOTE_ROOT/data/lee_lab/$RAT/$SESSION"
  else
    echo "  FAILED: no mat for $RAT/$SESSION (tried $FLAT_MAT and $DIR_MAT)"
    exit 1
  fi

  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p \
    '$REMOTE_SESSION_DIR' \
    '$REMOTE_ROOT/results/$RAT/$SESSION'"

  scp "${SSH_OPTS[@]}" "$SRC_MAT" "$REMOTE_HOST:$REMOTE_SESSION_DIR/"

  # Only meaningful for the per-session layout; the flat sessions have no auxiliary files.
  for aux in "${AUX_FILES[@]}"; do
    SRC_AUX="$LOCAL_ROOT/data/lee_lab/$RAT/$SESSION/$aux"
    if [[ -f "$SRC_AUX" ]]; then
      echo "  aux: $aux"
      scp "${SSH_OPTS[@]}" "$SRC_AUX" "$REMOTE_HOST:$REMOTE_SESSION_DIR/"
    fi
  done

  # At least one map archive must exist, or the fit has nothing to decode against. A missing
  # WIDTH is fine (not every session needs both); a session with none is fatal.
  FOUND_MAP=0
  for bw in "${BWS[@]}"; do
    SRC_MAP="$LOCAL_ROOT/results/$RAT/$SESSION/lustig_maps_abs_bw${bw}.npz"
    if [[ -f "$SRC_MAP" ]]; then
      echo "  map: bw${bw}"
      scp "${SSH_OPTS[@]}" "$SRC_MAP" "$REMOTE_HOST:$REMOTE_ROOT/results/$RAT/$SESSION/"
      FOUND_MAP=1
    fi
  done
  if [[ "$FOUND_MAP" -eq 0 ]]; then
    echo "  FAILED: no map archive for $RAT/$SESSION at widths ${BWS[*]}"
    exit 1
  fi
done
