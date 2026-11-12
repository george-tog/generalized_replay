#!/bin/bash
# The one SLURM job script of `replay-hmm submit` (src/replay_hmm/cluster/submit.py). It
# carries no #SBATCH header: the submitter passes each job kind's resources (SLURM_PROFILES)
# as sbatch options and exports REPLAY_HMM_ROOT, the checkout to run in. Its arguments are the
# `replay-hmm` argv, e.g.
#   sbatch <profile> slurm/run_replay_hmm.sbatch fit --dataset lustig --rat TinyTim --session rec01 ...
# Nothing here depends on the job kind: the worker takes --n-jobs from $SLURM_CPUS_PER_TASK, and
# an array task's index through `--slurm-array shard|session`.

set -euo pipefail

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "Usage: $0 REPLAY_HMM_ARGS..." >&2
  echo "Submit with \`replay-hmm submit ...\`, which passes the resources and REPLAY_HMM_ROOT." >&2
  exit 2
fi

module load python/3.12.11-fasrc01
eval "$(mamba shell hook --shell bash)"
mamba activate tcmsr

cd "$REPLAY_HMM_ROOT"
echo "[slurm] job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:--}, ${SLURM_CPUS_PER_TASK:-?} cpus: replay-hmm ${ARGS[*]}"
# The `+` form survives an empty array under `set -u` in bash 3.2.
exec replay-hmm "${ARGS[@]+"${ARGS[@]}"}"
