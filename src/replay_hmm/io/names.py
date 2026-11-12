"""Persisted file and directory names: one constant per file the pipeline writes and reads back.

Each value is part of the frozen on-disk layout. Saved results/ trees are found by these
exact strings, so a value must never change, and two constants never spell one file. Each
constant is named for the file it is, not for the command that writes it.
"""


# ---------------------------------------------------------------- one fit run directory
# <fits root>/<rat>/<session>/<run tag>/ (Lustig, Pfeiffer) or <fits root>/<session>/<run tag>/
# (Buzsaki), written by the three fit CLIs.
CV_SUMMARY_NAME = "cv_model_summary_full_em_mbtraj.csv"
FIT_SUMMARY_NAME = "full_fit_summary_full_em_mbtraj.json"
BIC_NAME = "full_data_bic_full_em_mbtraj.csv"
CV_FOLD_ROWS_NAME = "cv_fold_rows_full_em_mbtraj.csv"
TRANSITION_MATS_NAME = "full_data_transition_matrices_full_em_mbtraj.npz"
# The two-map fits also save the decoded inputs they were fitted on.
BUZSAKI_INPUTS_NAME = "buzsaki_hmm_mbtraj_inputs.npz"
PFEIFFER_INPUTS_NAME = "pfeiffer_hmm_mbtraj_inputs.npz"
# Resumable CV (Lustig fit CLI): one pickle per finished (kernel, fold) fit, written under the
# run directory by replay_hmm.hmm.checkpoints.
CV_CHECKPOINT_DIRNAME = "_cv_checkpoints"

# ---------------------------------------------------------------- jump distributions
JUMP_DIST_DIRNAME = "jump_distribution"
JUMP_DIST_NAME = "jump_distributions.npz"
JUMP_META_NAME = "jump_distribution_metadata.json"
TRANSITION_KERNELS_NAME = "fitted_transition_kernels.npz"

# ---------------------------------------------------------------- per-event kernel scores
# Kernel-evidence caches (`replay-hmm score-events --dataset <ds>`), under
# <results_root>/replay_classification/<tag>/; what each file holds is `io.evidence`'s.
EVIDENCE_NAME = "event_kernel_evidence.csv"
EVIDENCE_META_NAME = "event_kernel_evidence_metadata.json"
CLASSIFIED_NAME = "replay_events_classified.csv"
# The direction arm's cache (Lustig `--direction`).
DIRECTION_EVIDENCE_NAME = "event_direction_evidence.csv"
DIRECTION_EVIDENCE_META_NAME = "event_direction_evidence_metadata.json"

# ---------------------------------------------------------------- downstream analyses
TRANSFER_SUMMARY_NAME = "cross_session_within_rat_summary.csv"
VIEWER_CACHE_DIRNAME = "viewer_cache"
MISSPEC_SWEEP_ROWS_NAME = "sweep_rows.csv"
MISSPEC_SWEEP_PROVENANCE_NAME = "sweep_provenance.json"
SYNTHETIC_RECOVERY_ROWS_NAME = "synthetic_recovery_rows.csv"
SYNTHETIC_RECOVERY_GRID_ROWS_NAME = "synthetic_recovery_grid_rows.csv"
SYNTHETIC_RECOVERY_RANDOM_ROWS_NAME = "synthetic_recovery_random_rows.csv"
