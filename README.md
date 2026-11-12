# generalized_rollouts

Code to reproduce figures and tables. Each table row gives the figure, the command that makes it, and its output. Paths in the *Figure path* column are relative to
the repository root.

## Setup

**Environment.** Every command runs in the `tcmsr` conda environment (Python 3.12):

```bash
conda env create -f environment.yml
conda activate tcmsr
```

on cluster:
```bash
module load python/3.12.11-fasrc01
eval "$(mamba shell hook --shell bash)"
mamba activate tcmsr
```

**ssm (required, modified fork).** The paper's HMM transition kernels live in a modified copy of
[lindermanlab/ssm](https://github.com/lindermanlab/ssm) based on commit `eb6c8aa` (changes to
`ssm/transitions.py`, `hmm.py`, `messages.py`, `observations.py`; see
[`third_party/ssm/UPSTREAM.md`](third_party/ssm/UPSTREAM.md)). It is vendored in this repository at
`third_party/ssm/`. *Plain upstream ssm will not work*. `environment.yml` installs the modified version
(`pip install -e ./third_party/ssm`); to add it to an existing environment, run that `pip` command
from the repository root, then `pip install -e .` to install the repository itself as the
`replay_hmm` package (`environment.yml` does both). Building ssm compiles a Cython/C++ extension,
so a system C++ compiler is required (macOS: `xcode-select --install`; Linux: `g++`). Do not edit
the `.py` files under `third_party/ssm/ssm/`: they are hashed into the CV checkpoint keys.

**Data.** Raw data go under `data/`. Only these directories are read:

| Dataset | Directory | Source |
|---|---|---|
| Lustig (Lee lab, M-maze) | `data/lee_lab/{TinyTim,Lucky,Ned}/<Rat>_recNN.mat`, `data/lee_lab/{BB,Spot,Huey,Stinky}/recNN/recNN_BehavElectrDataLFP.mat` | Lee lab (on request) |
| Grosmark & Buzsáki 2016 | `data/buzsaki_lab/{Achilles_10252013,Buddy_06272013,Cicero_09012014,Cicero_09172014,Gatsby_08022013}/…` (see `replay_hmm.datasets.registry.BUZSAKI_SESSIONS`) | [CRCNS hc-11](https://crcns.org/data-sets/hc/hc-11/about-hc-11) |
| Pfeiffer & Foster 2015 | `data/PfeifferFoster2015/{Rat1,Rat2,Rat5}/Linear*/` | on request |

Note that we internally refer to the dataset from Grosmark & Buzsáki 2016 as "buzsaki" rather than "grosmark".

Outputs go to `results/` and `figures/` at the repository root (both git-ignored), and Tables S1-S3
go to `report/tables/`.
[doc/code_map.md](doc/code_map.md) maps the package (`src/replay_hmm/`) module by module to the
commands below.

## Package commands

Every row is a `replay-hmm` command (the package CLI, installed by `pip install -e .`). Run it
from the repository root: paths the rows pass (`results/...`, `figures/...`, shell globs) are
relative to it, and the default data, results and figures roots are the checkout's own `data/`,
`results/` and `figures/` from any directory. The Fig 1/2A notebook is
`notebooks/fig1_fig2_kernels.ipynb`, opened from the repository root as well.

`notebooks/kernel_comparison_demo.ipynb` is a worked example of the kernel comparison, not a
manuscript figure. It simulates replay events from known kernels on a toy track and shows the
comparison recovering them, explains what your own data must look like and how to read the
results, and, when `data/PfeifferFoster2015` and the saved Pfeiffer fits are present, refits one
real session and checks it against the saved fit. It runs in about a minute and a half.

## Cluster rows 

HMM fits and recovery are long but parallelizable. We used the Harvard FASRC cluster to speed up computation, although analyses can be reproduced locally as well. Rows marked **[cluster]** submit SLURM jobs on Harvard FASRC through the `submit` subcommand of
`replay-hmm`, run from the repository root of the cluster checkout. Every job runs
`slurm/run_replay_hmm.sbatch`, which runs `module load python/3.12.11-fasrc01` and
`mamba activate tcmsr`, then the job's worker subcommand from the repository root, and logs to
`slurm_out/`. The submitter passes each job's resources (cores, walltime, partition, memory,
array) to `sbatch` itself, and `--dry-run` prints the `sbatch` calls without submitting. Workers
take `--n-jobs` from the job's cores. The cluster checkout needs the package installed in tcmsr
(`pip install -e .`), as the local one does. To get an email when a recovery or shard-check job
fails, set `REPLAY_HMM_SLURM_MAIL=<your address>` in the cluster shell before submitting.

| Submitter | Worker subcommand | Used for |
|---|---|---|
| `replay-hmm submit fit --dataset lustig` | `fit --dataset lustig`, one job per session | all Lustig HMM fits |
| `replay-hmm submit recovery` | `recovery run`, one array per session, one shard per task (120 or 240 shards for Fig 3E, 40 for S6) | Fig 3E, 3F, S5; with `--dataset buzsaki` / `pfeiffer`, Fig S6A / S6B |
| `replay-hmm submit misspec-sweep` | `misspec sweep`, a 14-task array (one session per task) | Table 1 (optional) |
| `replay-hmm submit fit --dataset buzsaki`, `replay-hmm submit fit --dataset pfeiffer` | `fit --dataset buzsaki` / `fit --dataset pfeiffer`, one job per session | cluster equivalents of the Fig S2 fit rows |
| `replay-hmm submit check-shards` | `recovery check-shards`, one job | the shard check below, on the cluster |

To resubmit array tasks that timed out or were preempted, repeat the submission with the same
flags plus `--array` and the task indices, e.g. `--array 5` for the sixth misspec-sweep session or
`--array 7,12` for two recovery shards; both resume where the task stopped.

To fit one Lustig session locally instead, run `replay-hmm fit --dataset lustig --rat TinyTim
--session rec01 <same flags, minus --pairwise>` from the repository root (`--pairwise` belongs to
the submitter, which pairs `--rats` with `--sessions`). The submitters take the plural
`--rats`/`--sessions` even for one session (`--rats TinyTim --sessions rec04`); only the local
fit takes the singular `--rat`/`--session`.
The 3 ms all-PBE fits (S4C) exceed the Lustig fit job's 12 h time limit. Resubmit them: CV resumes per (kernel, fold) from
`_cv_checkpoints/`.

**Send the Lustig data** before the first Lustig [cluster] fit: `bash cluster/move_hmm_to_cluster.sh` copies the 19
sessions' `.mat` files (with BB's cleaned tracking) and their 8 px and 10 px map archives (built under Prerequisites)
into the cluster checkout's `data/`. The Buzsaki and Pfeiffer fits read the same `data/` layout on the cluster; copy
those dataset directories there yourself.

**Bring results back** after each block of cluster fits. These scripts never copy `_cv_checkpoints/`.
Copy `cluster/cluster.env.example` to `cluster/cluster.env` (git-ignored) and set `REMOTE_HOST` and
`REMOTE_ROOT` for your account there, or export them.

```bash
bash cluster/pull_lustig_hmm_from_cluster.sh --tag '<run_tag>'   # Lustig fits, one run tag
bash cluster/pull_hmm_from_cluster.sh                            # all Buzsaki and Pfeiffer fits
bash cluster/pull_pfeiffer_hmm_from_cluster.sh Rat1 --tag '<run_tag>'  # selected Pfeiffer sessions or run tags
```

Check the 3E recovery shards from the repository root with `replay-hmm recovery check-shards
--run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67`, which checks every shard count on disk for that tag
(120 for TinyTim rec04, 240 for Ned rec02 and Lucky rec05).
Check the S6A / S6B recovery shards with `replay-hmm recovery check-shards --dataset buzsaki` and
`--dataset pfeiffer`; `pull_hmm_from_cluster.sh` brings their `synthetic_recovery/` folders back with the fits.

## Tests

Run from the repository root: `pytest`. Tests whose dataset is missing under `data/` skip.
`RUN_LUSTIG_DATA_TESTS=1 pytest` also runs the slow real-data Lustig checks (they need `data/lee_lab`
and the Lucky rec01 map). `RUN_NOTEBOOK_TESTS=1 pytest tests/test_demo_notebook.py` executes the
kernel-comparison demo notebook and checks what it reports.

## Reproduce everything (dependency order)

1. Prerequisites: map archives, then the three traditional decodes.
2. Lustig fits [cluster]: Fig 3 main, Fig 5 hard-zero, S4A tradreplay, S4C, S4D (TinyTim rec04, Ned rec02 and Lucky rec05 of the Fig 3 main fits are also the 3E source fits). Then pull them back.
3. Buzsaki and Pfeiffer fits: S2A, S2C, S2E; S2B, S2D, S2F.
4. 3E recovery [cluster], which needs those three Fig 3 main fits; S6A and S6B recovery [cluster], which need the S2A and S2B fits. Pull them back.
5. Lustig caches: results for 3C, then 3D, then S4B.
6. Misspecification (needs the Fig 3 **and** S2B fits): results for 2B/2C, then Table 1.
7. Jump distributions: 4C-F/3A/S9A, S9B, S7C/E, S9C/S7D/S7F.
8. Memoryless-Bayes and traditional jumps: 4A, 4B, S7A, S7B.
9. PPC: S8A-C.
10. Plots: 3B/3C, S3, 3D, 3E/S5, S4A-D, 2B/2C `--plot-only`, S2A-F, S6A/S6B.
11. Tables: the six data-budget rows, then `replay-hmm tables cv-budget`.
12. Fig 5 panels; optional viewers.

The Fig 1/2A notebook has no dependencies.

## Prerequisites
| Figure | Call | Figure path |
|---|---|---|
| Lustig map archives, bw8 (TinyTim, Lucky, Ned, BB) | `replay-hmm maps build --dataset lustig --rats TinyTim TinyTim TinyTim TinyTim TinyTim TinyTim Lucky Lucky Lucky Lucky Lucky Ned Ned Ned Ned BB --sessions rec01 rec02 rec03 rec04 rec05 rec06 rec01 rec02 rec03 rec04 rec05 rec01 rec02 rec03 rec04 rec01 --pairwise --linear-bin-mode absolute --linear-bin-width 8 --save-template 'results/{rat}/{session}/lustig_maps_abs_bw8.npz'` | `results/<rat>/<session>/lustig_maps_abs_bw8.npz` (template relative to the repo root) |
| Lustig map archives, bw10 (Spot, Huey, Stinky: 3.5 cm on their camera calibration) | `replay-hmm maps build --dataset lustig --rats Spot Huey Stinky --sessions rec01 --linear-bin-mode absolute --linear-bin-width 10 --save-template 'results/{rat}/{session}/lustig_maps_abs_bw10.npz'` | `results/{Spot,Huey,Stinky}/rec01/lustig_maps_abs_bw10.npz` |
| Lustig traditional decode (4A, S4A) | `replay-hmm traditional decode --dataset lustig --bin-width-cm 3.5 --n-jobs 6` | `results/lustig_traditional_decode/bwcm3p5__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8__2dbins_100__physarm/traditional_decode_events.csv` |
| Buzsaki traditional decode (S2C) | `replay-hmm traditional decode --dataset buzsaki --bin-width-m 0.04 --n-jobs 5` | `results/buzsaki_traditional_decode/bin_0p04__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8/traditional_decode_events.csv` |
| Pfeiffer traditional decode (S2D) | `replay-hmm traditional decode --dataset pfeiffer --event-source spike-density --bin-width-cm 4 --n-jobs 4` | `results/PfeifferFoster2015/traditional_decode/bin_4__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8__src_spikedens/traditional_decode_events.csv` |

## Fig 1, 2A
| Figure | Call | Figure path |
|---|---|---|
| 1, 2A | `jupyter lab notebooks/fig1_fig2_kernels.ipynb`: run all cells. It needs no data. Its `savefig` lines are commented out on purpose; panels were exported interactively. | exported by hand |

## Fig 2B, C, Table 1
Needs the Fig 3 HMM fits and the S2B Pfeiffer fits (the momentum generator is the median of the Pfeiffer & Foster momentum fits).

| Figure | Call | Figure path |
|---|---|---|
| results for 2B (TinyTim rec04; also covered by the 2C row) | `replay-hmm misspec gallery` | `results/hmm_misspec_gallery/TinyTim/rec04/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/` |
| results for 2C | `replay-hmm misspec gallery --all-sessions` | `results/hmm_misspec_gallery/<rat>/<session>/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/` |
| 2B | `replay-hmm misspec gallery --plot-only --event-ms 150 --example-rank 4` | [misspec_gallery__150ms__ex4.png](figures/hmm_misspec_gallery/TinyTim_rec04__dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__misspec_gallery__150ms__ex4.png) |
| 2C | `replay-hmm misspec gallery --all-sessions --plot-only` | [misspec_ll_vs_error.png](figures/hmm_misspec_gallery/all_sessions__dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__misspec_ll_vs_error.png) |
| Table 1 (local; takes days) | `replay-hmm misspec sweep --n-jobs 8` | [rmse_ratio_table.tex](figures/hmm_misspec_sweep/sweep__dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__rmse_ratio_table.tex) |
| Table 1 [cluster] | `replay-hmm submit misspec-sweep`, then `replay-hmm misspec sweep --plot-only` | same file |

## Fig 3A-D
| Figure | Call | Figure path |
|---|---|---|
| HMM fits for 2B/C, Table 1, 3A-D, 4C-F, S1, S3, S4B, S8A, S9A [cluster] | `replay-hmm submit fit --dataset lustig --rats TinyTim TinyTim TinyTim TinyTim TinyTim TinyTim Lucky Lucky Lucky Lucky Lucky Ned Ned Ned Ned BB Spot Huey Stinky --sessions rec01 rec02 rec03 rec04 rec05 rec06 rec01 rec02 rec03 rec04 rec05 rec01 rec02 rec03 rec04 rec01 rec01 rec01 rec01 --pairwise --bin-width-cm 3.5 --pbe-filter-mode nonrun_nonwheel --speed-smooth-cutoff-hz none --pbe-spike-density-stdev 3.0 --pbe-baseline-scope session --pbe-min-duration-s 0.100 --pbe-merge-gap-s 0.005 --trajectory-min-duration-s 0.050 --trajectory-merge-gap-s 0.005 --trajectory-mass-threshold 0.60` | `results/hmm_fits_full_em_mbtraj/<rat>/<session>/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/` |
| 3A | `replay-hmm jumps --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --n-jobs 4` (same call as Fig 4C-F; run once) | [Huey__rec01_transition_kernel_R_out_row](figures/hmm_jump_distribution/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/fitted_transition_kernels/Huey__rec01_transition_kernel_R_out_row.png) |
| results for 3C | `replay-hmm score-events --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67` | `results/hmm_fits_full_em_mbtraj/replay_classification/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/event_kernel_evidence.csv` |
| 3B, 3C | `replay-hmm plot cv-strip --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --min-events 10` | 3B: [cv_relative_LL_strip](figures/hmm_mbtraj_cv_relative_LL_strip_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67_minev10.png)<br>3C: [best_kernel_share_4way_momentum](figures/hmm_mbtraj_best_kernel_share_4way_momentum_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67_minev10.png) |
| S3 (fitted parameters; all sessions, no event floor) | `replay-hmm plot cv-strip --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67` | [cv_fold_transition_params](figures/hmm_mbtraj_cv_fold_transition_params_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67.png) |
| results for 3D | `replay-hmm score-events --dataset lustig --direction --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67` | `results/hmm_fits_full_em_mbtraj/replay_classification/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/event_direction_evidence.csv` |
| 3D (needs both caches) | `replay-hmm plot direction-share --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --min-events 10 --save-data results/direction_kernel_share_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67_minev10.csv` | [direction_kernel_share_coherent](figures/hmm_mbtraj_direction_kernel_share_coherent_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67_minev10.png) |

## Fig 3E, 3F, S5
The recovery analysis runs on three sessions' fits from the Fig 3A-D fit row (run tag `dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67`): TinyTim rec04 (120 shards), Ned rec02 and Lucky rec05 (240 shards each, same settings otherwise). The plot pools all three into one figure.

| Figure | Call | Figure path |
|---|---|---|
| results for 3E [cluster] | `replay-hmm submit recovery --rats TinyTim --sessions rec04 --generator-random-sample --random-sample-size 1000 --generator-num-shards 120 --num-em-iters 50 --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --kernels sticky_SR sticky_SR_rev sticky_TCMSR sticky_TCMSR_sym continuous_EM movement_empirical momentum segment_uniform stationary fragmented --grid-generator-kernels sticky_SR sticky_SR_rev sticky_TCMSR sticky_TCMSR_sym continuous_EM momentum segment_uniform --random-fixed-generator-reps 5 --random-fixed-generator-kernels movement_empirical stationary fragmented --random-sample-seed 42` | `results/hmm_fits_full_em_mbtraj/TinyTim/rec04/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/synthetic_recovery/synthetic_recovery_random_rows_shard*-of0120.csv` |
| results for 3E, Ned rec02 [cluster] | `replay-hmm submit recovery --rats Ned --sessions rec02 --generator-random-sample --random-sample-size 1000 --generator-num-shards 240 --num-em-iters 50 --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --kernels sticky_SR sticky_SR_rev sticky_TCMSR sticky_TCMSR_sym continuous_EM movement_empirical momentum segment_uniform stationary fragmented --grid-generator-kernels sticky_SR sticky_SR_rev sticky_TCMSR sticky_TCMSR_sym continuous_EM momentum segment_uniform --random-fixed-generator-reps 5 --random-fixed-generator-kernels movement_empirical stationary fragmented --random-sample-seed 42` | `results/hmm_fits_full_em_mbtraj/Ned/rec02/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/synthetic_recovery/synthetic_recovery_random_rows_shard*-of0240.csv` |
| results for 3E, Lucky rec05 [cluster] | `replay-hmm submit recovery --rats Lucky --sessions rec05 --generator-random-sample --random-sample-size 1000 --generator-num-shards 240 --num-em-iters 50 --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --kernels sticky_SR sticky_SR_rev sticky_TCMSR sticky_TCMSR_sym continuous_EM movement_empirical momentum segment_uniform stationary fragmented --grid-generator-kernels sticky_SR sticky_SR_rev sticky_TCMSR sticky_TCMSR_sym continuous_EM momentum segment_uniform --random-fixed-generator-reps 5 --random-fixed-generator-kernels movement_empirical stationary fragmented --random-sample-seed 42` | `results/hmm_fits_full_em_mbtraj/Lucky/rec05/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/synthetic_recovery/synthetic_recovery_random_rows_shard*-of0240.csv` |
| 3E, 3F, S5A-C | `replay-hmm recovery plot results/hmm_fits_full_em_mbtraj/TinyTim/rec04/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/synthetic_recovery/*of0120.csv results/hmm_fits_full_em_mbtraj/Ned/rec02/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/synthetic_recovery/*of0240.csv results/hmm_fits_full_em_mbtraj/Lucky/rec05/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/synthetic_recovery/*of0240.csv --figures-root figures/recovery_pooled --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --generator-random-sample --kernels sticky_TCMSR sticky_TCMSR_sym sticky_SR sticky_SR_rev continuous_EM momentum movement_empirical fragmented stationary --grid-generator-kernels sticky_TCMSR sticky_TCMSR_sym sticky_SR sticky_SR_rev continuous_EM momentum --random-fixed-generator-kernels movement_empirical stationary fragmented` | 3E, 3F: [winner_confusion_random](figures/recovery_pooled/synthetic_recovery/synthetic_recovery_winner_confusion_random_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67.png)<br>S5A: [sticky_TCMSR_pairwise_recoverability](figures/recovery_pooled/synthetic_recovery/synthetic_recovery_random_sticky_TCMSR_pairwise_recoverability_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67.png)<br>S5B: [continuous_EM_pairwise_recoverability](figures/recovery_pooled/synthetic_recovery/synthetic_recovery_random_continuous_EM_pairwise_recoverability_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67.png)<br>S5C: [momentum_pairwise_recoverability](figures/recovery_pooled/synthetic_recovery/synthetic_recovery_random_momentum_pairwise_recoverability_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67.png) |

## Fig 4
| Figure | Call | Figure path |
|---|---|---|
| results for 4A | see Prerequisites (the Lustig traditional decode, `traditional decode --dataset lustig`) | `results/lustig_traditional_decode/bwcm3p5__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8__2dbins_100__physarm/traditional_decode_events.csv` |
| 4A | `replay-hmm traditional jumps --dataset lustig --events-csv results/lustig_traditional_decode/bwcm3p5__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8__2dbins_100__physarm/traditional_decode_events.csv --n-jobs 8` | [traditional_jump_hist](figures/lustig/lustig_traditional_jump_distribution/bwcm3p5__v3_tbin20ms_tstep5ms_hoverthresh0bins_bwcm3p5/traditional_jump_hist.png) |
| 4B (`--continue-on-error` skips Huey rec02, which has no 3.5 cm map) | `replay-hmm mb-jump --dataset lustig --rats BB Huey Lucky Ned Spot Stinky TinyTim --bin-width-cm 3.5 --plot-pooled-jumps --continue-on-error --n-jobs 12` | [MB_jump_dist_pooled_…_v3_tbin10ms_tstep10ms_hoverthresh0bins_bwcm3p5](figures/mb_jump_pooled/MB_jump_dist_pooled_BB-Huey-Lucky-Ned-Spot-Stinky-TinyTim_v3_tbin10ms_tstep10ms_hoverthresh0bins_bwcm3p5.png) |
| 4C-F, S9A, 3A (needs the Fig 3 fits; each worker holds one session in memory, up to about 5 GB measured, so lower `--n-jobs` on small machines) | `replay-hmm jumps --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --n-jobs 4` | 4C: [sticky_TCMSR_generated](figures/hmm_jump_distribution/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/sticky_TCMSR_generated_jump_distribution_all_sessions.png)<br>4D: [momentum_generated](figures/hmm_jump_distribution/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/momentum_generated_jump_distribution_all_sessions.png)<br>4E: [sticky_TCMSR_traj_filtered](figures/hmm_jump_distribution/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/sticky_TCMSR_traj_filtered_jump_distribution_all_sessions.png)<br>4F: [momentum_traj_filtered](figures/hmm_jump_distribution/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/momentum_traj_filtered_jump_distribution_all_sessions.png) |

## Fig 5
The event viewer is interactive (arrow keys or n/p step through events; `s` saves). The rows below save one panel each without opening a window. `--event-index` is the position in `--sort-by dll` order, so `event0056` means `--event-index 56`.

| Figure | Call | Figure path |
|---|---|---|
| hard-zero HMM fits for 5 [cluster] | the Fig 3A-D fit call with `--hard-zero` added | `results/hmm_fits_full_em_mbtraj/<rat>/<session>/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__hardzero/` |
| 5 (col 1) | `replay-hmm viewer --dataset lustig --rat Huey --session rec01 --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__hardzero --kernel sticky_TCMSR --compare-kernel momentum continuous_EM fragmented --sort-by dll --session-backdrop none --event-index 8 --save-initial --no-show` | [Huey rec01 event0008](figures/Huey/rec01/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__hardzero/events/event0008_lap37_ev10_sticky_TCMSR_vs_momentum_vs_continuous_EM_vs_fragmented_traj.png) |
| 5 (col 2) | same with `--rat Lucky --session rec01 --event-index 0` | [Lucky rec01 event0000](figures/Lucky/rec01/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__hardzero/events/event0000_lap76_ev2_sticky_TCMSR_vs_momentum_vs_continuous_EM_vs_fragmented_traj.png) |
| 5 (col 3) | same with `--rat TinyTim --session rec02 --event-index 30` | [TinyTim rec02 event0030](figures/TinyTim/rec02/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__hardzero/events/event0030_lap91_ev1_sticky_TCMSR_vs_momentum_vs_continuous_EM_vs_fragmented_traj.png) |
| 5 (col 4) | same with `--rat Lucky --session rec01 --event-index 56` | [Lucky rec01 event0056](figures/Lucky/rec01/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__hardzero/events/event0056_lap58_ev5_sticky_TCMSR_vs_momentum_vs_continuous_EM_vs_fragmented_traj.png) |
| 5 (col 5) | same with `--rat TinyTim --session rec02 --event-index 87` | [TinyTim rec02 event0087](figures/TinyTim/rec02/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__hardzero/events/event0087_lap63_ev1_sticky_TCMSR_vs_momentum_vs_continuous_EM_vs_fragmented_traj.png) |

#### Other viewers (not manuscript figures; interactive)
The default `--session-backdrop saved` needs that dataset's jump distribution for the tag: S9C for Pfeiffer, S7C/E for Buzsaki.

| Figure | Call | Figure path |
|---|---|---|
| Pfeiffer viewer | `replay-hmm viewer --dataset pfeiffer --rat Rat1 --session Linear1 --run-tag dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens --kernel sticky_TCMSR --compare-kernel momentum continuous_EM fragmented --sort-by dll` | saves on `s` under `figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/Rat1/Linear1/<tag>/events/` |
| Buzsaki viewer | `replay-hmm viewer --dataset buzsaki --session Achilles_10252013 --run-tag dtbin_0p01__step_0p01__hmmbin_0p02 --kernel sticky_TCMSR --compare-kernel momentum continuous_EM fragmented --sort-by dll` | saves on `s` under `figures/buzsaki/buzsaki_hmm_fits_full_em/Achilles_10252013/<tag>/events/` |

## Tables S1 (Lustig), S2 (Pfeiffer), S3 (Buzsaki): event counts
Run the six data-budget rows, then build the tables.

| Figure | Call | Figure path |
|---|---|---|
| results for S1 | `replay-hmm plot cv-budget --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --save-data results/cv_data_budget_traj.csv` | `results/cv_data_budget_traj.csv`, [hmm_cv_data_budget](figures/hmm_cv_data_budget_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67.png) |
| results for S2 | `replay-hmm plot cv-budget --dataset pfeiffer --run-tag 'dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens' --save-data results/pfeiffer_cv_data_budget_traj.csv` | `results/pfeiffer_cv_data_budget_traj.csv`, [hmm_cv_data_budget](figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/hmm_cv_data_budget_dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens.png) |
| results for S3 | `replay-hmm plot cv-budget --dataset buzsaki --hmm-bin-width-cm 2.0 --save-data results/buzsaki_cv_data_budget_traj.csv` | `results/buzsaki_cv_data_budget_traj.csv`, [buzsaki_hmm_cv_data_budget](figures/buzsaki/buzsaki_hmm_fits_full_em/buzsaki_hmm_cv_data_budget_dtbin_0p01__step_0p01__hmmbin_0p02.png) |
| results for tradreplay tables | `replay-hmm plot cv-budget --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__tradreplay --save-data results/cv_data_budget_tradreplay.csv`; `replay-hmm plot cv-budget --dataset pfeiffer --run-tag 'dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens__tradreplay' --save-data results/pfeiffer_cv_data_budget_tradreplay.csv`; `replay-hmm plot cv-budget --dataset buzsaki --run-tag dtbin_0p01__step_0p01__hmmbin_0p02__tradreplay --save-data results/buzsaki_cv_data_budget_tradreplay.csv` | the three `results/*cv_data_budget_tradreplay.csv` |
| S1, S2, S3 (+ tradreplay) | `replay-hmm tables cv-budget` | [lustig](report/tables/cv_data_budget_lustig.tex), [pfeiffer](report/tables/cv_data_budget_pfeiffer.tex), [buzsaki](report/tables/cv_data_budget_buzsaki.tex); tradreplay: [lustig](report/tables/cv_data_budget_lustig_tradreplay.tex), [pfeiffer](report/tables/cv_data_budget_pfeiffer_tradreplay.tex), [buzsaki](report/tables/cv_data_budget_buzsaki_tradreplay.tex) |

## Fig S2 (Grosmark and Pfeiffer HMM comparison)
| Figure | Call | Figure path |
|---|---|---|
| results for S2A | `replay-hmm fit --dataset buzsaki --hmm-bin-width-cm 2.0 --num-em-iters 200` | `results/buzsaki_hmm_fits/<session>/dtbin_0p01__step_0p01__hmmbin_0p02/` |
| S2A | `replay-hmm plot cv-strip --dataset buzsaki --hmm-bin-width-cm 2.0` | [relative_LL_strip hmmbin_0p02](figures/buzsaki/buzsaki_hmm_fits_full_em/buzsaki_hmm_mbtraj_cv_relative_LL_strip_dtbin_0p01__step_0p01__hmmbin_0p02.png) |
| results for S2C (needs the Buzsaki traditional decode) | `replay-hmm fit --dataset buzsaki --traditional-events-csv results/buzsaki_traditional_decode/bin_0p04__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8/traditional_decode_events.csv --hmm-bin-width-cm 2 --num-em-iters 200 --n-jobs 3` | `results/buzsaki_hmm_fits/<session>/dtbin_0p01__step_0p01__hmmbin_0p02__tradreplay/` (Buddy has no accepted events) |
| S2C | `replay-hmm plot cv-strip --dataset buzsaki --run-tag 'dtbin_0p01__step_0p01__hmmbin_0p02__tradreplay'` | [relative_LL_strip tradreplay](figures/buzsaki/buzsaki_hmm_fits_full_em/buzsaki_hmm_mbtraj_cv_relative_LL_strip_dtbin_0p01__step_0p01__hmmbin_0p02__tradreplay.png) |
| results for S2E (4 of 5 sessions; Cicero_09012014 has none) | `replay-hmm fit --dataset buzsaki --hmm-bin-width-cm 2.0 --decode-tbin-s 0.003 --decode-tbin-step-s 0.003 --num-em-iters 200 --all-pbes --n-jobs 4` | `results/buzsaki_hmm_fits/<session>/dtbin_0p003__step_0p003__hmmbin_0p02__allpbe/` |
| S2E | `replay-hmm plot cv-strip --dataset buzsaki --hmm-bin-width-cm 2.0 --decode-tbin-s 0.003 --decode-tbin-step-s 0.003 --all-pbes` | [relative_LL_strip 3 ms allpbe](figures/buzsaki/buzsaki_hmm_fits_full_em/buzsaki_hmm_mbtraj_cv_relative_LL_strip_dtbin_0p003__step_0p003__hmmbin_0p02__allpbe.png) |
| results for S2B (also needed by Fig 2B/2C) | `replay-hmm fit --dataset pfeiffer --hmm-bin-width-cm 4.0 --num-em-iters 200 --event-detection-source spike-density --n-jobs 8` | `results/PfeifferFoster2015/hmm_fits/<rat>/<session>/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/` |
| (optional) event cache for the best-kernel share figures `plot cv-strip` draws with S2B | `replay-hmm score-events --dataset pfeiffer --run-tag dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens` | `results/PfeifferFoster2015/hmm_fits/replay_classification/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/event_kernel_evidence.csv` |
| S2B | `replay-hmm plot cv-strip --dataset pfeiffer --run-tag 'dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens'` | [relative_LL_strip spikedens](figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/hmm_mbtraj_cv_relative_LL_strip_dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens_minev10.png) |
| results for S2D (needs the Pfeiffer traditional decode) | `replay-hmm fit --dataset pfeiffer --hmm-bin-width-cm 4.0 --num-em-iters 200 --event-detection-source spike-density --traditional-events-csv results/PfeifferFoster2015/traditional_decode/bin_4__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8__src_spikedens/traditional_decode_events.csv --n-jobs 6` | `results/PfeifferFoster2015/hmm_fits/<rat>/<session>/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens__tradreplay/` |
| (optional) event cache for the best-kernel share figures `plot cv-strip` draws with S2D | `replay-hmm score-events --dataset pfeiffer --run-tag dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens__tradreplay --traditional-events-csv results/PfeifferFoster2015/traditional_decode/bin_4__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8__src_spikedens/traditional_decode_events.csv` | `results/PfeifferFoster2015/hmm_fits/replay_classification/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens__tradreplay/event_kernel_evidence.csv` |
| S2D | `replay-hmm plot cv-strip --dataset pfeiffer --run-tag 'dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens__tradreplay'` | [relative_LL_strip tradreplay](figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/hmm_mbtraj_cv_relative_LL_strip_dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens__tradreplay_minev10.png) |
| results for S2F | `replay-hmm fit --dataset pfeiffer --hmm-bin-width-cm 4.0 --decode-tbin-s 0.003 --decode-tbin-step-s 0.003 --num-em-iters 200 --event-detection-source spike-density --n-jobs 3 --all-pbes` | `results/PfeifferFoster2015/hmm_fits/<rat>/<session>/dtbin_0p003__step_0p003__hmmbin_4__evt_spikedens__allpbe/` |
| (optional) event cache for the best-kernel share figures `plot cv-strip` draws with S2F | `replay-hmm score-events --dataset pfeiffer --run-tag dtbin_0p003__step_0p003__hmmbin_4__evt_spikedens__allpbe` | `results/PfeifferFoster2015/hmm_fits/replay_classification/dtbin_0p003__step_0p003__hmmbin_4__evt_spikedens__allpbe/event_kernel_evidence.csv` |
| S2F | `replay-hmm plot cv-strip --dataset pfeiffer --run-tag 'dtbin_0p003__step_0p003__hmmbin_4__evt_spikedens__allpbe'` | [relative_LL_strip 3 ms allpbe](figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/hmm_mbtraj_cv_relative_LL_strip_dtbin_0p003__step_0p003__hmmbin_4__evt_spikedens__allpbe_minev10.png) |

## Fig S4 (Lustig HMM additional comparisons)
| Figure | Call | Figure path |
|---|---|---|
| results for S4A [cluster] (needs the Lustig traditional decode) | the Fig 3A-D fit call with `--sigma-cap 3 --traditional-events-csv results/lustig_traditional_decode/bwcm3p5__dtbin_0p02__step_0p005__pmin_0p01__cells_4__corr_0p8__steps_8__2dbins_100__physarm/traditional_decode_events.csv` added | `results/hmm_fits_full_em_mbtraj/<rat>/<session>/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__tradreplay/` |
| S4A | `replay-hmm plot cv-strip --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__tradreplay --min-events 10` | [relative_LL_strip tradreplay](figures/hmm_mbtraj_cv_relative_LL_strip_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__tradreplay_minev10.png) |
| results for S4B (needs the Fig 3 fits) | `replay-hmm cross-session-ll --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --kernels sticky_TCMSR continuous_EM momentum sticky_SR --min-test-events 10 --sessions TinyTim/rec01 TinyTim/rec02 TinyTim/rec03 TinyTim/rec04 TinyTim/rec05 TinyTim/rec06 Lucky/rec01 Lucky/rec02 Lucky/rec03 Lucky/rec04 Lucky/rec05 Ned/rec01 Ned/rec02 Ned/rec03 Ned/rec04 --save-data results/cross_session_transfer/cross_session_ll_traj_pairs.csv` | `results/cross_session_transfer/<rat>/<session>/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/cross_session_within_rat_summary.csv` (Lucky, Ned, TinyTim: within-rat pairs need ≥2 sessions) |
| S4B | `replay-hmm plot kernel-contrast --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --min-events 10 --rivals continuous_EM momentum --results-root results/cross_session_transfer --summary-name cross_session_within_rat_summary.csv --out-suffix _transfer_momentum` | [kernel_contrast_transfer_momentum](figures/hmm_kernel_contrast_sticky_TCMSR_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67_transfer_momentum_minev10.png) |
| results for S4C [cluster; resubmit until complete] | the Fig 3A-D fit call with `--decode-tbin-s 0.003 --decode-tbin-step-s 0.003 --all-pbes` | `results/hmm_fits_full_em_mbtraj/<rat>/<session>/dtbin_0p003__step_0p003__bwcm3p5__cfgf58e7e21b2__allpbe/` (16 of 19 sessions) |
| S4C (every 3 ms session has ≥62 events, so no event floor) | `replay-hmm plot cv-strip --dataset lustig --run-tag dtbin_0p003__step_0p003__bwcm3p5__cfgf58e7e21b2__allpbe` | [relative_LL_strip 3 ms allpbe](figures/hmm_mbtraj_cv_relative_LL_strip_dtbin_0p003__step_0p003__bwcm3p5__cfgf58e7e21b2__allpbe.png) |
| results for S4D [cluster] | the Fig 3A-D fit call with `--center-arm-only` | `results/hmm_fits_full_em_mbtraj/<rat>/<session>/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__centerarm/` (17 of 19 sessions fitted; 13 pass the 10-event CV floor and are plotted in S4D) |
| S4D | `replay-hmm plot cv-strip --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__centerarm --min-events 10` | [relative_LL_strip centerarm](figures/hmm_mbtraj_cv_relative_LL_strip_dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__centerarm_minev10.png) |

## Fig S5
Rendered by the Fig 3E plot call; see the Fig 3E, 3F, S5 table.

## Fig S6 (Grosmark and Pfeiffer synthetic recovery)
Synthetic model recovery: A Grosmark & Buzsáki (on the S2A fits), B Pfeiffer & Foster (on the S2B fits).

| Figure | Call | Figure path |
|---|---|---|
| results for S6A [cluster] (the three sessions S2A plots; Cicero_09172014 and Buddy fall under its 10-CV-event floor) | `replay-hmm submit recovery --dataset buzsaki --sessions Achilles_10252013 Cicero_09012014 Gatsby_08022013 --run-tag dtbin_0p01__step_0p01__hmmbin_0p02 --generator-random-sample --random-sample-size 1000 --num-em-iters 50 --random-fixed-generator-reps 10 --random-sample-seed 42 --generator-num-shards 40` | `results/buzsaki_hmm_fits/<session>/dtbin_0p01__step_0p01__hmmbin_0p02/synthetic_recovery/synthetic_recovery_random_rows_shard*-of0040.csv` |
| S6A | `replay-hmm recovery plot results/buzsaki_hmm_fits/Achilles_10252013/dtbin_0p01__step_0p01__hmmbin_0p02/synthetic_recovery/*of0040.csv results/buzsaki_hmm_fits/Cicero_09012014/dtbin_0p01__step_0p01__hmmbin_0p02/synthetic_recovery/*of0040.csv results/buzsaki_hmm_fits/Gatsby_08022013/dtbin_0p01__step_0p01__hmmbin_0p02/synthetic_recovery/*of0040.csv --dataset buzsaki --figures-root figures/buzsaki/buzsaki_hmm_fits_full_em/recovery_pooled --run-tag dtbin_0p01__step_0p01__hmmbin_0p02 --generator-random-sample --kernels sticky_TCMSR sticky_TCMSR_sym sticky_SR sticky_SR_rev continuous_EM momentum movement_empirical fragmented stationary --grid-generator-kernels sticky_TCMSR sticky_TCMSR_sym sticky_SR sticky_SR_rev continuous_EM momentum --random-fixed-generator-kernels movement_empirical stationary fragmented` | [winner_confusion_random](figures/buzsaki/buzsaki_hmm_fits_full_em/recovery_pooled/synthetic_recovery/synthetic_recovery_winner_confusion_random_dtbin_0p01__step_0p01__hmmbin_0p02.png) |
| results for S6B [cluster] (4 of the 9 S2B sessions, from the smallest, Rat1 Linear4 with 25 events, to Rat5 Linear1 with 161; `--rats` × `--sessions` matches exactly Rat1 Linear1, Rat1 Linear4, Rat2 Linear1 and Rat5 Linear1) | `replay-hmm submit recovery --dataset pfeiffer --rats Rat1 Rat2 Rat5 --sessions Linear1 Linear4 --run-tag dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens --generator-random-sample --random-sample-size 1000 --num-em-iters 50 --random-fixed-generator-reps 10 --random-sample-seed 42 --generator-num-shards 40` | `results/PfeifferFoster2015/hmm_fits/<rat>/<session>/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/synthetic_recovery/synthetic_recovery_random_rows_shard*-of0040.csv` |
| S6B | `replay-hmm recovery plot results/PfeifferFoster2015/hmm_fits/Rat1/Linear1/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/synthetic_recovery/*of0040.csv results/PfeifferFoster2015/hmm_fits/Rat1/Linear4/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/synthetic_recovery/*of0040.csv results/PfeifferFoster2015/hmm_fits/Rat2/Linear1/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/synthetic_recovery/*of0040.csv results/PfeifferFoster2015/hmm_fits/Rat5/Linear1/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/synthetic_recovery/*of0040.csv --dataset pfeiffer --figures-root figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/recovery_pooled --run-tag dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens --generator-random-sample --kernels sticky_TCMSR sticky_TCMSR_sym sticky_SR sticky_SR_rev continuous_EM momentum movement_empirical fragmented stationary --grid-generator-kernels sticky_TCMSR sticky_TCMSR_sym sticky_SR sticky_SR_rev continuous_EM momentum --random-fixed-generator-kernels movement_empirical stationary fragmented` | [winner_confusion_random](figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/recovery_pooled/synthetic_recovery/synthetic_recovery_winner_confusion_random_dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens.png) |

## Fig S7
Memoryless-Bayes jumps (A, B) and HMM prior-generated jumps (C-F).

| Figure | Call | Figure path |
|---|---|---|
| S7A (decode grid 20 ms window / 20 ms step: the `--decode-tbin-s` / `--decode-tbin-step-s` defaults, 0.020 / 0.020; the grid is not in the filename, and a non-default grid writes to its own `dtbin_<W>__step_<S>` subdirectory) | `replay-hmm mb-jump --dataset buzsaki --hover-thresh-bins 0` | [MB_jump_dist_pooled_excl_low_yield](figures/buzsaki/MB_jump_dist_pooled_excl_low_yield_Achilles-Cicero-Gatsby_hover0bins.png) |
| S7B | `replay-hmm mb-jump --dataset pfeiffer --event-source spike-density --bin-width-cm 4` | [MB_jump_dist_pooled spikedens](figures/PfeifferFoster2015/mb_jump/MB_jump_dist_pooled__bin_4__dtbin_0p02__step_0p02__hover_0__mass_0p6__mindur_0p065__src_spikedens.png) |
| S7C, S7E (also the Buzsaki viewer backdrop) | `replay-hmm jumps --dataset buzsaki --hmm-bin-width-cm 2 --n-jobs 5` | S7C: [momentum_generated](figures/buzsaki/buzsaki_hmm_fits_full_em/hmm_jump_distribution/dtbin_0p01__step_0p01__hmmbin_0p02/momentum_generated_excl_low_yield_jump_distribution_all_sessions.png)<br>S7E: [sticky_TCMSR_generated](figures/buzsaki/buzsaki_hmm_fits_full_em/hmm_jump_distribution/dtbin_0p01__step_0p01__hmmbin_0p02/sticky_TCMSR_generated_jump_distribution_all_sessions.png) |
| S7D, S7F, S9C (also the Pfeiffer viewer backdrop) | `replay-hmm jumps --dataset pfeiffer --hmm-bin-width-cm 4.0 --event-detection-source spike-density --n-jobs 4` | S9C: [sym_traj_filtered_by_rat_row](figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/hmm_jump_distribution/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/sticky_TCMSR_sym_traj_filtered_jump_distribution_by_rat_row.png)<br>S7D: [momentum_generated](figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/hmm_jump_distribution/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/momentum_generated_jump_distribution_all_sessions.png)<br>S7F: [sym_generated](figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/hmm_jump_distribution/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/sticky_TCMSR_sym_generated_jump_distribution_all_sessions.png) |

## Fig S8
Q-Q grids: A Lustig, B Grosmark & Buzsáki, C Pfeiffer & Foster. Add `--plot-only` to re-render from saved PPC results.

| Figure | Call | Figure path |
|---|---|---|
| S8A | `replay-hmm ppc --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --qq-include-zero --qq-symlog` | [jump_qq_grid](figures/hmm_kernel_ppc/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/all_sessions__jump_qq_grid_with_zero_symlog.png) |
| S8B | `replay-hmm ppc --dataset buzsaki --run-tag dtbin_0p01__step_0p01__hmmbin_0p02 --qq-include-zero --qq-symlog` | [jump_qq_grid](figures/buzsaki/buzsaki_hmm_fits_full_em/hmm_kernel_ppc/dtbin_0p01__step_0p01__hmmbin_0p02/all_sessions__jump_qq_grid_with_zero_symlog.png) |
| S8C | `replay-hmm ppc --dataset pfeiffer --run-tag dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens --qq-include-zero --qq-symlog` | [jump_qq_grid](figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/hmm_kernel_ppc/dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens/all_sessions__jump_qq_grid_with_zero_symlog.png) |

## Fig S9
HMM-decoded jumps: Viterbi decoding with post-decode trajectory filtering.

| Figure | Call | Figure path |
|---|---|---|
| S9A (same call as Fig 4C-F) | `replay-hmm jumps --dataset lustig --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67 --n-jobs 4` | [by_rat_grid4](figures/hmm_jump_distribution/dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67/sticky_TCMSR_traj_filtered_jump_distribution_by_rat_grid4.png) |
| S9B | `replay-hmm jumps --dataset buzsaki --hmm-bin-width-cm 2 --exclude-map-edge-bins 2 --n-jobs 5 --hover-thresh-bins 1` | [by_rat_row](figures/buzsaki/buzsaki_hmm_fits_full_em/hmm_jump_distribution/dtbin_0p01__step_0p01__hmmbin_0p02__edge2__hover1/sticky_TCMSR_traj_filtered_excl_low_yield_jump_distribution_by_rat_row.png) |
| S9C | the S7D, S7F call above (one Pfeiffer jump-distribution run writes all three) | see the S7D, S7F, S9C row |

