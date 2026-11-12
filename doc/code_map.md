# Code map: `src/replay_hmm/`

A reader's map of the `replay_hmm` package. Every README row is a
`replay-hmm` command; the dispatcher (`cli/__main__.py`) imports only that command's module,
which imports the rest.

The package is layered. The generic layers (1-2) never import a dataset, analysis, plot, viewer,
report or CLI module; datasets (3) build on them; analyses, viewer, plots and reports (4) on
the datasets; and the dispatcher and the cluster submitter (5) sit on top. The light modules marked *stdlib* import nothing heavier than the standard library,
so the dispatcher's help and a run tag cost no numpy.

Command shorthand in the "Reached by" column: *fit* is `replay-hmm fit --dataset
lustig|buzsaki|pfeiffer` (the HMM fits of Fig 2B/C, Table 1, 3A-F, 4C-F, 5, S1-S6, S7C-F, S8,
S9; on the cluster through `replay-hmm submit fit --dataset <ds>`), *jumps* is `replay-hmm jumps
--dataset lustig|buzsaki|pfeiffer` (3A, 4C-F, S9, S7C-F), *every command* means the module is
imported by all of them.

## Package root

| Module | What it holds | Reached by |
|---|---|---|
| [`__init__.py`](../src/replay_hmm/__init__.py), [`__main__.py`](../src/replay_hmm/__main__.py) | The package version; `python -m replay_hmm` as the `replay-hmm` console script | every command |
| [`paths.py`](../src/replay_hmm/paths.py) | *stdlib.* The repo root (`REPLAY_HMM_ROOT` or the walk-up to `pyproject.toml`), the data/results/figures/report roots and their overrides, the per-dataset subtrees | every command |

## Layer 1: `util/`, `io/`

| Module | What it holds | Reached by |
|---|---|---|
| [`util/tags.py`](../src/replay_hmm/util/tags.py) | *stdlib.* Run-tag tokens: how floats, rat lists and free tags are spelled in directory names | every command that names a run |
| [`util/argtypes.py`](../src/replay_hmm/util/argtypes.py) | *stdlib.* argparse `type=` callables | every parser |
| [`util/cli.py`](../src/replay_hmm/util/cli.py) | The argparse group the two-map fit CLIs share (`add_em_args`: the EM flags) | `replay-hmm fit --dataset buzsaki\|pfeiffer` |
| [`util/jsonio.py`](../src/replay_hmm/util/jsonio.py) | JSON encoding of numpy scalars and arrays | `replay-hmm ppc` |
| [`util/mpl.py`](../src/replay_hmm/util/mpl.py) | *stdlib.* `configure_for_cli`: the Matplotlib backend and config directory each command runs under | the dispatcher, before any command |
| [`util/slurm.py`](../src/replay_hmm/util/slurm.py) | What a SLURM job hands its worker (`--n-jobs` from the cores, `--slurm-array` from the task id) | the cluster workers (`fit`, `recovery run`, `misspec sweep`) |
| [`io/names.py`](../src/replay_hmm/io/names.py) | *stdlib.* One constant per persisted file and directory name | every command |
| [`io/matlab.py`](../src/replay_hmm/io/matlab.py) | `.mat` reading (v7 and v7.3) into plain dicts and arrays | every command that loads a session |
| [`io/fit_summary.py`](../src/replay_hmm/io/fit_summary.py) | *stdlib.* Readers of a saved fit summary (event source, arm, momentum grid) | *jumps*, `replay-hmm score-events`, `replay-hmm ppc`, `replay-hmm viewer` |
| [`io/fit_tree.py`](../src/replay_hmm/io/fit_tree.py) | *stdlib.* Finding fitted runs in a results tree (`select_saved_sessions`: the saved sessions under a run tag, filtered by `--rats`/`--sessions`); rat/session/tag from a path | the plots, `replay-hmm ppc`, `replay-hmm cross-session-ll`, `replay-hmm viewer` |
| [`io/cv_tables.py`](../src/replay_hmm/io/cv_tables.py) | The per-session CV tables the fits write, read for the plotters | `replay-hmm plot cv-strip`, `replay-hmm plot direction-share`, `replay-hmm plot kernel-contrast` |
| [`io/evidence.py`](../src/replay_hmm/io/evidence.py) | *stdlib at import.* The per-event kernel evidence caches: layout, columns, readers | `replay-hmm score-events`, `replay-hmm plot cv-strip`, `replay-hmm plot direction-share` (3C, 3D) |
| [`io/jump_store.py`](../src/replay_hmm/io/jump_store.py) | The Lustig jump distribution's saved arrays (jump npz, transition kernels) | `replay-hmm jumps --dataset lustig`, `replay-hmm viewer` |
| [`io/run_paths.py`](../src/replay_hmm/io/run_paths.py) | Paths of one fit run directory and of its synthetic-recovery subtree | `replay-hmm jumps --dataset lustig`, `replay-hmm score-events --dataset lustig`, `replay-hmm recovery run` |

## Layer 2: `decode/`, `events/`, `hmm/`, `jumps/`, `maps/`

| Module | What it holds | Reached by |
|---|---|---|
| [`decode/config.py`](../src/replay_hmm/decode/config.py) | *stdlib.* `DecodeConfig`, its `cfg<hash>` run tag and the flags generated from it | *fit*, *jumps*, `replay-hmm score-events`, `replay-hmm mb-jump --dataset lustig` |
| [`decode/grid.py`](../src/replay_hmm/decode/grid.py) | `DecodeGrid` (decode window and step, the type of every dataset's `grids.py` defaults); the Lustig decode windows on the 20 kHz spike timebase | every command that decodes |
| [`decode/poisson.py`](../src/replay_hmm/decode/poisson.py) | The memoryless Poisson decoder: cache, posterior, confidence gates | every Lustig command |
| [`decode/two_map.py`](../src/replay_hmm/decode/two_map.py) | The sliding-window decode on a two-map `[outbound \| inbound]` state space | every Buzsaki and Pfeiffer command |
| [`decode/traditional/criteria.py`](../src/replay_hmm/decode/traditional/criteria.py) | The traditional replay criteria (weighted Spearman, step count, verdict) every dataset applies | `replay-hmm traditional decode --dataset lustig\|buzsaki\|pfeiffer` (4A, S2C, S2D) |
| [`events/runs.py`](../src/replay_hmm/events/runs.py) | Runs of a boolean mask and gap merging | every command that calls events |
| [`events/trajectories.py`](../src/replay_hmm/events/trajectories.py) | Per-map posterior mass and MB trajectory calling | *fit*, *jumps*, `replay-hmm mb-jump`, `replay-hmm ppc` |
| [`hmm/kernel_sets.py`](../src/replay_hmm/hmm/kernel_sets.py) | *stdlib.* The kernel vocabulary: the ballot, its partitions, the off-ballot arms | every HMM command and plot |
| [`hmm/constants.py`](../src/replay_hmm/hmm/constants.py), [`hmm/tags.py`](../src/replay_hmm/hmm/tags.py) | Kernel caps and inits; the fit run-tag suffixes (`__allpbe`, `__hardzero`, ...) | *fit* and every reader of a fit tag |
| [`hmm/base_matrices.py`](../src/replay_hmm/hmm/base_matrices.py), [`hmm/build.py`](../src/replay_hmm/hmm/build.py) | Fixed base transition matrices; the eps=0 freeze and hard-zero arm helpers | *fit* and every rebuild of a saved fit |
| [`hmm/two_map_kernels.py`](../src/replay_hmm/hmm/two_map_kernels.py) | The two-map (Buzsaki, Pfeiffer) kernel factory | `replay-hmm fit --dataset buzsaki\|pfeiffer` (S2) |
| [`hmm/two_map_movement.py`](../src/replay_hmm/hmm/two_map_movement.py) | The two-map empirical movement counter (outbound/inbound transitions of behaviour) and `DEFAULT_BEHAVIOR_DT_S`, behind each two-map dataset's `movement.py` | `replay-hmm fit --dataset buzsaki\|pfeiffer` (S2) |
| [`hmm/momentum.py`](../src/replay_hmm/hmm/momentum.py) | The order-2 momentum kernel and its own forward-backward | every fit and rebuild |
| [`hmm/momentum_sim.py`](../src/replay_hmm/hmm/momentum_sim.py) | Momentum-diffusion simulation on a linear track | no command: `tests/test_hmm_order2_momentum.py` checks the fitted kernel against it |
| [`hmm/em.py`](../src/replay_hmm/hmm/em.py), [`hmm/cv.py`](../src/replay_hmm/hmm/cv.py) | Full EM for one kernel; K-fold CV plus the full-data fit | *fit*, `replay-hmm recovery run`, `replay-hmm misspec` |
| [`hmm/checkpoints.py`](../src/replay_hmm/hmm/checkpoints.py) | Resumable CV: one checkpoint per finished fit, keyed by a hash of the fit code | *fit* (resubmitted cluster jobs) |
| [`hmm/fit_session.py`](../src/replay_hmm/hmm/fit_session.py) | The dataset-agnostic half of one session's fit: CV loop, tables, writers | *fit* |
| [`hmm/restore.py`](../src/replay_hmm/hmm/restore.py), [`hmm/rebuild_two_map.py`](../src/replay_hmm/hmm/rebuild_two_map.py), [`hmm/scoring.py`](../src/replay_hmm/hmm/scoring.py) | Rebuilding a saved fit from its summary (the saved-matrix check; for the two-map fits the provenance check, saved inputs, HMMs and below-floor restore), and the stored-loglik identity it is checked against | *jumps*, `replay-hmm score-events`, `replay-hmm ppc`, `replay-hmm viewer`, `replay-hmm cross-session-ll`, `replay-hmm misspec` |
| [`hmm/sampling.py`](../src/replay_hmm/hmm/sampling.py) | The generated-null samplers of the jump distributions | *jumps*, `replay-hmm misspec` |
| [`jumps/mb.py`](../src/replay_hmm/jumps/mb.py), [`jumps/signed.py`](../src/replay_hmm/jumps/signed.py) | Signed MB jumps from the posterior; signed jumps from an HMM state path | `replay-hmm mb-jump` (4B, S7A, S7B), *jumps* |
| [`jumps/stats.py`](../src/replay_hmm/jumps/stats.py), [`jumps/pooling.py`](../src/replay_hmm/jumps/pooling.py) | Directionality and the asymmetry tests; reference-bin pooling across sessions | `replay-hmm mb-jump`, `replay-hmm traditional jumps --dataset lustig`, *jumps* |
| [`jumps/hmm_distribution.py`](../src/replay_hmm/jumps/hmm_distribution.py) | What the three HMM jump-distribution commands share | *jumps* |
| [`maps/smoothing.py`](../src/replay_hmm/maps/smoothing.py) | Occupancy-normalised Gaussian map smoothing (port of `na_smoothMap`) | `replay-hmm maps build --dataset lustig`, the Pfeiffer rate maps |
| [`maps/placefields.py`](../src/replay_hmm/maps/placefields.py) | Per-cell map normalisation and peak bins | `replay-hmm diagnose direction-check --dataset pfeiffer` (through `analyses/diagnostics/direction_check.py`) |

## Layer 3: `datasets/`

| Module | What it holds | Reached by |
|---|---|---|
| [`datasets/registry.py`](../src/replay_hmm/datasets/registry.py), [`datasets/display.py`](../src/replay_hmm/datasets/display.py) | *stdlib.* Which sessions each analysis runs on (including `BUZSAKI_LINEAR_SESSIONS`, the Buzsaki fit's sessions); the figure labels of animals and sessions | every command and plot |
| [`datasets/base.py`](../src/replay_hmm/datasets/base.py) | The adapter protocols the dataset-agnostic fit engine (`hmm/fit_session.py`) takes; each dataset's `adapters` implements them | *fit* |
| [`datasets/two_map_event_sets.py`](../src/replay_hmm/datasets/two_map_event_sets.py) | The Buzsaki and Pfeiffer HMM event sequences (trajectory spans or whole events) and their traditional-event provenance, read through each dataset's decoded-event view | *fit*, *jumps*, `replay-hmm score-events`, `replay-hmm ppc` |
| [`datasets/lustig/`](../src/replay_hmm/datasets/lustig/) | The Lee-lab M-maze: [`constants`](../src/replay_hmm/datasets/lustig/constants.py), [`labels`](../src/replay_hmm/datasets/lustig/labels.py) (arm/slot names), [`sessions`](../src/replay_hmm/datasets/lustig/sessions.py) and [`units`](../src/replay_hmm/datasets/lustig/units.py), [`calibration`](../src/replay_hmm/datasets/lustig/calibration.py) (mm per pixel), [`events`](../src/replay_hmm/datasets/lustig/events.py) (PBE detection), [`bundle`](../src/replay_hmm/datasets/lustig/bundle.py) (one decoded session), [`event_sets`](../src/replay_hmm/datasets/lustig/event_sets.py), [`grids`](../src/replay_hmm/datasets/lustig/grids.py) (the MB, HMM and traditional decode-grid defaults), [`movement`](../src/replay_hmm/datasets/lustig/movement.py) (`empirical_movement_P`), [`kernels`](../src/replay_hmm/datasets/lustig/kernels.py), [`session`](../src/replay_hmm/datasets/lustig/session.py) and [`rebuild`](../src/replay_hmm/datasets/lustig/rebuild.py), [`adapters`](../src/replay_hmm/datasets/lustig/adapters.py) | every Lustig command |
| [`datasets/lustig/maps/`](../src/replay_hmm/datasets/lustig/maps/) | The port of the lab's MATLAB map pipeline: session loading, linearization, map building, the npz archive, the batch driver | `replay-hmm maps build --dataset lustig` (prerequisites) |
| [`datasets/lustig/fit.py`](../src/replay_hmm/datasets/lustig/fit.py) | The Lustig fit command | `replay-hmm fit --dataset lustig` |
| [`datasets/lustig/jumps.py`](../src/replay_hmm/datasets/lustig/jumps.py) | Jump distributions from the saved Lustig fits | `replay-hmm jumps --dataset lustig` (3A, 4C-F, S9A) |
| [`datasets/lustig/mb_jump.py`](../src/replay_hmm/datasets/lustig/mb_jump.py) | Memoryless-Bayes jump decoding from PBEs (figures in `plots/mb/`) | `replay-hmm mb-jump --dataset lustig` (4B) |
| [`datasets/lustig/traditional.py`](../src/replay_hmm/datasets/lustig/traditional.py), [`traditional_io.py`](../src/replay_hmm/datasets/lustig/traditional_io.py), [`traditional_jumps.py`](../src/replay_hmm/datasets/lustig/traditional_jumps.py) | The traditional decode, its event tables, and their jump distribution | `replay-hmm traditional decode --dataset lustig`, `replay-hmm traditional jumps --dataset lustig` (4A, S4A) |
| [`datasets/buzsaki/`](../src/replay_hmm/datasets/buzsaki/) | Grosmark & Buzsáki hc-11: [`loader`](../src/replay_hmm/datasets/buzsaki/loader.py), [`kinematics`](../src/replay_hmm/datasets/buzsaki/kinematics.py), [`ratemaps`](../src/replay_hmm/datasets/buzsaki/ratemaps.py), [`events`](../src/replay_hmm/datasets/buzsaki/events.py) (PBE and ripple events), [`session`](../src/replay_hmm/datasets/buzsaki/session.py) (detect and decode; `rat_of_session`), [`grids`](../src/replay_hmm/datasets/buzsaki/grids.py) (decode-grid defaults), [`constants`](../src/replay_hmm/datasets/buzsaki/constants.py) (the fit's default roots, kernels and HMM bin width), [`movement`](../src/replay_hmm/datasets/buzsaki/movement.py) (`empirical_movement_P`), [`rebuild`](../src/replay_hmm/datasets/buzsaki/rebuild.py) (a saved fit's events, `rebuild_events`), [`adapters`](../src/replay_hmm/datasets/buzsaki/adapters.py) | every Buzsaki command |
| [`datasets/buzsaki/fit.py`](../src/replay_hmm/datasets/buzsaki/fit.py), [`jumps.py`](../src/replay_hmm/datasets/buzsaki/jumps.py), [`mb_jump.py`](../src/replay_hmm/datasets/buzsaki/mb_jump.py), [`traditional.py`](../src/replay_hmm/datasets/buzsaki/traditional.py), [`kernel_share.py`](../src/replay_hmm/datasets/buzsaki/kernel_share.py) | The Buzsaki fit, jump distribution, per-event kernel scorer, MB jumps and traditional decode | `replay-hmm fit --dataset buzsaki` (S2A, S2C, S2E), `replay-hmm jumps --dataset buzsaki` (S9B, S7C, S7E), `replay-hmm score-events --dataset buzsaki`, `replay-hmm mb-jump --dataset buzsaki` (S7A), `replay-hmm traditional decode --dataset buzsaki` (S2C) |
| [`datasets/pfeiffer/`](../src/replay_hmm/datasets/pfeiffer/) | Pfeiffer & Foster 2015: [`loader`](../src/replay_hmm/datasets/pfeiffer/loader.py), [`kinematics`](../src/replay_hmm/datasets/pfeiffer/kinematics.py), [`ratemaps`](../src/replay_hmm/datasets/pfeiffer/ratemaps.py) (and the `[outbound \| inbound]` map order), [`events`](../src/replay_hmm/datasets/pfeiffer/events.py) (the four event sources), [`session`](../src/replay_hmm/datasets/pfeiffer/session.py) (decode and call trajectories), [`grids`](../src/replay_hmm/datasets/pfeiffer/grids.py) (decode-grid defaults), [`constants`](../src/replay_hmm/datasets/pfeiffer/constants.py) (the fit's default roots, kernels, HMM bin width and event-set labels), [`movement`](../src/replay_hmm/datasets/pfeiffer/movement.py) (`empirical_movement_P`), [`rebuild`](../src/replay_hmm/datasets/pfeiffer/rebuild.py) (a saved fit's events, `rebuild_events`), [`adapters`](../src/replay_hmm/datasets/pfeiffer/adapters.py) | every Pfeiffer command |
| [`datasets/pfeiffer/fit.py`](../src/replay_hmm/datasets/pfeiffer/fit.py), [`jumps.py`](../src/replay_hmm/datasets/pfeiffer/jumps.py), [`mb_jump.py`](../src/replay_hmm/datasets/pfeiffer/mb_jump.py), [`traditional.py`](../src/replay_hmm/datasets/pfeiffer/traditional.py), [`kernel_share.py`](../src/replay_hmm/datasets/pfeiffer/kernel_share.py) | The Pfeiffer fit, jump distribution, per-event kernel scorer, MB jumps and traditional decode | `replay-hmm fit --dataset pfeiffer` (S2B, S2D, S2F), `replay-hmm jumps --dataset pfeiffer` (S9C, S7D, S7F), `replay-hmm score-events --dataset pfeiffer`, `replay-hmm mb-jump --dataset pfeiffer` (S7B), `replay-hmm traditional decode --dataset pfeiffer` (S2D) |

## Layer 4: `analyses/`, `viewer/`, `plots/`, `reports/`

| Module | What it holds | Reached by |
|---|---|---|
| [`analyses/model_comparison.py`](../src/replay_hmm/analyses/model_comparison.py) | *stdlib at import.* CV/BIC deficits and the per-event best-kernel share | `replay-hmm plot cv-strip` (3B, 3C, S2, S3, S4), `replay-hmm plot direction-share` (3D) |
| [`analyses/kernel_share.py`](../src/replay_hmm/analyses/kernel_share.py), [`datasets/lustig/kernel_share.py`](../src/replay_hmm/datasets/lustig/kernel_share.py) | The dataset-agnostic per-event kernel and direction evidence (and the two-map scoring driver); the Lustig scorer | `replay-hmm score-events --dataset lustig\|buzsaki\|pfeiffer` (results for 3C, 3D, S2B/D/F companions) |
| [`analyses/cross_session.py`](../src/replay_hmm/analyses/cross_session.py) | Cross-session transfer of the fitted transition scalars | `replay-hmm cross-session-ll` (results for S4B) |
| [`analyses/ppc/`](../src/replay_hmm/analyses/ppc/) | The kernel posterior-predictive check: [`engine`](../src/replay_hmm/analyses/ppc/engine.py), [`simulate`](../src/replay_hmm/analyses/ppc/simulate.py), [`stats`](../src/replay_hmm/analyses/ppc/stats.py), [`containers`](../src/replay_hmm/analyses/ppc/containers.py), [`traditional`](../src/replay_hmm/analyses/ppc/traditional.py), and one module per dataset ([`lustig`](../src/replay_hmm/analyses/ppc/lustig.py), whose `main` is the Lustig command, and [`two_map`](../src/replay_hmm/analyses/ppc/two_map.py) with [`buzsaki`](../src/replay_hmm/analyses/ppc/buzsaki.py) and [`pfeiffer`](../src/replay_hmm/analyses/ppc/pfeiffer.py)); the engine takes the dataset as an argument | `replay-hmm ppc --dataset lustig\|buzsaki\|pfeiffer` (S8A-C) |
| [`analyses/recovery/`](../src/replay_hmm/analyses/recovery/) | Synthetic model recovery: [`config`](../src/replay_hmm/analyses/recovery/config.py), [`specs`](../src/replay_hmm/analyses/recovery/specs.py), [`run`](../src/replay_hmm/analyses/recovery/run.py), [`shards`](../src/replay_hmm/analyses/recovery/shards.py), [`scoring`](../src/replay_hmm/analyses/recovery/scoring.py) | `replay-hmm recovery run`, `replay-hmm recovery check-shards`, `replay-hmm recovery plot` (3E, 3F, S5, S6) |
| [`analyses/misspec/`](../src/replay_hmm/analyses/misspec/) | Kernel misspecification on synthetic center-arm events: [`exact_zero`](../src/replay_hmm/analyses/misspec/exact_zero.py), [`sim`](../src/replay_hmm/analyses/misspec/sim.py), [`fit`](../src/replay_hmm/analyses/misspec/fit.py), [`metrics`](../src/replay_hmm/analyses/misspec/metrics.py), [`summary`](../src/replay_hmm/analyses/misspec/summary.py), [`gallery`](../src/replay_hmm/analyses/misspec/gallery.py), [`sweep`](../src/replay_hmm/analyses/misspec/sweep.py) | `replay-hmm misspec gallery` (2B, 2C), `replay-hmm misspec sweep` (Table 1) |
| [`analyses/diagnostics/`](../src/replay_hmm/analyses/diagnostics/) | Convention checks against behaviour: the Lustig map slot order, the directional maps and jump sign, and the traversal segmentation the direction check splits laps with | `replay-hmm diagnose map-order`, `replay-hmm diagnose direction-check --dataset pfeiffer` (no README row) |
| [`viewer/`](../src/replay_hmm/viewer/) | The per-event HMM browser: the shared [`engine`](../src/replay_hmm/viewer/engine.py), the Lustig source ([`lustig`](../src/replay_hmm/viewer/lustig.py), [`input_cache`](../src/replay_hmm/viewer/input_cache.py)) and the two-map ones ([`two_map`](../src/replay_hmm/viewer/two_map.py), [`buzsaki`](../src/replay_hmm/viewer/buzsaki.py), [`pfeiffer`](../src/replay_hmm/viewer/pfeiffer.py)) | `replay-hmm viewer --dataset lustig\|buzsaki\|pfeiffer` (Fig 5) |
| [`plots/style.py`](../src/replay_hmm/plots/style.py), [`plots/labels.py`](../src/replay_hmm/plots/labels.py), [`plots/save.py`](../src/replay_hmm/plots/save.py) | The house rcParams (applied at import, the package's one import-time side effect), kernel labels and colours, `savefig` with retries | every figure |
| [`plots/cv_strip/`](../src/replay_hmm/plots/cv_strip/), [`plots/cv_panels.py`](../src/replay_hmm/plots/cv_panels.py), [`plots/kernel_share.py`](../src/replay_hmm/plots/kernel_share.py) | The CV ΔNLL, BIC, fold-parameter and best-kernel-share strips (one engine, one module per dataset) | `replay-hmm plot cv-strip --dataset lustig\|buzsaki\|pfeiffer` (3B, 3C, S2A-F, S3, S4A, S4C, S4D) |
| [`plots/cv_budget/`](../src/replay_hmm/plots/cv_budget/) | The train/validation data budget per session | `replay-hmm plot cv-budget --dataset lustig\|buzsaki\|pfeiffer` (results for Tables S1-S3) |
| [`plots/direction_share.py`](../src/replay_hmm/plots/direction_share.py) | The per-event direction ballot and its figure | `replay-hmm plot direction-share --dataset lustig` (3D) |
| [`plots/kernel_contrast.py`](../src/replay_hmm/plots/kernel_contrast.py) | The paired within-session contrast of one kernel against its rivals | `replay-hmm plot kernel-contrast` (S4B) |
| [`plots/recovery.py`](../src/replay_hmm/plots/recovery.py) | Recovery winners, confusion and parameter recovery | `replay-hmm recovery plot` (3E, 3F, S5, S6) |
| [`plots/misspec.py`](../src/replay_hmm/plots/misspec.py) | The misspecification figures and the RMSE-ratio table | `replay-hmm misspec gallery` (2B, 2C), `replay-hmm misspec sweep` (Table 1) |
| [`plots/ppc.py`](../src/replay_hmm/plots/ppc.py) | The PPC figures (the jump Q-Q grids) | `replay-hmm ppc` (S8A-C) |
| [`plots/jumps.py`](../src/replay_hmm/plots/jumps.py) | The HMM jump-histogram renderers and drawing primitives | *jumps*, `replay-hmm mb-jump`, `replay-hmm traditional jumps --dataset lustig` |
| [`plots/mb/`](../src/replay_hmm/plots/mb/) | The MB jump figures: [`engine`](../src/replay_hmm/plots/mb/engine.py) (the single-axis jump/step histogram and the by-rat row every dataset draws), and one module per dataset ([`lustig`](../src/replay_hmm/plots/mb/lustig.py), [`buzsaki`](../src/replay_hmm/plots/mb/buzsaki.py), [`pfeiffer`](../src/replay_hmm/plots/mb/pfeiffer.py)) | `replay-hmm mb-jump --dataset lustig\|buzsaki\|pfeiffer` (4B, S7A, S7B), `replay-hmm traditional jumps --dataset lustig` (4A) |
| [`plots/fit_diagnostics.py`](../src/replay_hmm/plots/fit_diagnostics.py) | The per-session CV and BIC figures a fit saves | *fit* |
| [`reports/budget_tables.py`](../src/replay_hmm/reports/budget_tables.py) | LaTeX tables of the CV data budget | `replay-hmm tables cv-budget` (Tables S1-S3) |

## Layer 5: `cluster/`, `cli/`

| Module | What it holds | Reached by |
|---|---|---|
| [`cluster/submit.py`](../src/replay_hmm/cluster/submit.py) | *stdlib.* The SLURM job kinds, their resources (`SLURM_PROFILES`) and the `sbatch` dispatch of `slurm/run_replay_hmm.sbatch` | `replay-hmm submit fit --dataset lustig\|buzsaki\|pfeiffer`, `replay-hmm submit recovery`, `replay-hmm submit misspec-sweep`, `replay-hmm submit check-shards` ([cluster] rows) |
| [`cli/__main__.py`](../src/replay_hmm/cli/__main__.py) | *stdlib.* The dispatcher: the first word of a `replay-hmm` command picks one command module, imported only when run | every command |
| [`cli/*.py`](../src/replay_hmm/cli/) | *stdlib.* One thin module per command (`fit`, `jumps`, `score-events`, `traditional`, `mb-jump`, `maps`, `plot`, `tables`, `ppc`, `recovery`, `misspec`, `viewer`, `cross-session-ll`, `diagnose`, `submit`): pre-parse `--dataset` or the subcommand word, then call the module above | that command |
