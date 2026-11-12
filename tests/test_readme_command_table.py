"""Every README command, each command's defaults and each cluster worker, as one table.

COMMANDS is the one id -> (module, argv, cwd) table; the tests here hold it equal to the README,
match on (module, argv, cwd), which `_cli_capture.readme_commands` derives from each README
`replay-hmm` call, and check that every README command parses (nothing runs past the parse).
Ids are
  `<figure>.<what>`     a README row;
  `defaults.<name>`     the no-argument parse (or, where the parser has required flags, the
                        parse with only those flags) of each README command and each worker;
  `worker.<row id>`     the command a README row's submitter or batch driver actually runs
                        (the first job of it, as array task 0), derived from the submitter's
                        --dry-run by test_worker_row_is_what_its_caller_runs, so a flag renamed
                        on only one side of the hand-off fails here. Every cluster job runs
                        `replay-hmm <argv>` through slurm/run_replay_hmm.sbatch and takes
                        --n-jobs and its array index from the job's environment; the row writes
                        those as the explicit flags they stand for (`cc.slurm_job_argv`).
"""
from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from pathlib import Path

import pytest

import _cli_capture as cc

# --- shared README fragments (the README repeats these verbatim) -------------------------
_RATS_19 = (
    "--rats TinyTim TinyTim TinyTim TinyTim TinyTim TinyTim Lucky Lucky Lucky Lucky Lucky "
    "Ned Ned Ned Ned BB Spot Huey Stinky --sessions rec01 rec02 rec03 rec04 rec05 rec06 "
    "rec01 rec02 rec03 rec04 rec05 rec01 rec02 rec03 rec04 rec01 rec01 rec01 rec01 --pairwise"
)
_FIG3_DECODE = (
    "--pbe-filter-mode nonrun_nonwheel "
    "--speed-smooth-cutoff-hz none --pbe-spike-density-stdev 3.0 --pbe-baseline-scope session "
    "--pbe-min-duration-s 0.100 --pbe-merge-gap-s 0.005 --trajectory-min-duration-s 0.050 "
    "--trajectory-merge-gap-s 0.005 --trajectory-mass-threshold 0.60"
)
_FIG3_FIT = f"{_RATS_19} --bin-width-cm 3.5 {_FIG3_DECODE}"
_LT = "dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67"
_LT_3E = _LT  # the recovery sources are TinyTim rec04, Ned rec02, Lucky rec05 of the Fig 3 fit


def _recovery_submit(rat: str, session: str, n_shards: int) -> str:
    return (
        f"submit recovery --rats {rat} --sessions {session} --generator-random-sample "
        f"--random-sample-size 1000 --generator-num-shards {n_shards} --num-em-iters 50 "
        f"--run-tag {_LT_3E} "
        "--kernels sticky_SR sticky_SR_rev sticky_TCMSR sticky_TCMSR_sym continuous_EM "
        "movement_empirical momentum segment_uniform stationary fragmented "
        "--grid-generator-kernels sticky_SR sticky_SR_rev sticky_TCMSR sticky_TCMSR_sym "
        "continuous_EM momentum segment_uniform --random-fixed-generator-reps 5 "
        "--random-fixed-generator-kernels movement_empirical stationary fragmented "
        "--random-sample-seed 42"
    )
_LT_3MS = "dtbin_0p003__step_0p003__bwcm3p5__cfgf58e7e21b2__allpbe"
_BT = "dtbin_0p01__step_0p01__hmmbin_0p02"
_PT = "dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens"
_PT_3MS = "dtbin_0p003__step_0p003__hmmbin_4__evt_spikedens__allpbe"
_TWO_MAP_RECOVERY = (
    "--generator-random-sample --random-sample-size 1000 --num-em-iters 50 "
    "--random-fixed-generator-reps 10 --random-sample-seed 42 --generator-num-shards 40"
)
_S6A_SESSIONS = ("Achilles_10252013", "Cicero_09012014", "Gatsby_08022013")
_S6B_SESSIONS = ("Rat1/Linear1", "Rat1/Linear4", "Rat2/Linear1", "Rat5/Linear1")


def _recovery_rows(fits_root: str, tag: str, sessions: tuple[str, ...]) -> str:
    return " ".join(
        f"'results/{fits_root}/{s}/{tag}/synthetic_recovery/*of0040.csv'" for s in sessions
    )


_LUSTIG_TRAD_CSV = (
    "results/lustig_traditional_decode/bwcm3p5__dtbin_0p02__step_0p005__pmin_0p01__cells_4"
    "__corr_0p8__steps_8__2dbins_100__physarm/traditional_decode_events.csv"
)
_BUZSAKI_TRAD_CSV = (
    "results/buzsaki_traditional_decode/bin_0p04__dtbin_0p02__step_0p005__pmin_0p01__cells_4"
    "__corr_0p8__steps_8/traditional_decode_events.csv"
)
_PFEIFFER_TRAD_CSV = (
    "results/PfeifferFoster2015/traditional_decode/bin_4__dtbin_0p02__step_0p005__pmin_0p01"
    "__cells_4__corr_0p8__steps_8__src_spikedens/traditional_decode_events.csv"
)
_FIG5 = (
    f"--rat Huey --session rec01 --run-tag {_LT}__hardzero --kernel sticky_TCMSR "
    "--compare-kernel momentum continuous_EM fragmented --sort-by dll --session-backdrop none "
    "--event-index 8 --save-initial --no-show"
)
_VIEWER_KERNELS = "--kernel sticky_TCMSR --compare-kernel momentum continuous_EM fragmented --sort-by dll"
_RECOVERY_KERNELS = (
    "--kernels sticky_TCMSR sticky_TCMSR_sym sticky_SR sticky_SR_rev continuous_EM momentum "
    "movement_empirical fragmented stationary --grid-generator-kernels sticky_TCMSR "
    "sticky_TCMSR_sym sticky_SR sticky_SR_rev continuous_EM momentum "
    "--random-fixed-generator-kernels movement_empirical stationary fragmented"
)

# The first worker call `replay-hmm maps build --dataset lustig` makes for each prereq row: the
# per-session parser `maps.build.main`, called in-process with every flag forwarded explicitly
# and `--save` absolute under the repo root (written relative here).
_MAP_WORKER_MODULE = f"{cc.PACKAGE_PREFIX}replay_hmm.datasets.lustig.maps.build"
_MAP_WORKER = (
    "--linear-bin-mode absolute --linear-bin-width {w} --linear-n-bins 52 "
    "--save results/{rat}/rec01/lustig_maps_abs_bw{bw}.npz"
)
_CV_WORKER_DECODE = (
    "--bin-width-cm 3.5 "
    "--speed-smooth-cutoff-hz none --pbe-filter-mode nonrun_nonwheel --pbe-min-duration-s 0.1 "
    "--pbe-merge-gap-s 0.005 --pbe-spike-density-stdev 3.0 --pbe-baseline-scope session "
    "--trajectory-mass-threshold 0.6 --trajectory-min-duration-s 0.05 "
    "--trajectory-merge-gap-s 0.005"
)

# id -> (the "package:" module, argv as the README writes it after `replay-hmm`
#        [, cwd: "repo", the default]).
COMMANDS: dict[str, tuple[str, str] | tuple[str, str, str]] = {
    # Prerequisites
    "prereq.maps_bw8": (cc.PACKAGE_CLI,
        "maps build --dataset lustig "
        "--rats TinyTim TinyTim TinyTim TinyTim TinyTim TinyTim Lucky Lucky Lucky Lucky Lucky "
        "Ned Ned Ned Ned BB --sessions rec01 rec02 rec03 rec04 rec05 rec06 rec01 rec02 rec03 "
        "rec04 rec05 rec01 rec02 rec03 rec04 rec01 --pairwise --linear-bin-mode absolute "
        "--linear-bin-width 8 "
        "--save-template 'results/{rat}/{session}/lustig_maps_abs_bw8.npz'", "repo"),
    "prereq.maps_bw10": (cc.PACKAGE_CLI,
        "maps build --dataset lustig "
        "--rats Spot Huey Stinky --sessions rec01 --linear-bin-mode absolute --linear-bin-width 10 "
        "--save-template 'results/{rat}/{session}/lustig_maps_abs_bw10.npz'",
        "repo"),
    "prereq.lustig_trad_decode": (cc.PACKAGE_CLI,
        "traditional decode --dataset lustig --bin-width-cm 3.5 --n-jobs 6", "repo"),
    "prereq.buzsaki_trad_decode": (cc.PACKAGE_CLI,
        "traditional decode --dataset buzsaki "
        "--bin-width-m 0.04 --n-jobs 5", "repo"),
    "prereq.pfeiffer_trad_decode": (cc.PACKAGE_CLI,
        "traditional decode --dataset pfeiffer "
        "--event-source spike-density --bin-width-cm 4 --n-jobs 4", "repo"),
    # Cluster rows (prose)
    "cluster.local_fit": (cc.PACKAGE_CLI,
        f"fit --dataset lustig --rat TinyTim --session rec01 --bin-width-cm 3.5 {_FIG3_DECODE}",
        "repo"),
    "cluster.recovery_shard_check": (cc.PACKAGE_CLI,
        f"recovery check-shards --run-tag {_LT_3E}", "repo"),
    "cluster.recovery_shard_check_two_map": (cc.PACKAGE_CLI,
        "recovery check-shards --dataset buzsaki", "repo"),
    # Fig 2B, 2C, Table 1
    "fig2b.results": (cc.PACKAGE_CLI, "misspec gallery", "repo"),
    "fig2c.results": (cc.PACKAGE_CLI, "misspec gallery --all-sessions", "repo"),
    "fig2b.plot": (cc.PACKAGE_CLI, "misspec gallery --plot-only --event-ms 150 --example-rank 4",
        "repo"),
    "fig2c.plot": (cc.PACKAGE_CLI, "misspec gallery --all-sessions --plot-only", "repo"),
    "table1.sweep": (cc.PACKAGE_CLI, "misspec sweep --n-jobs 8", "repo"),
    "table1.plot": (cc.PACKAGE_CLI, "misspec sweep --plot-only", "repo"),
    "table1.submit": (cc.PACKAGE_CLI, "submit misspec-sweep", "repo"),
    # Fig 3A-D
    "fig3.fit": (cc.PACKAGE_CLI, f"submit fit --dataset lustig {_FIG3_FIT}", "repo"),
    "fig3a.jumps": (cc.PACKAGE_CLI, f"jumps --dataset lustig --run-tag {_LT} --n-jobs 4", "repo"),
    "fig3c.event_cache": (cc.PACKAGE_CLI, f"score-events --dataset lustig --run-tag {_LT}", "repo"),
    "fig3bc.plot": (cc.PACKAGE_CLI,
        f"plot cv-strip --dataset lustig --run-tag {_LT} --min-events 10", "repo"),
    "figS3.plot": (cc.PACKAGE_CLI, f"plot cv-strip --dataset lustig --run-tag {_LT}", "repo"),
    "fig3d.direction_cache": (cc.PACKAGE_CLI,
        f"score-events --dataset lustig --direction --run-tag {_LT}", "repo"),
    "fig3d.plot": (cc.PACKAGE_CLI,
        f"plot direction-share --dataset lustig --run-tag {_LT} --min-events 10 "
        f"--save-data results/direction_kernel_share_{_LT}_minev10.csv", "repo"),
    # Fig 3E, 3F, S5: one recovery run per session, pooled by the plot.
    "fig3e.recovery": (cc.PACKAGE_CLI, _recovery_submit("TinyTim", "rec04", 120), "repo"),
    "fig3e.recovery_ned": (cc.PACKAGE_CLI, _recovery_submit("Ned", "rec02", 240), "repo"),
    "fig3e.recovery_lucky": (cc.PACKAGE_CLI, _recovery_submit("Lucky", "rec05", 240), "repo"),
    "fig3e.plot": (cc.PACKAGE_CLI,
        f"recovery plot 'results/hmm_fits_full_em_mbtraj/TinyTim/rec04/{_LT_3E}/synthetic_recovery/*of0120.csv' "
        f"'results/hmm_fits_full_em_mbtraj/Ned/rec02/{_LT_3E}/synthetic_recovery/*of0240.csv' "
        f"'results/hmm_fits_full_em_mbtraj/Lucky/rec05/{_LT_3E}/synthetic_recovery/*of0240.csv' "
        f"--figures-root figures/recovery_pooled --run-tag {_LT_3E} --generator-random-sample "
        f"{_RECOVERY_KERNELS}", "repo"),
    # Fig 4 (4C-F is the fig3a.jumps call)
    "fig4a.trad_jumps": (cc.PACKAGE_CLI,
        f"traditional jumps --dataset lustig --events-csv {_LUSTIG_TRAD_CSV} --n-jobs 8",
        "repo"),
    "fig4b.mb_jump": (cc.PACKAGE_CLI,
        "mb-jump --dataset lustig --rats BB Huey Lucky Ned Spot Stinky TinyTim --bin-width-cm 3.5 "
        "--plot-pooled-jumps --continue-on-error "
        "--n-jobs 12", "repo"),
    # Fig 5 and the two-map event viewers
    "fig5.hardzero_fit": (cc.PACKAGE_CLI, f"submit fit --dataset lustig {_FIG3_FIT} --hard-zero",
        "repo"),
    "fig5.col1": (cc.PACKAGE_CLI, f"viewer --dataset lustig {_FIG5}", "repo"),
    "fig5.col2": (cc.PACKAGE_CLI,
        f"viewer --dataset lustig {_FIG5} --rat Lucky --session rec01 --event-index 0", "repo"),
    "fig5.col3": (cc.PACKAGE_CLI,
        f"viewer --dataset lustig {_FIG5} --rat TinyTim --session rec02 --event-index 30", "repo"),
    "fig5.col4": (cc.PACKAGE_CLI,
        f"viewer --dataset lustig {_FIG5} --rat Lucky --session rec01 --event-index 56", "repo"),
    "fig5.col5": (cc.PACKAGE_CLI,
        f"viewer --dataset lustig {_FIG5} --rat TinyTim --session rec02 --event-index 87", "repo"),
    "viewer.pfeiffer": (cc.PACKAGE_CLI,
        f"viewer --dataset pfeiffer --rat Rat1 --session Linear1 --run-tag {_PT} {_VIEWER_KERNELS}",
        "repo"),
    "viewer.buzsaki": (cc.PACKAGE_CLI,
        f"viewer --dataset buzsaki --session Achilles_10252013 --run-tag {_BT} {_VIEWER_KERNELS}",
        "repo"),
    # Tables S1-S3
    "tableS1.budget": (cc.PACKAGE_CLI,
        f"plot cv-budget --dataset lustig --run-tag {_LT} --save-data results/cv_data_budget_traj.csv",
        "repo"),
    "tableS2.budget": (cc.PACKAGE_CLI,
        f"plot cv-budget --dataset pfeiffer --run-tag {_PT} "
        "--save-data results/pfeiffer_cv_data_budget_traj.csv", "repo"),
    "tableS3.budget": (cc.PACKAGE_CLI,
        "plot cv-budget --dataset buzsaki "
        "--hmm-bin-width-cm 2.0 "
        "--save-data results/buzsaki_cv_data_budget_traj.csv", "repo"),
    "tableS1.budget_tradreplay": (cc.PACKAGE_CLI,
        f"plot cv-budget --dataset lustig --run-tag {_LT}__tradreplay "
        "--save-data results/cv_data_budget_tradreplay.csv", "repo"),
    "tableS2.budget_tradreplay": (cc.PACKAGE_CLI,
        f"plot cv-budget --dataset pfeiffer --run-tag {_PT}__tradreplay "
        "--save-data results/pfeiffer_cv_data_budget_tradreplay.csv", "repo"),
    "tableS3.budget_tradreplay": (cc.PACKAGE_CLI,
        f"plot cv-budget --dataset buzsaki --run-tag {_BT}__tradreplay "
        "--save-data results/buzsaki_cv_data_budget_tradreplay.csv", "repo"),
    "tables.make": (cc.PACKAGE_CLI, "tables cv-budget", "repo"),
    # Fig S2
    "figS2a.fit": (cc.PACKAGE_CLI,
        "fit --dataset buzsaki "
        "--hmm-bin-width-cm 2.0 --num-em-iters 200",
        "repo"),
    "figS2a.plot": (cc.PACKAGE_CLI,
        "plot cv-strip --dataset buzsaki "
        "--hmm-bin-width-cm 2.0", "repo"),
    "figS2c.fit": (cc.PACKAGE_CLI,
        f"fit --dataset buzsaki --traditional-events-csv {_BUZSAKI_TRAD_CSV} --hmm-bin-width-cm 2 "
        "--num-em-iters 200 --n-jobs 3", "repo"),
    "figS2c.plot": (cc.PACKAGE_CLI, f"plot cv-strip --dataset buzsaki --run-tag '{_BT}__tradreplay'",
        "repo"),
    "figS2e.fit": (cc.PACKAGE_CLI,
        "fit --dataset buzsaki --hmm-bin-width-cm 2.0 --decode-tbin-s 0.003 "
        "--decode-tbin-step-s 0.003 --num-em-iters 200 --all-pbes --n-jobs 4", "repo"),
    "figS2e.plot": (cc.PACKAGE_CLI,
        "plot cv-strip --dataset buzsaki "
        "--hmm-bin-width-cm 2.0 --decode-tbin-s 0.003 --decode-tbin-step-s 0.003 --all-pbes", "repo"),
    "figS2b.fit": (cc.PACKAGE_CLI,
        "fit --dataset pfeiffer --hmm-bin-width-cm 4.0 "
        "--num-em-iters 200 --event-detection-source spike-density "
        "--n-jobs 8", "repo"),
    "figS2b.score_events": (cc.PACKAGE_CLI,
        f"score-events --dataset pfeiffer --run-tag {_PT}", "repo"),
    "figS2d.score_events": (cc.PACKAGE_CLI,
        f"score-events --dataset pfeiffer --run-tag {_PT}__tradreplay "
        f"--traditional-events-csv {_PFEIFFER_TRAD_CSV}", "repo"),
    "figS2f.score_events": (cc.PACKAGE_CLI,
        f"score-events --dataset pfeiffer --run-tag {_PT_3MS}", "repo"),
    "figS2b.plot": (cc.PACKAGE_CLI, f"plot cv-strip --dataset pfeiffer --run-tag '{_PT}'", "repo"),
    "figS2d.fit": (cc.PACKAGE_CLI,
        "fit --dataset pfeiffer --hmm-bin-width-cm 4.0 "
        "--num-em-iters 200 --event-detection-source spike-density "
        f"--traditional-events-csv {_PFEIFFER_TRAD_CSV} --n-jobs 6", "repo"),
    "figS2d.plot": (cc.PACKAGE_CLI,
        f"plot cv-strip --dataset pfeiffer --run-tag '{_PT}__tradreplay'", "repo"),
    "figS2f.fit": (cc.PACKAGE_CLI,
        "fit --dataset pfeiffer --hmm-bin-width-cm 4.0 --decode-tbin-s 0.003 "
        "--decode-tbin-step-s 0.003 --num-em-iters 200 --event-detection-source spike-density "
        "--n-jobs 3 --all-pbes", "repo"),
    "figS2f.plot": (cc.PACKAGE_CLI, f"plot cv-strip --dataset pfeiffer --run-tag '{_PT_3MS}'",
        "repo"),
    # Fig S4
    "figS4a.fit": (cc.PACKAGE_CLI,
        f"submit fit --dataset lustig {_FIG3_FIT} --sigma-cap 3 "
        f"--traditional-events-csv {_LUSTIG_TRAD_CSV}", "repo"),
    "figS4a.plot": (cc.PACKAGE_CLI,
        f"plot cv-strip --dataset lustig --run-tag {_LT}__tradreplay --min-events 10", "repo"),
    "figS4b.cross_session": (cc.PACKAGE_CLI,
        f"cross-session-ll --run-tag {_LT} --kernels sticky_TCMSR continuous_EM momentum sticky_SR "
        "--min-test-events 10 --sessions TinyTim/rec01 TinyTim/rec02 TinyTim/rec03 TinyTim/rec04 "
        "TinyTim/rec05 TinyTim/rec06 Lucky/rec01 Lucky/rec02 Lucky/rec03 Lucky/rec04 Lucky/rec05 "
        "Ned/rec01 Ned/rec02 Ned/rec03 Ned/rec04 "
        "--save-data results/cross_session_transfer/cross_session_ll_traj_pairs.csv", "repo"),
    "figS4b.plot": (cc.PACKAGE_CLI,
        f"plot kernel-contrast --run-tag {_LT} --min-events 10 --rivals continuous_EM momentum "
        "--results-root results/cross_session_transfer "
        "--summary-name cross_session_within_rat_summary.csv --out-suffix _transfer_momentum",
        "repo"),
    "figS4c.fit": (cc.PACKAGE_CLI,
        f"submit fit --dataset lustig {_FIG3_FIT} --decode-tbin-s 0.003 --decode-tbin-step-s 0.003 "
        "--all-pbes", "repo"),
    "figS4c.plot": (cc.PACKAGE_CLI, f"plot cv-strip --dataset lustig --run-tag {_LT_3MS}", "repo"),
    "figS4d.fit": (cc.PACKAGE_CLI, f"submit fit --dataset lustig {_FIG3_FIT} --center-arm-only",
        "repo"),
    "figS4d.plot": (cc.PACKAGE_CLI,
        f"plot cv-strip --dataset lustig --run-tag {_LT}__centerarm --min-events 10", "repo"),
    # Fig S6: one recovery run per session, pooled by the plot.
    "figS6a.recovery": (cc.PACKAGE_CLI,
        "submit recovery --dataset buzsaki --sessions Achilles_10252013 Cicero_09012014 "
        f"Gatsby_08022013 --run-tag {_BT} {_TWO_MAP_RECOVERY}", "repo"),
    "figS6a.recovery_plot": (cc.PACKAGE_CLI,
        f"recovery plot {_recovery_rows('buzsaki_hmm_fits', _BT, _S6A_SESSIONS)} "
        "--dataset buzsaki --figures-root figures/buzsaki/buzsaki_hmm_fits_full_em/recovery_pooled "
        f"--run-tag {_BT} --generator-random-sample {_RECOVERY_KERNELS}", "repo"),
    "figS6b.recovery": (cc.PACKAGE_CLI,
        "submit recovery --dataset pfeiffer --rats Rat1 Rat2 Rat5 --sessions Linear1 Linear4 "
        f"--run-tag {_PT} {_TWO_MAP_RECOVERY}", "repo"),
    "figS6b.recovery_plot": (cc.PACKAGE_CLI,
        f"recovery plot {_recovery_rows('PfeifferFoster2015/hmm_fits', _PT, _S6B_SESSIONS)} "
        "--dataset pfeiffer "
        "--figures-root figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/recovery_pooled "
        f"--run-tag {_PT} --generator-random-sample {_RECOVERY_KERNELS}", "repo"),
    # Fig S8 (+ the documented --plot-only re-render)
    "figS8a.ppc": (cc.PACKAGE_CLI,
        f"ppc --dataset lustig --run-tag {_LT} --qq-include-zero --qq-symlog", "repo"),
    "figS8b.ppc": (cc.PACKAGE_CLI,
        f"ppc --dataset buzsaki --run-tag {_BT} --qq-include-zero --qq-symlog", "repo"),
    "figS8c.ppc": (cc.PACKAGE_CLI,
        f"ppc --dataset pfeiffer --run-tag {_PT} --qq-include-zero --qq-symlog", "repo"),
    "figS8a.plot_only": (cc.PACKAGE_CLI,
        f"ppc --dataset lustig --run-tag {_LT} --qq-include-zero --qq-symlog --plot-only", "repo"),
    "figS8b.plot_only": (cc.PACKAGE_CLI,
        f"ppc --dataset buzsaki --run-tag {_BT} --qq-include-zero --qq-symlog --plot-only", "repo"),
    "figS8c.plot_only": (cc.PACKAGE_CLI,
        f"ppc --dataset pfeiffer --run-tag {_PT} --qq-include-zero --qq-symlog --plot-only", "repo"),
    # Fig S9, S7 (S9A is the fig3a.jumps call)
    "figS9b.jumps": (cc.PACKAGE_CLI,
        "jumps --dataset buzsaki "
        "--hmm-bin-width-cm 2 --exclude-map-edge-bins 2 --n-jobs 5 --hover-thresh-bins 1", "repo"),
    "figS7ce.jumps": (cc.PACKAGE_CLI,
        "jumps --dataset buzsaki "
        "--hmm-bin-width-cm 2 --n-jobs 5", "repo"),
    "figS9c_S7df.jumps": (cc.PACKAGE_CLI,
        "jumps --dataset pfeiffer --hmm-bin-width-cm 4.0 "
        "--event-detection-source spike-density --n-jobs 4", "repo"),
    "figS7a.mb_jump": (cc.PACKAGE_CLI, "mb-jump --dataset buzsaki --hover-thresh-bins 0", "repo"),
    "figS7b.mb_jump": (cc.PACKAGE_CLI,
        "mb-jump --dataset pfeiffer --event-source spike-density --bin-width-cm 4", "repo"),
    # Defaults: no arguments, or only the flags argparse requires. Each id names the command it
    # parses by a stable short name.
    "defaults.maps_build_lustig": (cc.PACKAGE_CLI,
        "maps build --dataset lustig --rats TinyTim --sessions rec01", "repo"),
    "defaults.traditional_decode_lustig": (cc.PACKAGE_CLI, "traditional decode --dataset lustig",
        "repo"),
    "defaults.traditional_decode_buzsaki": (cc.PACKAGE_CLI,
        "traditional decode --dataset buzsaki", "repo"),
    "defaults.traditional_decode_pfeiffer": (cc.PACKAGE_CLI,
        "traditional decode --dataset pfeiffer", "repo"),
    "defaults.misspec_gallery": (cc.PACKAGE_CLI, "misspec gallery", "repo"),
    "defaults.misspec_sweep": (cc.PACKAGE_CLI, "misspec sweep", "repo"),
    "defaults.submit_fit_lustig": (cc.PACKAGE_CLI, "submit fit --dataset lustig", "repo"),
    "defaults.fit_lustig": (cc.PACKAGE_CLI, "fit --dataset lustig", "repo"),
    "defaults.jumps_lustig": (cc.PACKAGE_CLI, "jumps --dataset lustig", "repo"),
    "defaults.score_events_lustig": (cc.PACKAGE_CLI, "score-events --dataset lustig",
        "repo"),
    "defaults.plot_cv_strip_lustig": (cc.PACKAGE_CLI, "plot cv-strip --dataset lustig",
        "repo"),
    "defaults.plot_direction_share_lustig": (cc.PACKAGE_CLI,
        f"plot direction-share --dataset lustig --run-tag {_LT}", "repo"),
    "defaults.submit_recovery": (cc.PACKAGE_CLI, "submit recovery", "repo"),
    "defaults.recovery_plot": (cc.PACKAGE_CLI, "recovery plot", "repo"),
    "defaults.recovery_check_shards": (cc.PACKAGE_CLI, "recovery check-shards", "repo"),
    "defaults.traditional_jumps_lustig": (cc.PACKAGE_CLI,
        f"traditional jumps --dataset lustig --events-csv {_LUSTIG_TRAD_CSV}", "repo"),
    "defaults.mb_jump_lustig": (cc.PACKAGE_CLI, "mb-jump --dataset lustig --rats TinyTim", "repo"),
    "defaults.viewer_lustig": (cc.PACKAGE_CLI, "viewer --dataset lustig", "repo"),
    "defaults.viewer_pfeiffer": (cc.PACKAGE_CLI,
        f"viewer --dataset pfeiffer --rat Rat1 --session Linear1 --run-tag {_PT}", "repo"),
    "defaults.viewer_buzsaki": (cc.PACKAGE_CLI,
        f"viewer --dataset buzsaki --session Achilles_10252013 --run-tag {_BT}", "repo"),
    "defaults.plot_cv_budget_lustig": (cc.PACKAGE_CLI,
        f"plot cv-budget --dataset lustig --run-tag {_LT}", "repo"),
    "defaults.plot_cv_budget_pfeiffer": (cc.PACKAGE_CLI,
        f"plot cv-budget --dataset pfeiffer --run-tag {_PT}", "repo"),
    "defaults.plot_cv_budget_buzsaki": (cc.PACKAGE_CLI, "plot cv-budget --dataset buzsaki",
        "repo"),
    "defaults.tables_cv_budget": (cc.PACKAGE_CLI, "tables cv-budget", "repo"),
    "defaults.fit_buzsaki": (cc.PACKAGE_CLI, "fit --dataset buzsaki", "repo"),
    "defaults.plot_cv_strip_buzsaki": (cc.PACKAGE_CLI, "plot cv-strip --dataset buzsaki",
        "repo"),
    "defaults.fit_pfeiffer": (cc.PACKAGE_CLI, "fit --dataset pfeiffer", "repo"),
    "defaults.jumps_pfeiffer": (cc.PACKAGE_CLI, "jumps --dataset pfeiffer",
        "repo"),
    "defaults.plot_cv_strip_pfeiffer": (cc.PACKAGE_CLI, "plot cv-strip --dataset pfeiffer",
        "repo"),
    "defaults.cross_session_ll": (cc.PACKAGE_CLI, f"cross-session-ll --run-tag {_LT}",
        "repo"),
    "defaults.plot_kernel_contrast": (cc.PACKAGE_CLI,
        f"plot kernel-contrast --run-tag {_LT}", "repo"),
    "defaults.ppc_lustig": (cc.PACKAGE_CLI, f"ppc --dataset lustig --run-tag {_LT}",
        "repo"),
    "defaults.ppc_buzsaki": (cc.PACKAGE_CLI, f"ppc --dataset buzsaki --run-tag {_BT}",
        "repo"),
    "defaults.ppc_pfeiffer": (cc.PACKAGE_CLI, f"ppc --dataset pfeiffer --run-tag {_PT}",
        "repo"),
    "defaults.jumps_buzsaki": (cc.PACKAGE_CLI, "jumps --dataset buzsaki", "repo"),
    "defaults.mb_jump_buzsaki": (cc.PACKAGE_CLI, "mb-jump --dataset buzsaki", "repo"),
    "defaults.mb_jump_pfeiffer": (cc.PACKAGE_CLI, "mb-jump --dataset pfeiffer", "repo"),
    # Workers the README reaches only through a caller (their defaults). `replay-hmm recovery
    # run` requires a generator mode, so the row passes the 3E one.
    "defaults.recovery_run": (cc.PACKAGE_CLI,
        "recovery run --generator-random-sample"),
    "defaults.maps_worker_lustig": (_MAP_WORKER_MODULE, "", "repo"),
    # What the callers run. Every cluster job is `replay-hmm <worker argv>` from the repo root
    # (slurm/run_replay_hmm.sbatch), with --n-jobs = the job's -c and array task 0's shard /
    # session written in as the flags `--slurm-array` stands for (`cc.slurm_job_argv`).
    # `maps build` calls its per-session worker in-process.
    "worker.fig3.fit": (cc.PACKAGE_CLI,
        f"fit --dataset lustig --rat TinyTim --session rec01 {_CV_WORKER_DECODE} --n-jobs 3",
        "repo"),
    "worker.fig3e.recovery": (cc.PACKAGE_CLI,
        "recovery run --rat TinyTim --session rec04 --num-em-iters 50 "
        "--random-sample-size 1000 --random-sample-seed 42 "
        f"--random-fixed-generator-reps 5 --generator-num-shards 120 --run-tag {_LT_3E} "
        "--kernels sticky_SR sticky_SR_rev sticky_TCMSR sticky_TCMSR_sym continuous_EM "
        "movement_empirical momentum segment_uniform stationary fragmented "
        "--grid-generator-kernels sticky_SR sticky_SR_rev sticky_TCMSR sticky_TCMSR_sym "
        "continuous_EM momentum segment_uniform "
        "--random-fixed-generator-kernels movement_empirical stationary fragmented "
        "--generator-random-sample --generator-shard-index 0 --n-jobs 30", "repo"),
    "worker.figS6a.recovery": (cc.PACKAGE_CLI,
        "recovery run --rat Achilles --session Achilles_10252013 --dataset buzsaki "
        "--num-em-iters 50 --random-sample-size 1000 --random-sample-seed 42 "
        f"--random-fixed-generator-reps 10 --generator-num-shards 40 --run-tag {_BT} "
        "--generator-random-sample --generator-shard-index 0 --n-jobs 30", "repo"),
    "worker.figS6b.recovery": (cc.PACKAGE_CLI,
        "recovery run --rat Rat1 --session Linear1 --dataset pfeiffer "
        "--num-em-iters 50 --random-sample-size 1000 --random-sample-seed 42 "
        f"--random-fixed-generator-reps 10 --generator-num-shards 40 --run-tag {_PT} "
        "--generator-random-sample --generator-shard-index 0 --n-jobs 30", "repo"),
    # The encoder records the `rat/session` value as a path (it contains a separator); it is
    # not one, but the comparison is the same either way.
    "worker.table1.sweep": (cc.PACKAGE_CLI,
        "misspec sweep --sessions BB/rec01 --skip-summary --n-jobs 16", "repo"),
    "worker.prereq.maps_bw8": (_MAP_WORKER_MODULE,
        "--rat TinyTim --session rec01 " + _MAP_WORKER.format(w="8.0", rat="TinyTim", bw=8),
        "repo"),
    "worker.prereq.maps_bw10": (_MAP_WORKER_MODULE,
        "--rat Spot --session rec01 " + _MAP_WORKER.format(w="10.0", rat="Spot", bw=10),
        "repo"),
}

_NON_README = ("defaults.", "worker.")


def command_case(cmd_id: str) -> tuple[str, list[str], str]:
    """(module, argv, cwd) of a COMMANDS row."""
    module, argv, *rest = COMMANDS[cmd_id]
    return module, shlex.split(argv), (rest[0] if rest else "repo")


def command_argv(cmd_id: str) -> tuple[str, list[str]]:
    module, argv, _ = command_case(cmd_id)
    return module, argv


def _strip(result: dict) -> dict:
    # A rejected parse is pinned by its exit code only: the usage text names the program
    # (sys.argv[0]), which depends on how the capture invokes it.
    return result if result["ok"] else {"ok": False, "exit_code": result.get("exit_code")}


def capture_all() -> dict[str, dict]:
    cases = {cid: command_case(cid) for cid in COMMANDS}
    results = cc.capture_cases(cases.values())
    return {cid: _strip(results[cc.case_key(case)]) for cid, case in cases.items()}


@pytest.fixture(scope="module")
def captured() -> dict[str, dict]:
    return capture_all()


def test_table_covers_every_readme_command():
    readme, problems = cc.readme_commands()
    assert problems == []
    from_readme = {(c.module, c.argv, c.cwd) for c in readme}
    from_table = {
        cc.case_key(command_case(cid)) for cid in COMMANDS if not cid.startswith(_NON_README)
    }
    assert from_readme - from_table == set(), "README command missing from COMMANDS"
    assert from_table - from_readme == set(), "COMMANDS row no longer in the README"


def test_defaults_cover_every_readme_script_and_worker():
    readme, _ = cc.readme_commands()
    workers = {COMMANDS[cid][0] for cid in COMMANDS if cid.startswith("worker.")}
    assert {c.module for c in readme} | workers == {
        COMMANDS[cid][0] for cid in COMMANDS if cid.startswith("defaults.")
    }


def test_lustig_decode_grid_defaults_are_pinned(captured):
    # These three parsers (the Lustig kernel-evidence scorer, the Lustig jump distribution,
    # recovery run) default to the canonical 10 ms / 10 ms grid (the Lustig HMM_DECODE_GRID);
    # the jump distribution also takes the trajectory mass threshold.
    for cid in ("defaults.score_events_lustig", "defaults.jumps_lustig",
                "defaults.recovery_run"):
        ns = captured[cid]["namespace"]
        assert ns["decode_tbin_s"] == {"type": "float", "repr": "0.01"}, cid
        assert ns["decode_tbin_step_s"] == {"type": "float", "repr": "0.01"}, cid
    jumps = captured["defaults.jumps_lustig"]["namespace"]
    assert jumps["trajectory_mass_threshold"] == {"type": "float", "repr": "0.6"}


def test_every_readme_command_parses(captured):
    bad = [cid for cid, result in captured.items()
           if not cid.startswith("defaults.") and not result["ok"]]
    assert bad == []


# ---------------------------------------------------------------- worker rows

def _submitted_worker(cmd_id: str) -> tuple[str, list[str]]:
    """The first job a `replay-hmm submit` row's --dry-run prints, as its array task 0 runs it.

    The job script execs `replay-hmm` with every argument after it in the sbatch call
    (test_the_job_script_runs_its_arguments); `cc.slurm_job_argv` writes in what the job's
    environment adds (-c as --n-jobs, the task index as the flag --slurm-array stands for).
    """
    module, argv, cwd = command_case(cmd_id)
    out = cc.run_main(module, [*argv, "--dry-run"], cwd=cwd)
    first = next(line for line in out.splitlines() if " :: " in line)
    cmd = shlex.split(first.split(" :: ", 1)[1])
    assert cmd[0] == "sbatch", cmd
    at = cmd.index(str(cc.SBATCH_JOB_SCRIPT))
    options, worker = cc.sbatch_option_values(cmd[1:at]), cmd[at + 1:]
    return cc.PACKAGE_CLI, cc.slurm_job_argv(worker, cc.slurm_job_env(options, task=0))


def _map_batch_worker(cmd_id: str) -> tuple[str, list[str]]:
    """The argv of the first worker call `maps build` makes, repo paths made relative.

    Stopped at the worker itself (`maps.build.main`, called in-process by the batch), so the
    row is what really runs, not what the batch meant to pass.
    """
    from replay_hmm.datasets.lustig.maps import build

    module, argv, cwd = command_case(cmd_id)
    call = cc.run_main_until(module, argv, (build, "main"), cwd=cwd, call_through=False)
    (worker_argv,) = call.args
    # Absolute paths under the repo root become repo-relative: the worker's cwd is the root.
    rel = [Path(t).relative_to(cc.REPO_ROOT).as_posix() if t.startswith(f"{cc.REPO_ROOT}/") else t
           for t in worker_argv]
    return _MAP_WORKER_MODULE, rel


# worker id -> how its argv is derived from today's caller.
WORKER_SOURCES: dict[str, Callable[[], tuple[str, list[str]]]] = {
    "worker.fig3.fit": lambda: _submitted_worker("fig3.fit"),
    "worker.fig3e.recovery": lambda: _submitted_worker("fig3e.recovery"),
    "worker.figS6a.recovery": lambda: _submitted_worker("figS6a.recovery"),
    "worker.figS6b.recovery": lambda: _submitted_worker("figS6b.recovery"),
    "worker.table1.sweep": lambda: _submitted_worker("table1.submit"),
    "worker.prereq.maps_bw8": lambda: _map_batch_worker("prereq.maps_bw8"),
    "worker.prereq.maps_bw10": lambda: _map_batch_worker("prereq.maps_bw10"),
}


def test_every_worker_row_has_a_source():
    assert set(WORKER_SOURCES) == {cid for cid in COMMANDS if cid.startswith("worker.")}


@pytest.mark.parametrize("cmd_id", list(WORKER_SOURCES))
def test_worker_row_is_what_its_caller_runs(cmd_id):
    module, argv = command_argv(cmd_id)
    assert WORKER_SOURCES[cmd_id]() == (module, argv)


def test_the_job_script_runs_its_arguments():
    # The premise of the worker rows: the one job script execs `replay-hmm` with exactly its
    # own arguments, from $REPLAY_HMM_ROOT, and adds none.
    text = cc.SBATCH_JOB_SCRIPT.read_text()
    assert 'ARGS=("$@")' in text
    assert 'cd "$REPLAY_HMM_ROOT"' in text
    calls = [ln.strip() for ln in text.splitlines()
             if ln.strip().startswith(("exec ", "replay-hmm ", "python "))]
    assert calls == ['exec replay-hmm "${ARGS[@]+"${ARGS[@]}"}"'], calls


def test_readme_flags_direct_sbatch_calls():
    _, problems = cc.readme_commands("| X | `sbatch some_new_wrapper.sh` | y |\n")
    assert problems and "some_new_wrapper.sh" in problems[0]

