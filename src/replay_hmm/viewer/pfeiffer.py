"""Interactive per-event browser for Pfeiffer & Foster HMM decodes.

The Lustig viewer's panels, controls, fixed-ceiling heatmaps and multi-kernel ΔLL ranking, on
Pfeiffer's linear tracks. Each session's events are rebuilt with the fit's own detection and
decode, on the rate maps and cell order the fit saved. Works on MB-trajectory, all-PBE and
traditional-replay (__tradreplay, `--jump-scope traditional`) fits.

Run from the repo root with the tcmsr env:
    replay-hmm viewer --dataset pfeiffer --rat Rat1 --session Linear1 \\
        --run-tag dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens \\
        --kernel sticky_TCMSR --compare-kernel continuous_EM fragmented --sort-by dll
    replay-hmm viewer --dataset pfeiffer --rat Rat1 --session Linear1 \\
        --run-tag dtbin_0p003__step_0p003__hmmbin_4__evt_spikedens__allpbe --jump-scope full-pbe
    replay-hmm viewer --dataset pfeiffer --rat Rat1 --session Linear1 \\
        --run-tag dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens__tradreplay --jump-scope traditional

Figures go to figures/PfeifferFoster2015/pfeiffer_hmm_fits_full_em/<rat>/<session>/<tag>/events/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from replay_hmm import paths
from replay_hmm.datasets.pfeiffer.constants import DEFAULT_FIG_ROOT, DEFAULT_RESULTS_ROOT
from replay_hmm.viewer import two_map as tme
from replay_hmm.io import names

# The fit's own roots.
RESULTS_ROOT = DEFAULT_RESULTS_ROOT
FIGURES_ROOT = DEFAULT_FIG_ROOT


def add_session_args(parser) -> None:
    parser.add_argument("--rat", required=True, help="e.g. Rat1")
    parser.add_argument("--session", required=True, help="e.g. Linear1")


def load_traditional_rows(csv_path, statuses, rat: str, session: str, summary: dict):
    """This session's rows of a traditional events table, as the fit reads them
    (`datasets.pfeiffer.rebuild.traditional_rows`): verdict-filtered, in session_pbe_index
    order, and refused when the table was classified on a different event detection source
    than the fit's."""
    from replay_hmm.datasets.pfeiffer import rebuild as pfeiffer_rebuild

    return pfeiffer_rebuild.traditional_rows(csv_path, rat, session, summary, statuses=statuses)


def load_session(rat: str, session: str, summary: dict, inputs: dict, label: str, *,
                 traditional_rows=None) -> tme.TwoMapSession:
    """Every candidate event, detected and decoded as the fit CLI did on the maps and cell order
    the fit saved (`datasets.pfeiffer.rebuild.rebuild_events`, unrestricted), with the session
    data and detector diagnostics the panels draw. `tme.check_event_set` and the stored-loglik
    guard hold the events to the fit.
    """
    from replay_hmm.datasets.pfeiffer import rebuild as pfeiffer_rebuild

    rebuilt = pfeiffer_rebuild.rebuild_events(
        rat, session, summary, inputs, label=label, traditional_rows=traditional_rows,
        restrict_to_accepted=False,
    )
    setting, data, diag = rebuilt.setting, rebuilt.data, rebuilt.detection_diag

    out = []
    for dec in rebuilt.decoded:
        out.append(tme.TwoMapEvent(
            index=int(dec["event"]["session_pbe_index"]),
            starts_s=np.asarray(dec["bin_starts_s"], dtype=float),
            counts=dec["spike_traj"].T.astype(np.int32),
            # datasets.two_map_event_sets.collect_mb_traj_datas's order under the Pfeiffer view:
            # outbound, then inbound.
            segments=[(int(m), int(s), int(e)) for m, segs in dec["segments"].items()
                      for s, e in segs if e - s >= 2],
        ))
    return tme.TwoMapSession(
        events=out, spike_times=rebuilt.spike_times, cell_ids=rebuilt.cell_ids,
        density_grids=[(g["centers"], g["density"], g["eligible"]) for g in diag.get("grids", [])],
        baseline_mean=float(diag.get("baseline_mean", np.nan)),
        baseline_sd=float(diag.get("baseline_sd", np.nan)),
        z_thresholds=(float(setting.pbe_peak_z), float(setting.pbe_end_z)),
        pos_ts=np.asarray(data["pos_ts"], dtype=float),
        pos_cm=np.asarray(data["pos_lin_cm"], dtype=float),
        bin_edges_cm=np.asarray(inputs["bin_edges"], dtype=float),     # cm already
    )


def signed_jumps(z, bounds):
    from replay_hmm.jumps import signed

    return signed.two_map_signed_jumps(z, bounds)


DATASET = tme.TwoMapDataset(
    name="pfeiffer",
    inputs_name=names.PFEIFFER_INPUTS_NAME,
    default_results_root=RESULTS_ROOT,
    default_figures_root=FIGURES_ROOT,
    add_session_args=add_session_args,
    session_key=lambda args: (args.rat, args.session),
    session_rel=lambda rat, session: Path(rat) / session,
    display_names=lambda rat, session: (f"{rat}/{session}", f"{rat} / {session}"),
    load_session=load_session,
    signed_jumps=signed_jumps,
    backdrop_key=lambda kernel, rat, session: f"{kernel}__{rat}__{session}",
    description=__doc__,
    traditional_dir=paths.PFEIFFER_TRADITIONAL_DECODE_DIR,
    load_traditional_rows=load_traditional_rows,
)


def main(argv=None) -> int:
    return tme.main(argv, dataset=DATASET)

