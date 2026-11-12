"""Interactive per-event browser for Grosmark & Buzsaki HMM decodes.

The Lustig viewer's panels, controls, fixed-ceiling heatmaps and multi-kernel ΔLL ranking, on
Buzsaki's linear tracks. Each session's events are rebuilt with the fit's own detection and
decode, on the rate maps and cell order the fit saved. Works on MB-trajectory, all-PBE and
traditional-replay (__tradreplay, `--jump-scope traditional`) fits.

The saved backdrop is the default jump run's (no __edgeN / __hoverN): this viewer's jumps use
`jumps.signed.two_map_signed_jumps`'s defaults, and a filtered backdrop would not match them.

Run from the repo root with the tcmsr env:
    replay-hmm viewer --dataset buzsaki --session Achilles_10252013 \\
        --run-tag dtbin_0p01__step_0p01__hmmbin_0p02 \\
        --kernel sticky_TCMSR --compare-kernel continuous_EM fragmented --sort-by dll
    replay-hmm viewer --dataset buzsaki --session Achilles_10252013 \\
        --run-tag dtbin_0p003__step_0p003__hmmbin_0p02__allpbe --jump-scope full-pbe
    replay-hmm viewer --dataset buzsaki --session Achilles_10252013 \\
        --run-tag dtbin_0p01__step_0p01__hmmbin_0p02__tradreplay --jump-scope traditional

Figures go to figures/buzsaki/buzsaki_hmm_fits_full_em/<session>/<tag>/events/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from replay_hmm import paths
from replay_hmm.datasets import registry
from replay_hmm.datasets.buzsaki.constants import DEFAULT_FIG_ROOT, DEFAULT_RESULTS_ROOT
from replay_hmm.datasets.buzsaki.session import rat_of_session
from replay_hmm.viewer import two_map as tme
from replay_hmm.io import names

# The fit's own roots.
RESULTS_ROOT = DEFAULT_RESULTS_ROOT
FIGURES_ROOT = DEFAULT_FIG_ROOT


def add_session_args(parser) -> None:
    parser.add_argument("--session", required=True, help="e.g. Achilles_10252013")


def load_traditional_rows(csv_path, statuses, rat: str, session: str, summary: dict):
    """This session's rows of a traditional events table, as the fit reads them
    (`datasets.buzsaki.rebuild.traditional_rows`): verdict-filtered, in session_pbe_index order."""
    from replay_hmm.datasets.buzsaki import rebuild as buzsaki_rebuild

    return buzsaki_rebuild.traditional_rows(csv_path, rat, session, summary, statuses=statuses)


def load_session(rat: str, session: str, summary: dict, inputs: dict, label: str, *,
                 traditional_rows=None) -> tme.TwoMapSession:
    """Every candidate event, detected and decoded as the fit CLI did on the maps and cell order
    the fit saved (`datasets.buzsaki.rebuild.rebuild_events`, unrestricted), with the session
    data and detector diagnostics the panels draw. `tme.check_event_set` and the stored-loglik
    guard hold the events to the fit.
    """
    from replay_hmm.datasets.buzsaki import kinematics as buzsaki_kinematics
    from replay_hmm.datasets.buzsaki import rebuild as buzsaki_rebuild

    linear = dict(registry.BUZSAKI_LINEAR_SESSIONS)
    if session not in linear:
        raise SystemExit(f"Unknown linear session {session!r}; known: {sorted(linear)}")
    rebuilt = buzsaki_rebuild.rebuild_events(
        rat, session, summary, inputs, label=label, traditional_rows=traditional_rows,
        restrict_to_accepted=False,
    )
    setting, data, diag = rebuilt.setting, rebuilt.data, rebuilt.detection_diag
    grid_tbin, grid_step = float(summary["decode_tbin_s"]), float(summary["decode_tbin_step_s"])

    out = []
    for dec in rebuilt.decoded:
        out.append(tme.TwoMapEvent(
            index=dec["session_pbe_index"],
            # decode_pbe's own tiling; it returns only the centres. TwoMapSource re-counts these
            # windows against rate_traj, so a drift between the two fails loudly.
            starts_s=np.arange(dec["start_s"], dec["end_s"] - grid_tbin + 1e-9, grid_step),
            counts=dec["rate_traj"].T.astype(np.int32),
            segments=[(0 if t["dir"] == "out" else 1, t["win_slice"].start, t["win_slice"].stop)
                      for t in dec["trajectories"] if t["win_slice"].stop - t["win_slice"].start >= 2],
        ))

    # OneDLocation is NaN for 68-93% of MAZE samples, the immobile track ends included, so a
    # strict reading leaves almost every PBE without a position. Filled with the detector's own
    # policy (datasets.buzsaki.kinematics.fill_position, hold_last by default), inside the MAZE
    # epoch only: a POST-epoch ripple event is in the sleep box, not at the last track
    # position. The info block says when a window's position is a fill.
    pos_ts = np.asarray(data["pos_ts"], dtype=float)
    pos_m = np.asarray(data["pos_1d"], dtype=float)
    in_maze = (pos_ts >= float(data["maze_epoch"][0])) & (pos_ts <= float(data["maze_epoch"][1]))
    pos_filled = np.full_like(pos_m, np.nan)
    pos_filled[in_maze] = buzsaki_kinematics.fill_position(
        pos_m[in_maze], setting.arm_end_position_fill
    )
    return tme.TwoMapSession(
        events=out, spike_times=rebuilt.spike_times, cell_ids=rebuilt.cell_ids,
        density_grids=[(diag["pfr_centers"], diag["pfr"], diag["eligible"])] if "pfr" in diag else [],
        baseline_mean=float(diag.get("baseline_mean", np.nan)),
        baseline_sd=float(diag.get("baseline_sd", np.nan)),
        z_thresholds=(float(setting.peak_z), float(setting.end_z)),
        pos_ts=pos_ts,
        pos_cm=100.0 * pos_filled,
        bin_edges_cm=100.0 * np.asarray(inputs["bin_edges"], dtype=float),   # saved in metres
        pos_observed=np.isfinite(pos_m),
        pos_fill=str(setting.arm_end_position_fill),
    )


def signed_jumps(z, bounds):
    from replay_hmm.jumps import signed

    return signed.two_map_signed_jumps(z, bounds)


DATASET = tme.TwoMapDataset(
    name="buzsaki",
    inputs_name=names.BUZSAKI_INPUTS_NAME,
    default_results_root=RESULTS_ROOT,
    default_figures_root=FIGURES_ROOT,
    add_session_args=add_session_args,
    # The fit tree has no rat level (<session>/<tag>/); the animal is the session name's prefix.
    session_key=lambda args: (rat_of_session(args.session), args.session),
    session_rel=lambda rat, session: Path(session),
    display_names=lambda rat, session: (session, session),
    load_session=load_session,
    signed_jumps=signed_jumps,
    backdrop_key=lambda kernel, rat, session: f"{kernel}__{session}",
    description=__doc__,
    traditional_dir=paths.BUZSAKI_TRADITIONAL_DECODE_DIR,
    load_traditional_rows=load_traditional_rows,
)


def main(argv=None) -> int:
    return tme.main(argv, dataset=DATASET)

