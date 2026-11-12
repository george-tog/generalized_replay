"""Traditional-criteria event tables: reading them, and placing their events on a bundle's grid."""
from pathlib import Path

import numpy as np
import pandas as pd

from replay_hmm.decode.grid import assert_contiguous_selection, decode_window_grid, select_event_bins

# --------------------------------------------------------------------------------------
# Externally supplied event sets (traditional replay criteria)
# --------------------------------------------------------------------------------------

# The columns any traditional-criteria events table must carry to gate decode bins. Same
# contract datasets.lustig.traditional_jumps uses.
TRADITIONAL_EVENT_COLUMNS = (
    "rat",
    "session",
    "trial",
    "event_start_lfp",
    "event_end_lfp",
    "status",
)


def load_traditional_events(
    events_csv,
    *,
    statuses=("accepted",),
    rats=None,
    sessions=None,
) -> pd.DataFrame:
    """Events called by `datasets.lustig.traditional`, filtered by verdict.

    Only the identity and time columns are read: this loader exists for analyses that take
    the traditional criteria's EVENT SET and nothing else, rebuilding every downstream
    quantity on their own decode grid.

    - Wheel winners are kept: an analysis whose state space contains the wheel map must not
      lose them.
    - No `__physarm` check: nothing here reads an arm label, so an L<->R-transposed CSV gives
      the same event set. Record which CSV was used instead.
    """
    events_csv = Path(events_csv)
    if not events_csv.exists():
        raise FileNotFoundError(f"Traditional events CSV not found: {events_csv}")
    df = pd.read_csv(events_csv)
    missing = set(TRADITIONAL_EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{events_csv} is missing required columns: {sorted(missing)}")

    statuses = tuple(str(s) for s in statuses)
    unknown = set(statuses) - set(df["status"].astype(str).unique())
    if unknown:
        raise ValueError(
            f"{events_csv} contains no rows with status {sorted(unknown)}; "
            f"present statuses are {sorted(df['status'].astype(str).unique())}."
        )
    out = df[df["status"].astype(str).isin(statuses)].copy()

    for column in ("rat", "session", "status"):
        out[column] = out[column].astype(str)
    for column in ("trial", "event_start_lfp", "event_end_lfp"):
        out[column] = out[column].astype(int)

    if rats is not None:
        out = out[out["rat"].isin([str(r) for r in rats])]
    if sessions is not None:
        out = out[out["session"].isin([str(s) for s in sessions])]

    # Deterministic order: the CV split shuffles a LIST, so a reordered CSV would otherwise
    # change which events land in which fold at a fixed --split-seed.
    sort_cols = ["rat", "session", "trial", "event_start_lfp"]
    return out.sort_values(by=sort_cols).reset_index(drop=True)


def collect_traditional_event_handles(session_obj, events_df, *, min_len=2):
    """Per-event handles for externally supplied event windows, on the bundle's decode grid.

    Returns ``(handles, stats)``, where each handle is
    ``{"trial", "event_id", "s", "e", "spike_traj"}`` -- the same dict shape
    ``datasets.lustig.kernel_share.collect_events_with_metadata`` returns, so per-event analyses
    (kernel evidence, classification) can consume the traditional event set through the same
    code path as trajectory segments.

    Event boundaries come from a traditional-criteria events table instead of from trajectory
    calling. The traditional run supplies ONLY each event's start/stop time; the binning is
    this bundle's, so the counts are exactly what the pipeline would produce for those laps at
    this config's grid. `stats` records what the CSV offered against what was usable, so a run
    summary can show the difference rather than reporting a silently shortened set.

    `event_id` is the CSV's ``trial_pbe_index`` when that column is present -- which keeps every
    event joinable back to traditional_decode_events.csv -- and a within-lap counter otherwise.
    ``stats["event_id_source"]`` records which was used, because the two are only equal when
    the CSV lists every PBE of a lap.

    The fitter (through `collect_traditional_event_datas`, which wraps this) and every scorer
    place traditional events with this function, so a fit and its scoring see the same events.
    """
    if getattr(session_obj, "trial_starts_lfp", None) is None:
        raise ValueError(
            "SessionBundle has no `trial_starts_lfp`, so the lap decode grid cannot be "
            "rebuilt to place the event windows. Rebuild the bundle with build_session_bundle."
        )
    cfg = session_obj.cfg
    trial_starts = np.asarray(session_obj.trial_starts_lfp, dtype=int)
    trial_ends = np.asarray(session_obj.trial_ends_lfp, dtype=int)
    n_trials = int(trial_starts.size)

    events = events_df[
        (events_df["rat"].astype(str) == str(session_obj.rat))
        & (events_df["session"].astype(str) == str(session_obj.session))
    ]

    has_pbe_index = "trial_pbe_index" in events.columns
    handles: list[dict] = []
    n_dropped_short = 0
    n_dropped_no_bins = 0
    n_bins_total = 0

    for trial, trial_events in events.groupby("trial", sort=True):
        trial = int(trial)
        if not 1 <= trial <= n_trials:
            # Raise rather than warn: a mismatched lap convention would drop events silently,
            # and an understated event set looks identical to a genuinely small one.
            raise ValueError(
                f"{session_obj.rat}/{session_obj.session}: events reference trial {trial}, "
                f"outside 1..{n_trials}. The events CSV and this session disagree on laps."
            )
        rate_traj = np.asarray(session_obj.trial_rate_trajs[trial])  # (n_units, T_trial)
        dec_starts, dec_ends, _ = decode_window_grid(
            trial_starts[trial - 1], trial_ends[trial - 1], cfg
        )
        if dec_starts.size != rate_traj.shape[1]:
            raise ValueError(
                f"{session_obj.rat}/{session_obj.session} trial {trial}: rebuilt decode grid "
                f"has {dec_starts.size} bins but the bundle's counts have {rate_traj.shape[1]}; "
                "the bundle was built with a different config than the one it carries."
            )

        for lap_position, row in enumerate(trial_events.itertuples(index=False)):
            selected = select_event_bins(
                dec_starts, dec_ends, int(row.event_start_lfp), int(row.event_end_lfp)
            )
            context = (
                f"{session_obj.rat}/{session_obj.session} trial {trial} "
                f"event [{int(row.event_start_lfp)}, {int(row.event_end_lfp)}]"
            )
            assert_contiguous_selection(selected, context)
            if selected.size == 0:
                n_dropped_no_bins += 1
                continue
            if selected.size < min_len:
                n_dropped_short += 1
                continue
            first, last = int(selected[0]), int(selected[-1]) + 1
            event_id = (
                int(row.trial_pbe_index) if has_pbe_index else int(lap_position)
            )
            # int32 to match the other collectors: rate_traj is uint16 and signed arithmetic
            # inside ssm on the observation array would wrap.
            handles.append({
                "trial": trial,
                "event_id": event_id,
                "s": first,
                "e": last,
                "spike_traj": rate_traj[:, first:last].T.astype(np.int32),
            })
            n_bins_total += last - first

    stats = {
        "n_events_for_session": int(len(events)),
        "n_used": int(len(handles)),
        "n_dropped_short": int(n_dropped_short),
        "n_dropped_no_bins": int(n_dropped_no_bins),
        "n_bins_total": int(n_bins_total),
        "min_len": int(min_len),
        "event_id_source": "trial_pbe_index" if has_pbe_index else "lap_position",
    }
    return handles, stats


def collect_traditional_event_datas(session_obj, events_df, *, min_len=2):
    """Spike-count arrays for externally supplied event windows, on the bundle's decode grid.

    Mirrors `collect_mb_traj_event_datas`' contract -- a list of ``(T_event, n_units)`` int32
    arrays sliced out of `session_obj.trial_rate_trajs`. A thin projection of
    `collect_traditional_event_handles`, which owns the geometry; see its docstring.
    """
    handles, stats = collect_traditional_event_handles(
        session_obj, events_df, min_len=min_len
    )
    return [handle["spike_traj"] for handle in handles], stats
