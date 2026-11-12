"""Traditional memoryless Bayesian replay detection for the Buzsaki hc-11 dataset.

The Lustig traditional decode (``replay_hmm.datasets.lustig.traditional``) applied to the
Buzsaki linear-track sessions. The replay criteria are identical (weighted Spearman over
qualified decode windows plus a non-stationary-step count, both from
``replay_hmm.decode.traditional.criteria``), but the candidate map set collapses from Lustig's
ten maps (six directional arms + wheel + three arm-projected 2D maps) to the **two**
directional templates of a single linear track, and the timebase is seconds rather than
LFP/spike sample indices. Those two templates (`MAP_SPECS`), the per-event verdict and the
criteria sweep are the two-map half of ``replay_hmm.decode.traditional.criteria``, shared with
the Pfeiffer decoder.

Candidate events and the per-event sliding-window decode are the MB-jump analysis's own
(``datasets.buzsaki.events.candidate_events`` / ``decode.two_map.decode_pbe``), so the event
set and the posteriors are identical to it; only the accept/discard criteria applied on top
of them differ.

Run as `replay-hmm traditional decode --dataset buzsaki`. The default results root comes from
`replay_hmm.paths`, so the command writes the same place from any working directory.

Usage (from the repo root):
    replay-hmm traditional decode --dataset buzsaki --dry-run
    replay-hmm traditional decode --dataset buzsaki Achilles_10252013
    replay-hmm traditional decode --dataset buzsaki --n-jobs 5
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from replay_hmm import paths
from replay_hmm.datasets.buzsaki import events as buzsaki_events
from replay_hmm.datasets.buzsaki import kinematics as buzsaki_kinematics
from replay_hmm.datasets.buzsaki import session as buzsaki_session
from replay_hmm.datasets.buzsaki.session import rat_of_session
from replay_hmm.datasets.buzsaki.grids import TRADITIONAL_DECODE_GRID
from replay_hmm.decode import two_map
from replay_hmm.decode.grid import DecodeGrid
from replay_hmm.datasets.buzsaki import loader as buzsaki_loader
from replay_hmm.datasets import registry
from replay_hmm.datasets.buzsaki.ratemaps import (
    BIN_WIDTH_M,
    SPEED_THRESH,
    directional_rate_maps,
)
from replay_hmm.datasets.buzsaki.loader import BUZSAKI_DIR, load_session
from replay_hmm.decode import poisson
from replay_hmm.decode.traditional.criteria import (
    WINDOW_LABEL_COLUMNS,
    analyze_event_posteriors,
    criteria_sweep,
    print_acceptance_by_duration,
    print_criteria_sweep,
)
from replay_hmm.events import runs
from replay_hmm.util.tags import format_tag8


DEFAULT_RESULTS_ROOT = paths.BUZSAKI_TRADITIONAL_DECODE_DIR


@dataclass(frozen=True)
class DecodeSettings:
    bin_width_m: float = BIN_WIDTH_M
    decode_tbin_s: float = TRADITIONAL_DECODE_GRID.tbin_s
    decode_tbin_step_s: float = TRADITIONAL_DECODE_GRID.tbin_step_s
    min_prob: float = 0.01
    min_prob_factor: float | None = None
    min_active_neurons: int = 4
    min_abs_corval: float = 0.8
    min_numstep: int = 8
    max_event_duration_s: float | None = None
    # Candidate event source. Carried here for the same reason the PBE fields below are:
    # one DecodeSettings must fully determine the event set. `event_subset` is None when
    # the source's own default applies, which keeps the run-tag suffix empty by default.
    event_source: str = "spike-density"
    ripple_epochs: tuple[str, ...] = buzsaki_events.RIPPLE_EPOCHS
    event_subset: buzsaki_events.EventSubset | None = None
    # PBE detection. Carried here so one DecodeSettings fully determines both the event
    # set and the criteria, exactly as datasets.pfeiffer.traditional.DecodeSettings does.
    pbe_smooth_sigma_s: float = buzsaki_events.PBE_SMOOTH_SIGMA_S
    pbe_peak_z: float = buzsaki_events.PBE_THRESH_STDEV
    pbe_end_z: float = buzsaki_events.PBE_END_STDEV
    pbe_min_dur_s: float = buzsaki_events.PBE_MIN_DURATION_S
    pbe_merge_gap_s: float = buzsaki_events.PBE_MERGE_GAP_S
    speed_thresh_m_s: float = SPEED_THRESH
    pos_smooth_samples: float = buzsaki_kinematics.POS_SMOOTH_SAMPLES
    arm_end_gate_m: float | None = None
    arm_end_position_fill: str = "hold_last"

    def to_pbe_setting(self) -> buzsaki_events.PbeSetting:
        """The detection half of these settings, as the detector's own object.

        Mirrors ``datasets.pfeiffer.traditional.DecodeSettings.to_run_setting``. Validation
        lives in ``PbeSetting.__post_init__``, so a bad combination fails here too.
        """
        return buzsaki_events.PbeSetting(
            event_source=self.event_source,
            ripple_epochs=tuple(self.ripple_epochs),
            subset=self.event_subset,
            smooth_sigma_s=self.pbe_smooth_sigma_s,
            peak_z=self.pbe_peak_z,
            end_z=self.pbe_end_z,
            min_dur_s=self.pbe_min_dur_s,
            merge_gap_s=self.pbe_merge_gap_s,
            speed_thresh_m_s=self.speed_thresh_m_s,
            pos_smooth_samples=self.pos_smooth_samples,
            arm_end_gate_m=self.arm_end_gate_m,
            arm_end_position_fill=self.arm_end_position_fill,
        )

    def validate(self) -> None:
        if self.bin_width_m <= 0:
            raise ValueError("bin_width_m must be > 0.")
        if self.decode_tbin_s <= 0 or self.decode_tbin_step_s <= 0:
            raise ValueError("decode time-bin width and step must both be > 0.")
        if not 0 <= self.min_prob <= 1:
            raise ValueError("min_prob must be between 0 and 1.")
        if self.min_prob_factor is not None and self.min_prob_factor < 0:
            raise ValueError("min_prob_factor must be >= 0.")
        if self.min_active_neurons < 0:
            raise ValueError("min_active_neurons must be >= 0.")
        if not 0 <= self.min_abs_corval <= 1:
            raise ValueError("min_abs_corval must be between 0 and 1.")
        if self.min_numstep < 1:
            raise ValueError("min_numstep must be >= 1.")
        if self.max_event_duration_s is not None and self.max_event_duration_s <= 0:
            raise ValueError("max_event_duration_s must be > 0 when given.")
        self.to_pbe_setting()   # PBE-side validation, in one place

    def effective_min_prob(self, n_states: int) -> float:
        """Peak-posterior threshold for a joint space of ``n_states``.

        Buzsaki's joint space is only 80-100 states (uniform = 0.010-0.0125), so
        Lustig's absolute 0.01 threshold is effectively inert here. ``min_prob_factor``
        expresses the threshold as a multiple of uniform instead, mirroring
        ``jumps.mb.DEFAULT_MIN_TRAJ_PROB_FACTOR``.
        """
        if self.min_prob_factor is None:
            return float(self.min_prob)
        if n_states <= 0:
            raise ValueError("n_states must be > 0.")
        return float(self.min_prob_factor) / float(n_states)


EVENT_ID_COLUMNS = [
    "rat",
    "session",
    "bout_index",
    "bout_pbe_index",
    "session_pbe_index",
    "event_start_s",
    "event_end_s",
    "event_duration_s",
    "n_windows",
    # "maze" or "post". Always "maze" for spike-density events -- detect_pbes only ever
    # looks at the MAZE epoch -- and either for ripple events, which is what lets a
    # sleep-replay population be split from an awake one downstream.
    "epoch",
]
# The columns that identify one event (`criteria.criteria_sweep`'s keys): `session` is
# globally unique here.
EVENT_KEY_COLUMNS = ["session", "session_pbe_index"]
CANDIDATE_COLUMNS = EVENT_ID_COLUMNS + [
    "map_order",
    "map_label",
    "map_family",
    "template_direction",
    "n_qualified",
    "numstep",
    "corval",
    "mean_peak_probability",
    "is_candidate",
]
EVENT_COLUMNS = EVENT_ID_COLUMNS + [
    "status",
    "discard_reason",
    "n_candidate_maps",
    "selected_map",
    "selected_map_family",
    "selected_corval",
    "selected_numstep",
    "current_end",
    "replay_direction",
    "forward_reverse",
]
# One row per (event, decode window). Boundaries are half-open [start, end) intervals in
# seconds -- the same convention decode_pbe uses (searchsorted side="left" on both edges)
# -- so a downstream analysis can recount spike trajectories for any unit selection.
WINDOW_COLUMNS = [
    "rat",
    "session",
    "bout_index",
    "bout_pbe_index",
    "session_pbe_index",
    "window_index",
    "win_start_s",
    "win_end_s",
    "win_center_s",
] + WINDOW_LABEL_COLUMNS


def run_tag(settings: DecodeSettings) -> str:
    prob_tag = (
        f"pmin_{format_tag8(settings.min_prob)}"
        if settings.min_prob_factor is None
        else f"pfac_{format_tag8(settings.min_prob_factor)}"
    )
    tag = (
        f"bin_{format_tag8(settings.bin_width_m)}"
        f"__dtbin_{format_tag8(settings.decode_tbin_s)}"
        f"__step_{format_tag8(settings.decode_tbin_step_s)}"
        f"__{prob_tag}"
        f"__cells_{settings.min_active_neurons}"
        f"__corr_{format_tag8(settings.min_abs_corval)}"
        f"__steps_{settings.min_numstep}"
    )
    if settings.max_event_duration_s is not None:
        tag += f"__maxdur_{format_tag8(settings.max_event_duration_s)}"
    # Empty at the default source and subset. See buzsaki_events.event_source_tag for the
    # grammar and for why the detection token is conditional here where Pfeiffer makes it
    # mandatory.
    tag += buzsaki_events.event_source_tag(settings.to_pbe_setting())
    return tag


# The columns any consumer needs to re-identify an event written by this script: the flat
# session index it is keyed on, its time span, and the verdict. Lives here rather than in a
# consumer because this module writes the file.
REQUIRED_EVENT_COLUMNS = {
    "session",
    "session_pbe_index",
    "event_start_s",
    "event_end_s",
    "n_windows",
    "status",
}

# Written by this script but NOT required on read: a spike-density events CSV may lack it.
# `load_accepted_events` backfills the default.
OPTIONAL_EVENT_COLUMN_DEFAULTS = {
    # "maze" or "post". Always "maze" for spike-density events -- detect_pbes only ever
    # looks at the MAZE epoch -- and either for ripple events.
    "epoch": "maze",
}


def load_accepted_events(
    events_csv: Path,
    *,
    statuses: tuple[str, ...] = ("accepted",),
) -> pd.DataFrame:
    """Event rows with the columns needed to re-identify each event, filtered by verdict.

    `statuses` defaults to the accepted events, the set `replay-hmm jumps --dataset buzsaki
    --tradreplay-jumps` analyses and the HMM kernel comparison fits on. Only identity and time
    columns are read: consumers take the traditional
    criteria's EVENT SET and rebuild every downstream quantity on their own decode grid.
    """
    events_csv = Path(events_csv)
    if not events_csv.exists():
        raise FileNotFoundError(f"Traditional events CSV not found: {events_csv}")
    df = pd.read_csv(events_csv)
    missing = REQUIRED_EVENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{events_csv} is missing required columns: {sorted(missing)}")
    for column, default in OPTIONAL_EVENT_COLUMN_DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

    statuses = tuple(str(s) for s in statuses)
    present = set(df["status"].astype(str).unique())
    unknown = set(statuses) - present
    if unknown:
        # A typo'd verdict would otherwise yield an empty event set and surface much later
        # as a confusing "need >= 2 events" failure.
        raise ValueError(
            f"{events_csv} contains no rows with status {sorted(unknown)}; "
            f"present statuses are {sorted(present)}."
        )
    accepted = df[df["status"].astype(str).isin(statuses)].copy()

    accepted["session"] = accepted["session"].astype(str)
    accepted["session_pbe_index"] = accepted["session_pbe_index"].astype(int)
    accepted["n_windows"] = accepted["n_windows"].astype(int)
    accepted["event_start_s"] = accepted["event_start_s"].astype(float)
    accepted["event_end_s"] = accepted["event_end_s"].astype(float)
    if "rat" not in accepted:
        accepted["rat"] = accepted["session"].map(rat_of_session)
    accepted["rat"] = accepted["rat"].astype(str)
    return accepted


def eligible_sessions(
    sessions: list[str] | None = None,
    *,
    require_ripple_table: bool = False,
) -> list[tuple[str, Path]]:
    """(name, path) for every non-circular Buzsaki session, optionally filtered.

    ``registry.BUZSAKI_SESSIONS`` lists no circular session, so the name filter is a no-op;
    the real guard is the ``maze_type`` check applied after loading, as
    ``datasets.buzsaki.mb_jump.analyse_replay`` does.

    ``require_ripple_table`` drops the sessions that ship no SWR table -- Buddy_06272013 --
    which cannot take part in a ripple-sourced run at all. An explicitly named session
    without a table still raises below, so a typo is not silently swallowed.
    """
    available = [
        (name, BUZSAKI_DIR / rel_path)
        for name, rel_path in registry.BUZSAKI_SESSIONS
        if name not in registry.BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS
        and (not require_ripple_table or buzsaki_loader.has_ripple_table(name))
    ]
    if sessions is None:
        return available

    by_name = dict(available)
    unknown = [name for name in sessions if name not in by_name]
    if unknown:
        raise FileNotFoundError(
            "Unknown or non-linear Buzsaki session(s): "
            + ", ".join(unknown)
            + f". Available: {', '.join(sorted(by_name))}"
        )
    return [(name, by_name[name]) for name in dict.fromkeys(sessions)]


def decode_window_starts(
    t_start: float,
    t_end: float,
    tbin_s: float,
    tbin_step_s: float,
) -> np.ndarray:
    """The window start edges decode_pbe tiles inside an event.

    Reproduced rather than derived from ``dec["win_centers_s"]`` so the recorded edges are
    bit-identical to two_map.decode_pbe's own ``win_starts``.
    """
    return np.arange(float(t_start), float(t_end) - float(tbin_s) + 1e-9, float(tbin_step_s))


# ── Per-session processing ────────────────────────────────────────────────────────────

def _bout_assignment(
    pbes: list[dict],
    pfr_centers: np.ndarray,
    eligible: np.ndarray,
) -> tuple[list[int], int]:
    """Immobility-bout index per PBE: the Buzsaki analogue of Lustig's lap/trial.

    Candidates can occur anywhere on the track, so the grouping that structures the session
    is the contiguous run of eligible (tracked, non-running) bins a PBE sits in. This is the
    analogue of Pfeiffer's `interval_index`, which would be identically 0 here because
    Buzsaki has a single analysis interval; bouts number 300-1000 per session.

    The mask comes straight from the detection diagnostics rather than being rebuilt, so it
    is the mask that actually gated detection.

    A merged PBE can straddle two bouts (5 ms merge gap), so the bout is defined by the
    event's start sample. Returns (bout index per PBE, total number of bouts).
    """
    bouts = runs.runs_from_mask(np.asarray(eligible, dtype=bool), as_int=True, cast_bool=True)
    if not bouts:
        return [-1] * len(pbes), 0

    starts = np.array([s for s, _ in bouts], dtype=int)
    ends = np.array([e for _, e in bouts], dtype=int)

    assigned = []
    for pbe in pbes:
        sample = int(np.searchsorted(pfr_centers, float(pbe["start_s"]), side="left"))
        k = int(np.searchsorted(starts, sample, side="right")) - 1
        assigned.append(k if 0 <= k < starts.size and sample < ends[k] else -1)
    return assigned, len(bouts)


def process_session(
    session_name: str,
    mat_path: Path,
    settings: DecodeSettings,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    settings.validate()

    # The grid the event source (its two-window floor) and decode_pbe use. Built here from
    # the settings, which are what reach the joblib worker.
    decode_grid = DecodeGrid(
        tbin_s=float(settings.decode_tbin_s), tbin_step_s=float(settings.decode_tbin_step_s)
    )

    data = load_session(Path(mat_path))
    maze_type = str(data["maze_type"])
    pos_ts = data["pos_ts"]
    pos_1d = data["pos_1d"]
    track_len = float(np.nanmax(pos_1d))

    info: dict[str, object] = {
        "rat": rat_of_session(session_name),
        "session": session_name,
        "mat_path": str(mat_path),
        "maze_type": maze_type,
        "track_len_m": track_len,
        "n_bins": 0,
        "n_states": 0,
        "min_prob_effective": float("nan"),
        "n_decode_units": 0,
        "n_immobility_bouts": 0,
        "n_pbes": 0,
        "n_pbes_decoded": 0,
        "n_decode_windows": 0,
        "excluded": False,
    }
    if "Circular" in maze_type:
        info["excluded"] = True
        return [], [], [], info

    speed, velocity, nan_mask = buzsaki_kinematics.compute_speed(pos_1d, pos_ts, False, track_len)
    maps_out, maps_in, bin_edges, _, cell_ids = directional_rate_maps(
        data, speed, velocity, nan_mask, track_len=track_len, bin_width_m=settings.bin_width_m
    )
    n_bins = int(len(bin_edges) - 1)
    lin_map_concat = np.concatenate(
        [np.nan_to_num(maps_out, nan=0.0), np.nan_to_num(maps_in, nan=0.0)], axis=1
    )
    lin_map_bounds = [0, n_bins, 2 * n_bins]
    cache = poisson.build_decode_cache(
        lin_map_concat,
        lin_map_bounds,
        float(settings.decode_tbin_s),
        include_wheel=False,
    )
    info.update(
        {
            "n_bins": n_bins,
            "n_states": 2 * n_bins,
            "min_prob_effective": settings.effective_min_prob(2 * n_bins),
            "n_decode_units": int(len(cell_ids)),
        }
    )

    # candidate_events dispatches on the event source and applies the event subset; it
    # always returns (events, diagnostics), and the event list can still be empty when
    # every candidate fails the duration / speed / participation filters.
    pbes, pbe_diagnostics = buzsaki_events.candidate_events(
        data, settings.to_pbe_setting(), session_name=session_name, decode_grid=decode_grid
    )
    if not pbes:
        return [], [], [], info
    info["n_pbes_pre_subset"] = int(pbe_diagnostics.get("n_events_pre_subset", len(pbes)))

    bout_indices, n_bouts = _bout_assignment(
        pbes, pbe_diagnostics["pfr_centers"], pbe_diagnostics["eligible"]
    )
    info["n_pbes"] = int(len(pbes))
    info["n_immobility_bouts"] = int(n_bouts)

    cell_spike_times = buzsaki_session.pyramidal_spike_times(data, cell_ids)

    candidate_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    bout_counters: dict[int, int] = {}

    # `session_pbe_index` is carried by the event, NOT inferred from list position. The
    # two coincide for the default spike-density run, but a spike-density run with an
    # event subset drops events without renumbering (so CSV joins keep resolving), and
    # there position != index. Everything downstream keys on the carried value.
    # Strict monotonicity is what actually has to hold: it is the guarantee that detection
    # order is stable and that the index means the same thing in every consumer.
    previous_index = -1
    for position, pbe in enumerate(pbes):
        session_pbe_index = int(pbe["session_pbe_index"])
        if session_pbe_index <= previous_index:
            raise ValueError(
                f"event at position {position} carries session_pbe_index "
                f"{session_pbe_index} after {previous_index}; detection order is not stable."
            )
        previous_index = session_pbe_index
        bout_index = bout_indices[position]
        bout_pbe_index = bout_counters.get(bout_index, 0)
        bout_counters[bout_index] = bout_pbe_index + 1

        if (
            settings.max_event_duration_s is not None
            and float(pbe["duration_s"]) > settings.max_event_duration_s
        ):
            continue

        dec = two_map.decode_pbe(pbe, cell_spike_times, cell_ids, cache, n_bins,
                                 decode_grid=decode_grid)
        if dec is None:
            continue

        posterior_joint = dec["posterior_joint"]
        active_neurons = np.count_nonzero(np.asarray(dec["rate_traj"]) > 0, axis=0)
        win_centers = np.asarray(dec["win_centers_s"], dtype=float)
        win_starts = decode_window_starts(
            pbe["start_s"],
            pbe["end_s"],
            settings.decode_tbin_s,
            settings.decode_tbin_step_s,
        )
        if win_starts.size != win_centers.size:
            raise ValueError(
                f"{session_name} PBE {session_pbe_index}: reconstructed "
                f"{win_starts.size} window starts but decode_pbe produced "
                f"{win_centers.size} windows."
            )
        win_ends = win_starts + float(settings.decode_tbin_s)

        event_center_s = 0.5 * (float(pbe["start_s"]) + float(pbe["end_s"]))
        # LABEL ONLY, and None where position is unknown: interpolated from the raw
        # OneDLocation, not from a filled trace. OneDLocation is NaN for most MAZE samples,
        # and most events sit in exactly that time (immobility happens off the linearizable
        # segment), so this column is often None.
        # `datasets.pfeiffer.ratemaps.current_end_label` is the counterpart; Pfeiffer has no
        # position NaNs, so there it is always defined.
        animal_pos = float(np.interp(event_center_s, pos_ts, pos_1d, left=np.nan, right=np.nan))
        if not np.isfinite(animal_pos):
            current_end = None
        elif animal_pos <= buzsaki_kinematics.ARM_END_M:
            current_end = "near_0"
        elif animal_pos >= track_len - buzsaki_kinematics.ARM_END_M:
            current_end = "near_L"
        else:
            current_end = None

        candidates, event = analyze_event_posteriors(
            posterior_joint=posterior_joint,
            active_neurons=active_neurons,
            n_bins=n_bins,
            settings=settings,
            current_end=current_end,
        )
        assert event["status"] != "discarded", (
            "the multiple_contents discard is structurally unreachable on a "
            "single-track dataset"
        )

        identifiers = {
            "rat": rat_of_session(session_name),
            "session": session_name,
            "bout_index": int(bout_index),
            "bout_pbe_index": int(bout_pbe_index),
            "session_pbe_index": int(session_pbe_index),
            "event_start_s": float(pbe["start_s"]),
            "event_end_s": float(pbe["end_s"]),
            "event_duration_s": float(pbe["duration_s"]),
            "n_windows": int(win_centers.size),
            # Always "maze" for spike-density; "maze" or "post" for ripple events.
            "epoch": str(pbe.get("epoch", "maze")),
        }
        candidate_rows.extend({**identifiers, **row} for row in candidates)
        event_rows.append({**identifiers, **event})

        label_values = {key: event[key] for key in WINDOW_LABEL_COLUMNS}
        for window_index in range(int(win_centers.size)):
            window_rows.append(
                {
                    "rat": rat_of_session(session_name),
                    "session": session_name,
                    "bout_index": int(bout_index),
                    "bout_pbe_index": int(bout_pbe_index),
                    "session_pbe_index": int(session_pbe_index),
                    "window_index": window_index,
                    "win_start_s": float(win_starts[window_index]),
                    "win_end_s": float(win_ends[window_index]),
                    "win_center_s": float(win_centers[window_index]),
                    **label_values,
                }
            )

    info["n_pbes_decoded"] = int(len(event_rows))
    info["n_decode_windows"] = int(len(window_rows))
    return candidate_rows, event_rows, window_rows, info


# ── Summaries and diagnostics ─────────────────────────────────────────────────────────

def build_session_summary(
    event_df: pd.DataFrame,
    session_infos: list[dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for info in session_infos:
        subset = event_df[event_df["session"] == str(info["session"])]
        accepted = subset[subset["status"] == "accepted"]

        def count(column: str, value: str) -> int:
            return int(np.sum(accepted[column] == value)) if column in accepted else 0

        rows.append(
            {
                **info,
                "n_replay": int(len(accepted)),
                "n_no_candidate": int(np.sum(subset["status"] == "no_candidate")),
                "n_inbound": count("replay_direction", "Inbound"),
                "n_outbound": count("replay_direction", "Outbound"),
                "n_forward": count("forward_reverse", "Forward"),
                "n_reverse": count("forward_reverse", "Reverse"),
                "n_near_0": count("current_end", "near_0"),
                "n_near_L": count("current_end", "near_L"),
            }
        )
    return pd.DataFrame(rows)


def write_results(
    results_dir: Path,
    candidate_rows: list[dict[str, object]],
    event_rows: list[dict[str, object]],
    window_rows: list[dict[str, object]],
    session_infos: list[dict[str, object]],
    settings: DecodeSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_dir.mkdir(parents=True, exist_ok=True)
    candidate_df = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    event_df = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    window_df = pd.DataFrame(window_rows, columns=WINDOW_COLUMNS)
    summary_df = build_session_summary(event_df, session_infos)
    sweep_df = criteria_sweep(candidate_df, keys=EVENT_KEY_COLUMNS)

    candidate_path = results_dir / "traditional_decode_candidates.csv"
    event_path = results_dir / "traditional_decode_events.csv"
    window_path = results_dir / "traditional_decode_windows.csv"
    summary_path = results_dir / "traditional_decode_session_summary.csv"
    sweep_path = results_dir / "traditional_decode_criteria_sweep.csv"
    metadata_path = results_dir / "traditional_decode_metadata.json"
    candidate_df.to_csv(candidate_path, index=False)
    event_df.to_csv(event_path, index=False)
    window_df.to_csv(window_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    sweep_df.to_csv(sweep_path, index=False)

    metadata = {
        "run_tag": run_tag(settings),
        "settings": asdict(settings),
        "unit_selection": {
            "cells": "pyr_ids",
            "place_field_filter": False,
            "convention": "all pyramidal cells, as in datasets.buzsaki.mb_jump",
        },
        "pbe_provenance": settings.to_pbe_setting().pbe_provenance(),
        "event_source_provenance": buzsaki_events.event_source_provenance(
            settings.to_pbe_setting()
        ),
        "sessions": [
            {
                key: info[key]
                for key in (
                    "rat",
                    "session",
                    "mat_path",
                    "maze_type",
                    "track_len_m",
                    "n_bins",
                    "n_states",
                    "min_prob_effective",
                    "n_decode_units",
                    "n_immobility_bouts",
                    "n_pbes",
                    "excluded",
                )
            }
            for info in session_infos
        ],
        "output_files": {
            "candidates": str(candidate_path),
            "events": str(event_path),
            "windows": str(window_path),
            "session_summary": str(summary_path),
            "criteria_sweep": str(sweep_path),
        },
        "schemas": {
            "candidates": list(candidate_df.columns),
            "events": list(event_df.columns),
            "windows": list(window_df.columns),
            "session_summary": list(summary_df.columns),
            "criteria_sweep": list(sweep_df.columns),
        },
        "notes": {
            "window_boundary_convention": (
                "win_start_s/win_end_s are half-open [start, end) intervals in seconds, "
                "matching decode_pbe (searchsorted side='left' on both edges); recount "
                "spikes with the same cell selection to rebuild spike-count trajectories."
            ),
            "multiple_contents_discard": (
                "Unreachable here. Both templates describe the same physical track, so "
                "the Lustig content set is always a singleton and an out/in ambiguity is "
                "resolved by |corval| rather than discarded. status is only 'accepted' or "
                "'no_candidate'; n_candidate_maps == 2 flags the ambiguous events."
            ),
            "dropped_columns": (
                "replay_arm / current_arm / local_remote have no analogue on a single "
                "linear track and are omitted rather than written as nulls. current_end "
                "records which track end the animal occupied at the event midpoint."
            ),
            "state_assignment": (
                "Decoded position is the global argmax over the concatenated "
                "[outbound | inbound] posterior, matching Lustig. This differs from "
                "decode_pbe's own decoded_pos/dir_label, which choose the directional "
                "half by summed posterior mass; the two disagree on roughly 3% of windows."
            ),
            "min_prob_scale": (
                "The joint state space is only 2 * n_bins (80-100 states), so an absolute "
                "min_prob of 0.01 is at or below uniform and is effectively inert. See "
                "each session's min_prob_effective and n_states; use --min-prob-factor to "
                "set the threshold as a multiple of uniform instead."
            ),
            "forward_reverse": (
                "With two templates on a shared 0->track_len bin axis, forward_reverse is "
                "exactly (selected_map, sign(selected_corval)) re-expressed and carries no "
                "independent information."
            ),
        },
    }
    with metadata_path.open("w") as stream:
        json.dump(metadata, stream, indent=2, default=str)
        stream.write("\n")
    return candidate_df, event_df, window_df, summary_df, sweep_df


# ── CLI ───────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traditional memoryless Bayesian replay detection for the Buzsaki "
        "hc-11 dataset.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "sessions",
        nargs="*",
        help="Session name(s); defaults to every eligible linear session.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--bin-width-m", type=float, default=BIN_WIDTH_M)
    parser.add_argument("--decode-tbin-s", type=float,
                        default=TRADITIONAL_DECODE_GRID.tbin_s)
    parser.add_argument("--decode-tbin-step-s", type=float,
                        default=TRADITIONAL_DECODE_GRID.tbin_step_s)
    parser.add_argument("--min-prob", type=float, default=0.01)
    parser.add_argument(
        "--min-prob-factor",
        type=float,
        default=None,
        help="Peak-posterior threshold as a multiple of uniform (1/n_states). "
        "Overrides --min-prob when given.",
    )
    parser.add_argument("--min-active-neurons", type=int, default=4)
    parser.add_argument("--min-abs-corval", type=float, default=0.8)
    parser.add_argument("--min-numstep", type=int, default=8)
    parser.add_argument(
        "--max-event-duration-s",
        type=float,
        default=None,
        help="Skip PBEs longer than this; unset by default (detect_pbes has no cap).",
    )
    buzsaki_events.add_pbe_args(parser)
    buzsaki_events.add_event_source_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The event source, epochs and subset are read out of `event_setting_from_args`, which
    # resolves the source's subset defaults; the PBE fields are read off `args` and
    # converted to metres the same way `pbe_setting_from_args` does.
    event_setting = buzsaki_events.event_setting_from_args(args)
    settings = DecodeSettings(
        bin_width_m=args.bin_width_m,
        decode_tbin_s=args.decode_tbin_s,
        decode_tbin_step_s=args.decode_tbin_step_s,
        min_prob=args.min_prob,
        min_prob_factor=args.min_prob_factor,
        min_active_neurons=args.min_active_neurons,
        min_abs_corval=args.min_abs_corval,
        min_numstep=args.min_numstep,
        max_event_duration_s=args.max_event_duration_s,
        event_source=event_setting.event_source,
        ripple_epochs=tuple(event_setting.ripple_epochs),
        event_subset=event_setting.subset,
        pbe_smooth_sigma_s=args.pbe_smooth_sigma_s,
        pbe_peak_z=args.pbe_peak_z,
        pbe_end_z=args.pbe_end_z,
        pbe_min_dur_s=args.pbe_min_dur_s,
        pbe_merge_gap_s=args.pbe_merge_gap_s,
        # cm -> m, as in buzsaki_events.pbe_setting_from_args.
        speed_thresh_m_s=args.speed_thresh_cm_s / 100.0,
        pos_smooth_samples=args.pos_smooth_samples,
        arm_end_gate_m=(
            None if args.arm_end_gate_cm is None else args.arm_end_gate_cm / 100.0
        ),
        arm_end_position_fill=args.arm_end_position_fill,
    )
    settings.validate()
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs cannot be 0.")

    session_items = eligible_sessions(
        args.sessions or None,
        require_ripple_table=settings.to_pbe_setting().is_table_source,
    )
    if settings.to_pbe_setting().is_table_source:
        skipped = [
            name for name, _ in registry.BUZSAKI_SESSIONS
            if name not in registry.BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS
            and not buzsaki_loader.has_ripple_table(name)
        ]
        if skipped:
            print("  skipping (no ripple table): " + ", ".join(skipped))
    missing = [str(path) for _, path in session_items if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Session file(s) not found: " + ", ".join(missing))
    results_dir = Path(args.results_root) / run_tag(settings)

    print("Buzsaki traditional replay decode run plan")
    print(f"  run_tag:      {run_tag(settings)}")
    print(f"  sessions:     {[name for name, _ in session_items]}")
    print(f"  spatial bins: {settings.bin_width_m:g} m")
    print(
        f"  decode bins:  {settings.decode_tbin_s:g}s every "
        f"{settings.decode_tbin_step_s:g}s"
    )
    if settings.min_prob_factor is None:
        print(f"  thresholds:   p>{settings.min_prob:g} (absolute)", end="")
    else:
        print(f"  thresholds:   p>{settings.min_prob_factor:g}/n_states", end="")
    print(
        f", cells>={settings.min_active_neurons}, "
        f"|r|>={settings.min_abs_corval:g}, numstep>={settings.min_numstep}"
    )
    print(f"  n_jobs:       {args.n_jobs}")
    print(f"  results_dir:  {results_dir}")
    if args.dry_run:
        return 0

    if args.n_jobs == 1:
        processed = [
            process_session(name, path, settings) for name, path in session_items
        ]
    else:
        processed = Parallel(n_jobs=args.n_jobs)(
            delayed(process_session)(name, path, settings)
            for name, path in session_items
        )

    candidate_rows = [row for candidates, _, _, _ in processed for row in candidates]
    event_rows = [row for _, events, _, _ in processed for row in events]
    window_rows = [row for _, _, windows, _ in processed for row in windows]
    session_infos = [info for _, _, _, info in processed]

    candidate_df, event_df, window_df, summary_df, sweep_df = write_results(
        results_dir,
        candidate_rows,
        event_rows,
        window_rows,
        session_infos,
        settings,
    )

    print("\nTraditional replay summary")
    if summary_df.empty:
        print("  No sessions were processed.")
    else:
        print(
            summary_df[
                [
                    "session",
                    "n_states",
                    "n_immobility_bouts",
                    "n_pbes",
                    "n_pbes_decoded",
                    "n_replay",
                    "n_no_candidate",
                ]
            ].to_string(index=False)
        )

    print("\nCriteria sweep (diagnostic; min_prob / min_active_neurons held fixed)")
    print_criteria_sweep(sweep_df)
    print()
    print_acceptance_by_duration(event_df)

    print(
        f"\nSaved {len(candidate_df)} candidate rows, {len(event_df)} PBE rows, "
        f"and {len(window_df)} decode-window rows to {results_dir}"
    )
    return 0
