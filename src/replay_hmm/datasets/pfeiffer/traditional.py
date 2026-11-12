"""Traditional memoryless Bayesian replay detection for the Pfeiffer & Foster dataset.

The Pfeiffer counterpart of ``datasets.buzsaki.traditional``. The replay criteria are
**identical** -- weighted Spearman over qualified decode windows plus a non-stationary-step
count, accepted at |r| >= 0.8 and steps >= 8 -- and the two directional templates of a single
linear track are the same shape of problem. ``weighted_spearman`` and
``count_nonstationary_steps`` are the shared ``replay_hmm.decode.traditional.criteria``
functions that the Buzsaki and Lustig decoders call too, and
``tests/test_pfeiffer_traditional_decode.py`` pins their values. The two templates
(`MAP_SPECS`), the per-event verdict and the criteria sweep are the same module's two-map
half, which the Buzsaki decoder runs too.

Candidate events and the per-event spike counts come from ``pfeiffer_events``
(``candidate_events`` / ``build_spike_traj_for_event``), so the event set and posteriors
match the MB-jump analysis exactly; only the accept/discard criteria applied on top differ.
Any of the four event sources may be used.

Run as `replay-hmm traditional decode --dataset pfeiffer`. The default results root comes from
`replay_hmm.paths`, so the command writes the same place from any working directory.

Usage (from the repo root):
    replay-hmm traditional decode --dataset pfeiffer --dry-run
    replay-hmm traditional decode --dataset pfeiffer Rat1/Linear1
    replay-hmm traditional decode --dataset pfeiffer --n-jobs 5 --event-source swr-gated
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
from replay_hmm.decode import poisson
from replay_hmm.decode.traditional import criteria
from replay_hmm.decode.traditional.criteria import (
    WINDOW_LABEL_COLUMNS,
    analyze_event_posteriors,
    criteria_sweep,
    print_acceptance_by_duration,
    print_criteria_sweep,
)
from replay_hmm.datasets.pfeiffer import kinematics
from replay_hmm.datasets.pfeiffer import loader
from replay_hmm.datasets.pfeiffer import ratemaps
from replay_hmm.datasets.pfeiffer import events as pfeiffer_events
from replay_hmm.datasets.pfeiffer.grids import TRADITIONAL_DECODE_GRID
from replay_hmm.datasets.registry import PFEIFFER_EVENT_SOURCE_TAGS, PFEIFFER_EVENT_SOURCES
from replay_hmm.util.tags import format_tag8

DEFAULT_RESULTS_ROOT = paths.PFEIFFER_TRADITIONAL_DECODE_DIR


@dataclass(frozen=True)
class DecodeSettings:
    bin_width_cm: float = ratemaps.BIN_WIDTH_CM
    decode_tbin_s: float = TRADITIONAL_DECODE_GRID.tbin_s
    decode_tbin_step_s: float = TRADITIONAL_DECODE_GRID.tbin_step_s
    event_source: str = "spike-density"
    min_prob: float = 0.01
    min_prob_factor: float | None = None
    min_active_neurons: int = 4
    min_abs_corval: float = 0.8
    min_numstep: int = 8
    max_event_duration_s: float | None = None

    def validate(self) -> None:
        if self.bin_width_cm <= 0:
            raise ValueError("bin_width_cm must be > 0.")
        if self.decode_tbin_s <= 0 or self.decode_tbin_step_s <= 0:
            raise ValueError("decode time-bin width and step must both be > 0.")
        if self.event_source not in PFEIFFER_EVENT_SOURCES:
            raise ValueError(
                f"Unknown event source {self.event_source!r}; expected one of {PFEIFFER_EVENT_SOURCES}"
            )
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

    def effective_min_prob(self, n_states: int) -> float:
        """Peak-posterior threshold for a joint space of ``n_states``.

        On Pfeiffer, at 4 cm bins the track gives 40-46 bins, so
        ``n_states`` is 80-92 and uniform is 0.0109-0.0125. The absolute 0.01 threshold
        therefore sits **at or below uniform** and excludes no windows at all -- exactly
        the situation ``datasets.buzsaki.traditional`` documents for itself. It is retained
        as the default so the run tag stays comparable across datasets; the binding
        per-window filter here is ``min_active_neurons``. Use ``--min-prob-factor`` to
        express the threshold as a multiple of uniform instead.
        """
        if self.min_prob_factor is None:
            return float(self.min_prob)
        if n_states <= 0:
            raise ValueError("n_states must be > 0.")
        return float(self.min_prob_factor) / float(n_states)

    def to_run_setting(self) -> pfeiffer_events.RunSetting:
        """The matching ``pfeiffer_events.RunSetting`` for event detection and decoding."""
        return pfeiffer_events.RunSetting(
            event_source=self.event_source,
            bin_width_cm=self.bin_width_cm,
            tbin_s=self.decode_tbin_s,
            tbin_step_s=self.decode_tbin_step_s,
        )


# `rat` and `session` are BOTH explicit columns. Buzsaki gets away with a single
# `session` column because `Achilles_10252013` is globally unique; Pfeiffer's `Linear1`
# occurs under three rats, so (rat, session, session_pbe_index) is the join key.
EVENT_ID_COLUMNS = [
    "rat",
    "session",
    "interval_index",
    "interval_event_index",
    "session_pbe_index",
    "source_row",
    "event_start_s",
    "event_end_s",
    "event_duration_s",
    "n_windows",
]
# The columns that identify one event (`criteria.criteria_sweep`'s keys).
EVENT_KEY_COLUMNS = ["rat", "session", "session_pbe_index"]
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
WINDOW_COLUMNS = [
    "rat",
    "session",
    "interval_index",
    "interval_event_index",
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
        f"bin_{format_tag8(settings.bin_width_cm)}"
        f"__dtbin_{format_tag8(settings.decode_tbin_s)}"
        f"__step_{format_tag8(settings.decode_tbin_step_s)}"
        f"__{prob_tag}"
        f"__cells_{settings.min_active_neurons}"
        f"__corr_{format_tag8(settings.min_abs_corval)}"
        f"__steps_{settings.min_numstep}"
        f"__src_{PFEIFFER_EVENT_SOURCE_TAGS[settings.event_source]}"
    )
    if settings.max_event_duration_s is not None:
        tag += f"__maxdur_{format_tag8(settings.max_event_duration_s)}"
    return tag


# `criteria.read_traditional_metadata`'s error for a missing metadata JSON, as the Pfeiffer
# consumers (the fit and the jump distribution) word it.
METADATA_MISSING_MESSAGE = "Traditional decode metadata not found: {path}"

# The columns any consumer needs to re-identify an event written by this module.
REQUIRED_EVENT_COLUMNS = {
    "rat",
    "session",
    "session_pbe_index",
    "event_start_s",
    "event_end_s",
    "n_windows",
    "status",
}


def load_accepted_events(
    events_csv: Path,
    *,
    statuses: tuple[str, ...] = ("accepted",),
) -> pd.DataFrame:
    """Event rows needed to re-identify each event, filtered by verdict.

    Only identity and time columns are read: consumers take the traditional criteria's
    EVENT SET and rebuild every downstream quantity on their own decode grid.
    """
    events_csv = Path(events_csv)
    if not events_csv.exists():
        raise FileNotFoundError(f"Traditional events CSV not found: {events_csv}")
    df = pd.read_csv(events_csv, dtype={"rat": str, "session": str})
    missing = REQUIRED_EVENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{events_csv} is missing required columns: {sorted(missing)}")

    statuses = tuple(str(s) for s in statuses)
    present = set(df["status"].astype(str).unique())
    unknown = set(statuses) - present
    if unknown:
        # A typo'd verdict would otherwise yield an empty event set and surface much
        # later as a confusing "need >= 2 events" failure.
        raise ValueError(
            f"{events_csv} contains no rows with status {sorted(unknown)}; "
            f"present statuses are {sorted(present)}."
        )
    accepted = df[df["status"].astype(str).isin(statuses)].copy()
    accepted["rat"] = accepted["rat"].astype(str)
    accepted["session"] = accepted["session"].astype(str)
    accepted["session_pbe_index"] = accepted["session_pbe_index"].astype(int)
    accepted["n_windows"] = accepted["n_windows"].astype(int)
    accepted["event_start_s"] = accepted["event_start_s"].astype(float)
    accepted["event_end_s"] = accepted["event_end_s"].astype(float)
    if "source_row" in accepted:
        accepted["source_row"] = accepted["source_row"].astype(int)
    # Deterministic order: the CV split shuffles a list, so a reordered CSV would change
    # fold membership at a fixed --split-seed
    # (datasets.lustig.traditional_io.load_traditional_events does the same).
    return accepted.sort_values(
        ["rat", "session", "session_pbe_index"], kind="stable"
    ).reset_index(drop=True)


def verify_event_alignment(
    rat: str,
    session: str,
    events: list[dict],
    events_df: pd.DataFrame,
    *,
    event_source: str,
    metadata_event_source: str | None = None,
    tol_s: float = 1e-6,
) -> None:
    """Assert a re-detected event list still matches the CSV it is being joined to.

    ``criteria.verify_event_alignment`` (the check Buzsaki's consumers run too), keyed on
    list position, plus two Pfeiffer-specific checks: the event source, and ``source_row``.

    The ``event_source`` check matters because the four sources overlap heavily -- every
    Trajectory_Events row lies inside a Ripple_Events row -- so an index-match against the
    wrong source could coincidentally pass the start-time assertion. Comparing the source
    up front fails early with a clear message.
    """
    if metadata_event_source is not None and str(metadata_event_source) != str(event_source):
        raise ValueError(
            f"{rat}/{session}: traditional events were produced with event source "
            f"{metadata_event_source!r} but this run re-detected with {event_source!r}. "
            f"Pass --event-source {metadata_event_source} to match the CSV."
        )

    subset = events_df[
        (events_df["rat"].astype(str) == str(rat))
        & (events_df["session"].astype(str) == str(session))
    ]
    criteria.verify_event_alignment(
        f"{rat}/{session}", events, subset,
        key="position", tol_s=tol_s, check_source_row=True, missing_column_error=KeyError,
    )


# ── Per-session processing ────────────────────────────────────────────────────────────

def process_session(
    rat: str,
    session: str,
    sdir: Path,
    settings: DecodeSettings,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    settings.validate()
    setting = settings.to_run_setting()

    data = loader.load_session_data(Path(sdir))
    speed, velocity = kinematics.compute_speed(data)
    rate = ratemaps.directional_rate_maps(
        data, speed, velocity, bin_width_cm=settings.bin_width_cm
    )
    lin_map, lin_map_bounds = ratemaps.lin_map_concat(rate["maps"])
    n_bins = int(rate["n_bins"])
    cache = poisson.build_decode_cache(
        lin_map, lin_map_bounds, float(settings.decode_tbin_s), include_wheel=False
    )
    spikes = loader.cell_spike_times(data)
    cell_ids = rate["cell_ids"]

    info: dict[str, object] = {
        "rat": rat,
        "session": session,
        "session_dir": str(sdir),
        "linear_axis": int(data["linear_axis"]),
        "track_len_cm": float(data["track_len_cm"]),
        "n_bins": n_bins,
        "n_states": 2 * n_bins,
        "min_prob_effective": settings.effective_min_prob(2 * n_bins),
        "n_decode_units": int(cell_ids.size),
        "n_intervals": int(len(data["analysis_intervals_s"])),
        "n_pbes": 0,
        "n_pbes_decoded": 0,
        "n_decode_windows": 0,
        "excluded": False,
    }

    events, diagnostics = pfeiffer_events.candidate_events(data, setting)
    info["n_pbes"] = int(len(events))
    info["detection_diagnostics"] = {
        k: v for k, v in diagnostics.items() if k != "grids"
    }
    if not events:
        return [], [], [], info

    candidate_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    # session_pbe_index enumerates candidate_events' output *before* any events are
    # dropped here, so a consumer can index-match against a re-detected list.
    for event in events:
        if (
            settings.max_event_duration_s is not None
            and float(event["duration_s"]) > settings.max_event_duration_s
        ):
            continue

        spike_traj, win_starts, win_centers = pfeiffer_events.build_spike_traj_for_event(
            spikes, cell_ids, event["start_s"], event["end_s"],
            settings.decode_tbin_s, settings.decode_tbin_step_s,
        )
        if spike_traj.shape[1] < 2:
            continue
        posterior_joint = poisson.decode_posterior(spike_traj, cache)
        active_neurons = np.count_nonzero(spike_traj > 0, axis=0)

        expected = pfeiffer_events.decode_window_starts(event["start_s"], event["end_s"], setting)
        if expected.size != win_starts.size:
            raise ValueError(
                f"{rat}/{session} event {event['session_pbe_index']}: reconstructed "
                f"{expected.size} window starts but the decode produced {win_starts.size}."
            )
        win_ends = win_starts + float(settings.decode_tbin_s)

        candidates, resolved = analyze_event_posteriors(
            posterior_joint=posterior_joint,
            active_neurons=active_neurons,
            n_bins=n_bins,
            settings=settings,
            current_end=event["current_end"],
        )
        assert resolved["status"] != "discarded", (
            "the multiple_contents discard is structurally unreachable on a "
            "single-track dataset"
        )

        identifiers = {
            "rat": rat,
            "session": session,
            "interval_index": int(event["interval_index"]),
            "interval_event_index": int(event["interval_event_index"]),
            "session_pbe_index": int(event["session_pbe_index"]),
            "source_row": int(event["source_row"]),
            "event_start_s": float(event["start_s"]),
            "event_end_s": float(event["end_s"]),
            "event_duration_s": float(event["duration_s"]),
            "n_windows": int(win_centers.size),
        }
        candidate_rows.extend({**identifiers, **row} for row in candidates)
        event_rows.append({**identifiers, **resolved})

        label_values = {key: resolved[key] for key in WINDOW_LABEL_COLUMNS}
        for window_index in range(int(win_centers.size)):
            window_rows.append(
                {
                    "rat": rat,
                    "session": session,
                    "interval_index": int(event["interval_index"]),
                    "interval_event_index": int(event["interval_event_index"]),
                    "session_pbe_index": int(event["session_pbe_index"]),
                    "window_index": window_index,
                    "win_start_s": float(win_starts[window_index]),
                    "win_end_s": float(win_ends[window_index]),
                    "win_center_s": float(win_centers[window_index]),
                    **label_values,
                }
            )

    info["n_pbes_decoded"] = int(len(event_rows))
    info["n_decode_windows"] = int(len(window_rows))
    # How many events could not possibly clear min_numstep because they are too short.
    # Load-bearing for the table sources, whose windows are not length-filtered.
    info["n_events_below_numstep_floor"] = int(
        sum(1 for r in event_rows if int(r["n_windows"]) < settings.min_numstep)
    )
    return candidate_rows, event_rows, window_rows, info


# ── Summaries and diagnostics ─────────────────────────────────────────────────────────

def build_session_summary(
    event_df: pd.DataFrame,
    session_infos: list[dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for info in session_infos:
        if event_df.empty:
            subset = event_df
        else:
            subset = event_df[
                (event_df["rat"].astype(str) == str(info["rat"]))
                & (event_df["session"].astype(str) == str(info["session"]))
            ]
        accepted = subset[subset["status"] == "accepted"] if not subset.empty else subset

        def count(column: str, value: str) -> int:
            if accepted.empty or column not in accepted:
                return 0
            return int(np.sum(accepted[column] == value))

        rows.append(
            {
                **{k: v for k, v in info.items() if k != "detection_diagnostics"},
                "n_replay": int(len(accepted)),
                "n_no_candidate": (
                    0 if subset.empty else int(np.sum(subset["status"] == "no_candidate"))
                ),
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
        "event_source": settings.event_source,
        "length_units": "cm",
        "pbe_provenance": settings.to_run_setting().pbe_provenance(),
        "unit_selection": {
            "cells": "Excitatory_Neurons",
            "place_field_filter": False,
            "convention": (
                "all putative excitatory units, matching the Buzsaki traditional decode's "
                "unfiltered pyr_ids"
            ),
        },
        "event_detection": (
            f"datasets.pfeiffer.events.candidate_events(source={settings.event_source!r}); "
            "spike-density sources use 1 ms bins, sigma = 20 ms, session-wide immobility "
            "baseline, mean + 3 SD peak, mean-crossing boundaries, >= 100 ms, 5 ms merge "
            "gap, in-event speed <= 5 cm/s; table sources take the dataset's own windows "
            "with only the interval, immobility and >= 2-window gates."
        ),
        "sessions": [
            {
                key: info[key]
                for key in (
                    "rat",
                    "session",
                    "session_dir",
                    "linear_axis",
                    "track_len_cm",
                    "n_bins",
                    "n_states",
                    "min_prob_effective",
                    "n_decode_units",
                    "n_intervals",
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
            "join_key": (
                "(rat, session, session_pbe_index). Both rat and session are explicit "
                "columns because 'Linear1' occurs under three rats -- unlike Buzsaki, "
                "the session name alone is not unique."
            ),
            "window_boundary_convention": (
                "win_start_s/win_end_s are half-open [start, end) intervals in seconds, "
                "matching build_spike_traj_for_event (searchsorted side='left' on both "
                "edges); recount spikes with the same cell selection to rebuild "
                "spike-count trajectories."
            ),
            "multiple_contents_discard": (
                "Unreachable here. Both templates describe the same physical track, so "
                "the Lustig content set is always a singleton and an out/in ambiguity is "
                "resolved by |corval| rather than discarded. status is only 'accepted' or "
                "'no_candidate'; n_candidate_maps == 2 flags the ambiguous events."
            ),
            "map_order": (
                "The joint space is [outbound | inbound] (ratemaps.lin_map_concat), "
                "so state_half 0 is outbound. MAP_SPECS orders 'in' first, deliberately "
                "decoupled from state_half, so that |corval| ties go to inbound exactly "
                "as in Lustig and Buzsaki."
            ),
            "min_prob_scale": (
                "The joint state space is only 2 * n_bins (80-92 states on this dataset), "
                "so an absolute min_prob of 0.01 is at or below uniform and excludes no "
                "windows at all. See each session's min_prob_effective and n_states. The "
                "binding per-window filter is min_active_neurons; use --min-prob-factor "
                "to set the threshold as a multiple of uniform instead."
            ),
            "numstep_floor": (
                "min_numstep requires at least that many qualifying windows, i.e. about "
                "55 ms at 20 ms / 5 ms. Table event sources are NOT length-filtered "
                "(Trajectory_Events durations reach ~45 ms and applying Buzsaki's 100 ms "
                "floor would discard much of the dataset's own curated set), so short "
                "events enter as candidates and leave as 'no_candidate'. "
                "n_events_below_numstep_floor counts them per session."
            ),
            "state_assignment": (
                "Decoded position is the global argmax over the concatenated posterior, "
                "matching Lustig and Buzsaki. This differs from "
                "call_directional_trajectories, which chooses the directional half by "
                "summed posterior mass; the two disagree on a few percent of windows."
            ),
            "forward_reverse": (
                "With two templates on a shared 0->track_len bin axis, forward_reverse is "
                "exactly (selected_map, sign(selected_corval)) re-expressed and carries "
                "no independent information."
            ),
            "dropped_columns": (
                "replay_arm / current_arm / local_remote have no analogue on a single "
                "linear track and are omitted rather than written as nulls. current_end "
                "records which track end the animal occupied at the event midpoint; the "
                "20 cm end zone is a LABEL only and gates nothing."
            ),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    return candidate_df, event_df, window_df, summary_df, sweep_df


# ── CLI ───────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("sessions", nargs="*", default=None,
                   help="Rat or Rat/Session selectors; default is every linear session.")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--rats", nargs="+", default=None)
    p.add_argument("--event-source", choices=PFEIFFER_EVENT_SOURCES, default="spike-density")
    p.add_argument("--bin-width-cm", type=float, default=ratemaps.BIN_WIDTH_CM)
    p.add_argument("--decode-tbin-s", type=float, default=TRADITIONAL_DECODE_GRID.tbin_s)
    p.add_argument("--decode-tbin-step-s", type=float, default=TRADITIONAL_DECODE_GRID.tbin_step_s)
    p.add_argument("--min-prob", type=float, default=0.01)
    p.add_argument("--min-prob-factor", type=float, default=None,
                   help="Express the peak-posterior threshold as a multiple of uniform.")
    p.add_argument("--min-active-neurons", type=int, default=4)
    p.add_argument("--min-abs-corval", type=float, default=0.8)
    p.add_argument("--min-numstep", type=int, default=8)
    p.add_argument("--max-event-duration-s", type=float, default=None)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def settings_from_args(args) -> DecodeSettings:
    return DecodeSettings(
        bin_width_cm=args.bin_width_cm,
        decode_tbin_s=args.decode_tbin_s,
        decode_tbin_step_s=args.decode_tbin_step_s,
        event_source=args.event_source,
        min_prob=args.min_prob,
        min_prob_factor=args.min_prob_factor,
        min_active_neurons=args.min_active_neurons,
        min_abs_corval=args.min_abs_corval,
        min_numstep=args.min_numstep,
        max_event_duration_s=args.max_event_duration_s,
    )


def eligible_sessions(args) -> list[tuple[str, str, Path]]:
    selectors = list(args.sessions or [])
    if args.rats:
        selectors.extend(args.rats)
    return loader.eligible_sessions(selectors or None, args.data_dir)


def main(argv=None) -> None:
    args = parse_args(argv)
    settings = settings_from_args(args)
    settings.validate()
    sessions = eligible_sessions(args)
    tag = run_tag(settings)
    results_dir = Path(args.results_root) / tag

    print(f"Pfeiffer traditional decode: {len(sessions)} session(s)")
    print(f"  event source: {settings.event_source}")
    print(f"  run tag:      {tag}")
    print(f"  results:      {results_dir}")
    for rat, sess, _ in sessions:
        print(f"    {rat}/{sess}")
    if args.dry_run:
        return

    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(process_session)(rat, sess, sdir, settings)
        for rat, sess, sdir in sessions
    )

    candidate_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    session_infos: list[dict[str, object]] = []
    for cands, events, windows, info in results:
        candidate_rows.extend(cands)
        event_rows.extend(events)
        window_rows.extend(windows)
        session_infos.append(info)

    candidate_df, event_df, window_df, summary_df, sweep_df = write_results(
        results_dir, candidate_rows, event_rows, window_rows, session_infos, settings
    )

    print(f"\n{len(event_df)} events, {int((event_df['status'] == 'accepted').sum())} accepted")
    cols = [
        "rat", "session", "n_bins", "n_states", "n_decode_units",
        "n_pbes", "n_pbes_decoded", "n_replay", "n_forward", "n_reverse",
    ]
    print(summary_df[[c for c in cols if c in summary_df]].to_string(index=False))
    print()
    print_criteria_sweep(sweep_df)
    print()
    print_acceptance_by_duration(event_df)
    print(f"\nwrote {results_dir}")
