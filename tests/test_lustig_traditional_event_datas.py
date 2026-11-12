"""The traditional-criteria event set as an input to the HMM kernel comparison.

Covers `datasets.lustig.traditional_io.load_traditional_events` and
`collect_traditional_event_datas`: the path that lets `replay-hmm fit --dataset lustig
--traditional-events-csv` fit the kernels on the events the traditional replay criteria accepted, rebinned
on the model-comparison decode grid.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from replay_hmm.datasets.lustig import traditional_io
from replay_hmm.decode import grid
from replay_hmm.decode import config


LFP_SAMP_RATE = grid.LFP_SAMP_RATE
SPK_SAMP_RATE = grid.SPK_SAMP_RATE

# 10 ms / 10 ms: the grid the kernels are compared on, and non-overlapping so an event's
# bin count and its duration coincide.
CFG_10_10 = config.DecodeConfig(decode_tbin_s=0.010, decode_tbin_step_s=0.010)


def _events_frame(rows: list[dict]) -> pd.DataFrame:
    base = {"status": "accepted", "rat": "Rat", "session": "rec01"}
    return pd.DataFrame([{**base, **row} for row in rows])


def _write_events_csv(tmp_path: Path, df: pd.DataFrame, *, settings: dict | None = None) -> Path:
    import json

    out = tmp_path / "traditional_decode_events.csv"
    df.to_csv(out, index=False)
    meta = {"settings": settings or {"decode_tbin_s": 0.020, "decode_tbin_step_s": 0.005}}
    (tmp_path / "traditional_decode_metadata.json").write_text(json.dumps(meta))
    return out


class _Bundle:
    """The fields `collect_traditional_event_datas` reads off a SessionBundle."""

    def __init__(self, trial_starts, trial_ends, n_units=3, cfg=CFG_10_10):
        self.rat = "Rat"
        self.session = "rec01"
        self.cfg = cfg
        self.trial_starts_lfp = np.asarray(trial_starts, dtype=int)
        self.trial_ends_lfp = np.asarray(trial_ends, dtype=int)
        self.trial_rate_trajs = {}
        for i, (ts, te) in enumerate(zip(trial_starts, trial_ends), start=1):
            n_bins = grid.decode_window_grid(ts, te, cfg)[0].size
            # Column j holds the value j, so a slice can be checked against its bin indices.
            self.trial_rate_trajs[i] = np.tile(
                np.arange(n_bins, dtype=np.uint16), (n_units, 1)
            )


def _lfp_of_bin_edge(sample_20k: float) -> int:
    return int(np.ceil(sample_20k / SPK_SAMP_RATE * LFP_SAMP_RATE))


# --------------------------------------------------------------------------------------
# load_traditional_events
# --------------------------------------------------------------------------------------


def test_loader_keeps_only_the_requested_statuses(tmp_path):
    df = _events_frame([
        {"trial": 1, "event_start_lfp": 100, "event_end_lfp": 200, "status": "accepted"},
        {"trial": 1, "event_start_lfp": 300, "event_end_lfp": 400, "status": "no_candidate"},
        {"trial": 2, "event_start_lfp": 500, "event_end_lfp": 600, "status": "discarded"},
    ])
    csv = _write_events_csv(tmp_path, df)

    accepted = traditional_io.load_traditional_events(csv)
    assert accepted["status"].tolist() == ["accepted"]

    both = traditional_io.load_traditional_events(csv, statuses=("accepted", "discarded"))
    assert sorted(both["status"].tolist()) == ["accepted", "discarded"]


def test_loader_keeps_wheel_winners(tmp_path):
    # Unlike the arm-dropout null, which needs an arm to drop, the kernels' state space
    # contains the wheel map -- dropping wheel winners here would silently shrink the fit set.
    df = _events_frame([
        {"trial": 1, "event_start_lfp": 100, "event_end_lfp": 200, "replay_arm": None,
         "selected_map_family": "wheel"},
        {"trial": 1, "event_start_lfp": 300, "event_end_lfp": 400, "replay_arm": "Left",
         "selected_map_family": "directional"},
    ])
    assert len(traditional_io.load_traditional_events(_write_events_csv(tmp_path, df))) == 2


def test_loader_rejects_a_status_present_nowhere(tmp_path):
    df = _events_frame([{"trial": 1, "event_start_lfp": 100, "event_end_lfp": 200}])
    csv = _write_events_csv(tmp_path, df)
    # A typo'd verdict would otherwise produce an empty fit set and a confusing "need at
    # least 2 events" failure hours later.
    with pytest.raises(ValueError, match="no rows with status"):
        traditional_io.load_traditional_events(csv, statuses=("replay",))


def test_loader_requires_the_time_columns(tmp_path):
    df = _events_frame([{"trial": 1, "event_start_lfp": 100, "event_end_lfp": 200}])
    csv = _write_events_csv(tmp_path, df.drop(columns=["event_end_lfp"]))
    with pytest.raises(ValueError, match="event_end_lfp"):
        traditional_io.load_traditional_events(csv)


def test_loader_order_is_deterministic_regardless_of_csv_row_order(tmp_path):
    # The CV split shuffles a LIST, so row order would otherwise change fold membership at a
    # fixed --split-seed.
    rows = [
        {"trial": 2, "event_start_lfp": 900, "event_end_lfp": 1000},
        {"trial": 1, "event_start_lfp": 300, "event_end_lfp": 400},
        {"trial": 1, "event_start_lfp": 100, "event_end_lfp": 200},
    ]
    reversed_dir = tmp_path / "reversed"
    reversed_dir.mkdir()
    forward = traditional_io.load_traditional_events(_write_events_csv(tmp_path, _events_frame(rows)))
    reverse = traditional_io.load_traditional_events(
        _write_events_csv(reversed_dir, _events_frame(rows[::-1]))
    )
    assert forward["event_start_lfp"].tolist() == [100, 300, 900]
    assert reverse["event_start_lfp"].tolist() == [100, 300, 900]


# --------------------------------------------------------------------------------------
# collect_traditional_event_datas
# --------------------------------------------------------------------------------------


def test_event_spanning_whole_bins_yields_exactly_those_columns():
    bundle = _Bundle([0], [40_000])
    starts, ends, _ = grid.decode_window_grid(0, 40_000, CFG_10_10)
    start_lfp = _lfp_of_bin_edge(starts[3])
    end_lfp = int(np.floor(ends[7] / SPK_SAMP_RATE * LFP_SAMP_RATE))

    events = _events_frame([
        {"trial": 1, "event_start_lfp": start_lfp, "event_end_lfp": end_lfp}
    ])
    datas, stats = traditional_io.collect_traditional_event_datas(bundle, events)

    assert len(datas) == 1
    # (T_event, n_units), and column j of rate_traj holds j, so this reads back the bin indices.
    assert datas[0].shape == (5, 3)
    assert datas[0][:, 0].tolist() == [3, 4, 5, 6, 7]
    assert datas[0].dtype == np.int32
    assert stats["n_used"] == 1 and stats["n_bins_total"] == 5


def test_other_sessions_rows_are_ignored():
    bundle = _Bundle([0], [40_000])
    starts, ends, _ = grid.decode_window_grid(0, 40_000, CFG_10_10)
    span = {
        "event_start_lfp": _lfp_of_bin_edge(starts[3]),
        "event_end_lfp": int(np.floor(ends[7] / SPK_SAMP_RATE * LFP_SAMP_RATE)),
    }
    events = _events_frame([
        {"trial": 1, **span},
        {"trial": 1, "rat": "Other", **span},
        {"trial": 1, "session": "rec99", **span},
    ])
    datas, stats = traditional_io.collect_traditional_event_datas(bundle, events)
    assert len(datas) == 1 and stats["n_events_for_session"] == 1


def test_trial_outside_the_session_raises_rather_than_dropping_events():
    # An understated event set is indistinguishable from a genuinely small one, so a lap
    # convention mismatch must fail loudly.
    bundle = _Bundle([0], [40_000])
    events = _events_frame([{"trial": 7, "event_start_lfp": 100, "event_end_lfp": 200}])
    with pytest.raises(ValueError, match="outside 1..1"):
        traditional_io.collect_traditional_event_datas(bundle, events)


def test_bundle_without_lap_boundaries_raises():
    bundle = _Bundle([0], [40_000])
    bundle.trial_starts_lfp = None
    events = _events_frame([{"trial": 1, "event_start_lfp": 100, "event_end_lfp": 200}])
    with pytest.raises(ValueError, match="trial_starts_lfp"):
        traditional_io.collect_traditional_event_datas(bundle, events)


def test_grid_disagreeing_with_the_bundles_counts_raises():
    # The bundle's counts were binned on ITS config; rebuilding the grid at another one would
    # slice the wrong columns, so the sizes are checked instead of trusted.
    bundle = _Bundle([0], [40_000])
    bundle.cfg = config.DecodeConfig(decode_tbin_s=0.020, decode_tbin_step_s=0.020)
    events = _events_frame([{"trial": 1, "event_start_lfp": 100, "event_end_lfp": 200}])
    with pytest.raises(ValueError, match="different config"):
        traditional_io.collect_traditional_event_datas(bundle, events)


def test_bin_count_tracks_event_duration_on_the_nonoverlapping_grid():
    bundle = _Bundle([0], [200_000])
    starts, ends, _ = grid.decode_window_grid(0, 200_000, CFG_10_10)
    rng = np.random.default_rng(11)
    rows = []
    for _ in range(40):
        first, last = sorted(rng.integers(500, 12_000, size=2).tolist())
        if last - first < 50:
            continue
        rows.append({"trial": 1, "event_start_lfp": int(first), "event_end_lfp": int(last)})
    assert rows != sorted(rows, key=lambda r: r["event_start_lfp"])  # the sort must matter
    # The collector walks the frame in its stored order; in production that order is the one
    # load_traditional_events imposes, so sort here the same way rather than testing an order
    # the pipeline never produces.
    events = _events_frame(sorted(rows, key=lambda r: r["event_start_lfp"]))

    datas, stats = traditional_io.collect_traditional_event_datas(bundle, events)
    # collect_traditional_event_datas walks the frame in its stored order, so pair each
    # returned array with the surviving rows sorted the same way.
    kept = [
        row for row in sorted(rows, key=lambda r: r["event_start_lfp"])
        if round((row["event_end_lfp"] - row["event_start_lfp"]) / LFP_SAMP_RATE / 0.010) >= 2
    ]
    assert len(datas) == len(kept) == stats["n_used"]
    for data, row in zip(datas, kept):
        duration_s = (row["event_end_lfp"] - row["event_start_lfp"]) / LFP_SAMP_RATE
        # Half-coverage rounding at each edge, so at most one bin either way.
        assert abs(data.shape[0] - round(duration_s / 0.010)) <= 1
