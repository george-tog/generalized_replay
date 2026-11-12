"""The shared two-map collectors read each dataset's decoded dicts through its view."""

from pathlib import Path

import numpy as np
import pandas as pd

from replay_hmm.datasets import two_map_event_sets as tmes
from replay_hmm.datasets.buzsaki.session import BUZSAKI_DECODED_VIEW
from replay_hmm.datasets.pfeiffer.session import PFEIFFER_DECODED_VIEW


def _counts(n_cells, n_wins, offset):
    return (np.arange(n_cells * n_wins).reshape(n_cells, n_wins) + offset).astype(np.uint16)


def _buzsaki():
    return [
        {"rate_traj": _counts(3, 6, 0), "session_pbe_index": 4,
         "trajectories": [{"win_slice": slice(0, 3)}, {"win_slice": slice(3, 4)}, {"win_slice": slice(4, 6)}]},
        {"rate_traj": _counts(3, 1, 100), "session_pbe_index": 9, "trajectories": []},
    ]


def _pfeiffer():
    return [
        {"spike_traj": _counts(2, 7, 0), "event": {"session_pbe_index": 2},
         "segments": {0: [(4, 7)], 1: [(0, 3), (3, 4)]}},
    ]


def test_buzsaki_spans_follow_trajectory_order_and_drop_short_ones():
    handles = tmes.collect_event_handles(_buzsaki(), BUZSAKI_DECODED_VIEW)
    assert [(h["event_id"], h["s"], h["e"]) for h in handles] == [(4, 0, 3), (4, 4, 6)]
    first = handles[0]["spike_traj"]
    assert first.dtype == np.int32 and first.shape == (3, 3)
    np.testing.assert_array_equal(first, _buzsaki()[0]["rate_traj"][:, 0:3].T)


def test_pfeiffer_spans_are_map_0_then_map_1():
    handles = tmes.collect_event_handles(_pfeiffer(), PFEIFFER_DECODED_VIEW)
    assert [(h["s"], h["e"]) for h in handles] == [(4, 7), (0, 3)]
    assert all(h["event_id"] == 2 and h["index"] == 0 for h in handles)


def test_whole_events_respect_the_length_floor():
    datas = tmes.collect_whole_event_datas(_buzsaki(), BUZSAKI_DECODED_VIEW)
    assert [d.shape for d in datas] == [(6, 3)]
    assert tmes.collect_mb_traj_datas(_buzsaki(), BUZSAKI_DECODED_VIEW, min_len=1)[1].shape == (1, 3)


def test_traditional_provenance(tmp_path):
    assert set(tmes.traditional_provenance(None, (), None, None, [])) == {
        "traditional_events_csv", "traditional_run_tag", "traditional_statuses",
        "traditional_decode_settings", "traditional_event_stats",
    }
    csv = tmp_path / "run_tag_x" / "events.csv"
    events = pd.DataFrame({"n_windows": [5, 7, 9]})
    datas = [np.zeros((3, 2)), np.zeros((4, 2))]
    got = tmes.traditional_provenance(csv, ("accepted",), {"settings": {"a": 1}}, events, datas)
    assert got["traditional_events_csv"] == str(Path(csv).resolve())
    assert got["traditional_run_tag"] == "run_tag_x"
    assert got["traditional_event_stats"] == {
        "n_events_for_session": 3, "n_used": 2, "n_dropped_short": 1,
        "n_windows_csv_total": 21, "n_bins_total": 7,
    }
