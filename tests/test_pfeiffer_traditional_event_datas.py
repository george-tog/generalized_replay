"""The traditional events CSV loader and the event-alignment guards."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from replay_hmm.datasets.pfeiffer import traditional as ptd
from replay_hmm.decode.traditional import criteria


def _events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rat": ["Rat1", "Rat1", "Rat2"],
            "session": ["Linear1", "Linear1", "Linear1"],
            "session_pbe_index": [0, 2, 1],
            "source_row": [-1, -1, -1],
            "event_start_s": [10.0, 20.0, 30.0],
            "event_end_s": [10.2, 20.2, 30.2],
            "n_windows": [37, 41, 33],
            "status": ["accepted", "no_candidate", "accepted"],
        }
    )


@pytest.fixture
def events_csv(tmp_path) -> Path:
    path = tmp_path / "traditional_decode_events.csv"
    _events_frame().to_csv(path, index=False)
    (tmp_path / "traditional_decode_metadata.json").write_text(
        json.dumps({"event_source": "spike-density", "settings": {"decode_tbin_s": 0.02}})
    )
    return path


def test_default_keeps_only_accepted(events_csv):
    df = ptd.load_accepted_events(events_csv)
    assert len(df) == 2
    assert set(df["status"]) == {"accepted"}


def test_statuses_argument_widens_the_set(events_csv):
    df = ptd.load_accepted_events(events_csv, statuses=("accepted", "no_candidate"))
    assert len(df) == 3


def test_status_absent_from_the_file_raises(events_csv):
    # A typo'd verdict would otherwise yield an empty event set and surface much later.
    with pytest.raises(ValueError, match="no rows with status"):
        ptd.load_accepted_events(events_csv, statuses=("acepted",))


def test_missing_required_column_raises(tmp_path):
    frame = _events_frame().drop(columns=["n_windows"])
    path = tmp_path / "traditional_decode_events.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        ptd.load_accepted_events(path)


def test_rat_is_a_required_column(tmp_path):
    """Unlike Buzsaki, 'Linear1' is not unique -- it occurs under three rats."""
    assert "rat" in ptd.REQUIRED_EVENT_COLUMNS
    frame = _events_frame().drop(columns=["rat"])
    path = tmp_path / "traditional_decode_events.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        ptd.load_accepted_events(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        ptd.load_accepted_events(tmp_path / "absent.csv")


def test_loader_is_deterministically_ordered(events_csv):
    """The CV split shuffles a list, so a reordered CSV would change fold membership."""
    df = ptd.load_accepted_events(events_csv, statuses=("accepted", "no_candidate"))
    assert list(zip(df["rat"], df["session"], df["session_pbe_index"])) == [
        ("Rat1", "Linear1", 0), ("Rat1", "Linear1", 2), ("Rat2", "Linear1", 1)
    ]


def test_read_metadata_returns_the_event_source(events_csv):
    assert criteria.read_traditional_metadata(
        events_csv, missing_message=ptd.METADATA_MISSING_MESSAGE
    )["event_source"] == "spike-density"


def test_read_metadata_missing_file_raises(tmp_path):
    path = tmp_path / "traditional_decode_events.csv"
    _events_frame().to_csv(path, index=False)
    with pytest.raises(FileNotFoundError, match="metadata not found"):
        criteria.read_traditional_metadata(path, missing_message=ptd.METADATA_MISSING_MESSAGE)


# ── verify_event_alignment ────────────────────────────────────────────────────

def _events(starts, source_rows=None):
    return [
        {"session_pbe_index": i, "start_s": s,
         "source_row": -1 if source_rows is None else source_rows[i]}
        for i, s in enumerate(starts)
    ]


def _subset():
    df = _events_frame()
    return df[(df["rat"] == "Rat1") & (df["session"] == "Linear1")]


def test_alignment_accepts_matching_starts():
    ptd.verify_event_alignment(
        "Rat1", "Linear1", _events([10.0, 15.0, 20.0]), _subset(),
        event_source="spike-density", metadata_event_source="spike-density",
    )


def test_alignment_rejects_an_index_past_the_end():
    with pytest.raises(ValueError, match="out of range"):
        ptd.verify_event_alignment(
            "Rat1", "Linear1", _events([10.0, 15.0]), _subset(),
            event_source="spike-density",
        )


def test_alignment_rejects_a_start_time_mismatch():
    with pytest.raises(ValueError, match="drifted"):
        ptd.verify_event_alignment(
            "Rat1", "Linear1", _events([10.0, 15.0, 20.5]), _subset(),
            event_source="spike-density",
        )


def test_alignment_tolerance_is_one_microsecond():
    ptd.verify_event_alignment(
        "Rat1", "Linear1", _events([10.0 + 5e-7, 15.0, 20.0]), _subset(),
        event_source="spike-density",
    )
    with pytest.raises(ValueError, match="drifted"):
        ptd.verify_event_alignment(
            "Rat1", "Linear1", _events([10.0 + 2e-6, 15.0, 20.0]), _subset(),
            event_source="spike-density",
        )


def test_alignment_rejects_a_mismatched_event_source():
    """The four sources overlap heavily, so an index match against the wrong one could
    coincidentally pass the start-time check. Comparing the source up front is cheaper."""
    with pytest.raises(ValueError, match="event source"):
        ptd.verify_event_alignment(
            "Rat1", "Linear1", _events([10.0, 15.0, 20.0]), _subset(),
            event_source="swr-gated", metadata_event_source="spike-density",
        )


def test_alignment_rejects_a_source_row_mismatch():
    df = _subset().copy()
    df["source_row"] = [4, 9]
    with pytest.raises(ValueError, match="source_row"):
        ptd.verify_event_alignment(
            "Rat1", "Linear1",
            _events([10.0, 15.0, 20.0], source_rows=[4, 5, 6]), df,
            event_source="trajectory-events",
        )


def test_alignment_only_checks_the_named_session():
    """A Rat2 row must not be indexed into Rat1's event list."""
    df = _events_frame()
    ptd.verify_event_alignment(
        "Rat1", "Linear1", _events([10.0, 15.0, 20.0]), df,
        event_source="spike-density",
    )
