"""The traditional-criteria event set as an input to the Buzsaki HMM kernel comparison.

Covers `load_accepted_events` -- the loader behind `replay-hmm fit --dataset buzsaki
--traditional-events-csv` -- and the alignment guard that keeps the `session_pbe_index` join
honest (`decode.traditional.criteria.verify_event_alignment` with the Buzsaki fit's
key="session_pbe_index").
"""

from pathlib import Path

import pandas as pd
import pytest


from replay_hmm.datasets.buzsaki import traditional as btd
from replay_hmm.decode.traditional import criteria


def _events_frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "session": "Achilles_10252013",
        "status": "accepted",
        "n_windows": 30,
        "event_start_s": 100.0,
        "event_end_s": 100.2,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _write(tmp_path: Path, df: pd.DataFrame) -> Path:
    out = tmp_path / "traditional_decode_events.csv"
    df.to_csv(out, index=False)
    return out


# --------------------------------------------------------------------------------------
# load_accepted_events
# --------------------------------------------------------------------------------------


def test_default_keeps_only_accepted(tmp_path):
    df = _events_frame([
        {"session_pbe_index": 0, "status": "accepted"},
        {"session_pbe_index": 1, "status": "no_candidate"},
    ])
    out = btd.load_accepted_events(_write(tmp_path, df))
    assert out["status"].tolist() == ["accepted"]
    assert out["session_pbe_index"].tolist() == [0]


def test_statuses_argument_widens_the_set(tmp_path):
    df = _events_frame([
        {"session_pbe_index": 0, "status": "accepted"},
        {"session_pbe_index": 1, "status": "no_candidate"},
    ])
    out = btd.load_accepted_events(
        _write(tmp_path, df), statuses=("accepted", "no_candidate")
    )
    assert len(out) == 2


def test_status_absent_from_the_file_raises(tmp_path):
    # A typo'd verdict would otherwise produce an empty event set and surface much later as
    # a confusing "need >= 2 events for CV" failure.
    df = _events_frame([{"session_pbe_index": 0}])
    with pytest.raises(ValueError, match="no rows with status"):
        btd.load_accepted_events(_write(tmp_path, df), statuses=("replay",))


def test_missing_required_column_raises(tmp_path):
    df = _events_frame([{"session_pbe_index": 0}]).drop(columns=["event_end_s"])
    with pytest.raises(ValueError, match="event_end_s"):
        btd.load_accepted_events(_write(tmp_path, df))


def test_rat_is_backfilled_from_the_session_name(tmp_path):
    df = _events_frame([{"session_pbe_index": 0}])
    out = btd.load_accepted_events(_write(tmp_path, df))
    assert out["rat"].tolist() == ["Achilles"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        btd.load_accepted_events(tmp_path / "nope.csv")


# --------------------------------------------------------------------------------------
# Alignment guard
# --------------------------------------------------------------------------------------


def _pbes(starts: list[float], indices: list[int] | None = None) -> list[dict]:
    """Candidate events as the detectors emit them: carrying their own join key.

    `session_pbe_index` is explicit rather than implied by list position, because the
    alignment guard keys on the carried value. That matters for an event subset, which
    drops spike-density events WITHOUT renumbering, so position and index diverge.
    """
    if indices is None:
        indices = list(range(len(starts)))
    return [
        {"start_s": s, "end_s": s + 0.2, "duration_s": 0.2, "session_pbe_index": i}
        for s, i in zip(starts, indices)
    ]


def test_alignment_accepts_matching_starts():
    pbes = _pbes([10.0, 20.0, 30.0])
    events = _events_frame([
        {"session_pbe_index": 0, "event_start_s": 10.0},
        {"session_pbe_index": 2, "event_start_s": 30.0},
    ])
    criteria.verify_event_alignment("S", pbes, events, key="session_pbe_index")  # must not raise


def test_alignment_rejects_an_index_past_the_end():
    pbes = _pbes([10.0, 20.0])
    events = _events_frame([{"session_pbe_index": 5, "event_start_s": 10.0}])
    with pytest.raises(ValueError, match="not among the 2 events"):
        criteria.verify_event_alignment("S", pbes, events, key="session_pbe_index")


def test_alignment_rejects_a_start_time_mismatch():
    # The index is only meaningful if detect_pbes reproduces the decode run; an index that
    # silently points at a different burst would give a confidently wrong event set.
    pbes = _pbes([10.0, 20.0])
    events = _events_frame([{"session_pbe_index": 1, "event_start_s": 19.99}])
    with pytest.raises(ValueError, match="reproducing the decode run"):
        criteria.verify_event_alignment("S", pbes, events, key="session_pbe_index")


def test_alignment_keys_on_the_carried_index_not_on_position():
    """A subset drops events without renumbering, so position != session_pbe_index.

    Positional lookup would fetch the burst at 30.0 s for index 2 here and pass by luck;
    keying on the carried value is what makes the start-time check meaningful.
    """
    pbes = _pbes([10.0, 30.0], indices=[0, 2])
    ok = _events_frame([{"session_pbe_index": 2, "event_start_s": 30.0}])
    criteria.verify_event_alignment("S", pbes, ok, key="session_pbe_index")

    missing = _events_frame([{"session_pbe_index": 1, "event_start_s": 20.0}])
    with pytest.raises(ValueError, match="not among the 2 events"):
        criteria.verify_event_alignment("S", pbes, missing, key="session_pbe_index")


def test_alignment_tolerance_is_one_microsecond():
    pbes = _pbes([10.0])
    ok = _events_frame([{"session_pbe_index": 0, "event_start_s": 10.0 + 9e-7}])
    criteria.verify_event_alignment("S", pbes, ok, key="session_pbe_index")
    bad = _events_frame([{"session_pbe_index": 0, "event_start_s": 10.0 + 2e-6}])
    with pytest.raises(ValueError):
        criteria.verify_event_alignment("S", pbes, bad, key="session_pbe_index")
