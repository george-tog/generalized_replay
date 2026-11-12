"""Each two-map dataset's `rebuild.rebuild_events`, the one saved-fit event rebuild the kernel
evidence, the kernel PPC and the event viewer share (and, on Pfeiffer, the jump distribution).

What these pin, with no real data (detection and the decode loop are stubbed, so the guards and
the plumbing around them are what is under test):
  * the event setting and decode grid come from the fit summary, and the decode reads the maps,
    bounds and cell ids of the fit's inputs npz;
  * a provenance block that differs from the fit's raises;
  * the decoded-event and trajectory counts are held to the summary's `n_decoded_pbes` and
    `n_mb_trajectory_events`; on a traditional fit, to its table's events only;
  * a traditional table restricts the decode with `restrict_to_accepted` and not without it,
    and a traditional fit without its table is refused;
  * Pfeiffer jumps decode on the fit's maps in every fit-set x jump-set cell, with the count
    guard scoped to the fit's own events.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from replay_hmm.datasets import two_map_event_sets as tmes
from replay_hmm.datasets.buzsaki import rebuild as brb
from replay_hmm.datasets.pfeiffer import rebuild as prb

N_BINS = 6
N_UNITS = 3
STARTS = (1.0, 2.0, 3.0, 4.0)
TRADITIONAL_IDS = (1, 3)


def _inputs():
    rng = np.random.default_rng(0)
    return {
        "lin_map_concat": rng.uniform(0.5, 20.0, size=(N_UNITS, 2 * N_BINS)),
        "lin_map_bounds": np.array([0, N_BINS, 2 * N_BINS]),
        "cell_ids": np.array([7, 11, 13]),
    }


def _candidates():
    return [{"session_pbe_index": i, "start_s": s, "end_s": s + 0.1, "source_row": i}
            for i, s in enumerate(STARTS)]


def _rows(ids=TRADITIONAL_IDS, **extra):
    return pd.DataFrame({
        "session_pbe_index": list(ids),
        "event_start_s": [STARTS[i] for i in ids],
        **{k: [v] * len(ids) for k, v in extra.items()},
    })


def _json(block):
    return json.loads(json.dumps(block, default=float))


class Recorder:
    """What the stubbed detection and decode were handed."""

    def __init__(self):
        self.calls = {}


# ── Buzsaki ───────────────────────────────────────────────────────────────────

def _buzsaki_summary(**overrides):
    summary = {"event_source": "mb_traj", "event_detection_source": "spike-density",
               "decode_tbin_s": 0.003, "decode_tbin_step_s": 0.003, "hmm_bin_width_m": 0.02,
               "n_decoded_pbes": 4, "n_mb_trajectory_events": 4}
    summary.update(overrides)
    setting = brb.pbe_setting(summary)
    from replay_hmm.datasets.buzsaki import events as buzsaki_events

    summary.setdefault("pbe_provenance", _json(setting.pbe_provenance()))
    summary.setdefault("traj_provenance", _json(setting.traj_provenance()))
    summary.setdefault("event_source_provenance", _json(buzsaki_events.event_source_provenance(setting)))
    return summary


@pytest.fixture
def buzsaki(monkeypatch):
    rec = Recorder()

    def candidate_events(data, setting, *, session_name, decode_grid):
        rec.calls["candidates"] = (setting, session_name, decode_grid)
        return _candidates(), {"baseline_mean": 1.0}

    def iter_decoded_pbes(pbes, spikes, cell_ids, cache, n_bins, *, decode_grid,
                          accepted_indices=None, session_name=None, carry_event_ids=False):
        rec.calls["decode"] = dict(cell_ids=cell_ids, cache=cache, n_bins=n_bins,
                                   decode_grid=decode_grid, accepted=accepted_indices,
                                   carry_event_ids=carry_event_ids)
        for pbe in pbes:
            idx = int(pbe["session_pbe_index"])
            if accepted_indices is not None and idx not in accepted_indices:
                continue
            yield {"session_pbe_index": idx, "epoch": "maze",
                   "rate_traj": np.zeros((len(cell_ids), 5), dtype=int),
                   "trajectories": [{"dir": "out", "win_slice": slice(0, 3)}]}

    monkeypatch.setattr(brb, "candidate_events", candidate_events)
    monkeypatch.setattr(brb, "iter_decoded_pbes", iter_decoded_pbes)
    monkeypatch.setattr(brb, "pyramidal_spike_times", lambda data, cell_ids: {c: np.empty(0) for c in cell_ids})
    return rec


def _buzsaki_rebuild(summary, **kw):
    return brb.rebuild_events("Achilles", "Achilles_10252013", summary, _inputs(),
                              label="Achilles_10252013", data={"stub": True}, **kw)


def test_buzsaki_setting_grid_and_maps_come_from_the_fit(buzsaki):
    summary = _buzsaki_summary()
    rebuilt = _buzsaki_rebuild(summary)
    setting, session, grid = buzsaki.calls["candidates"]
    assert setting.event_source == "spike-density" and session == "Achilles_10252013"
    assert (grid.tbin_s, grid.tbin_step_s) == (0.003, 0.003)
    dec = buzsaki.calls["decode"]
    assert dec["decode_grid"] == grid and dec["n_bins"] == N_BINS and dec["carry_event_ids"]
    assert dec["cell_ids"] == [7, 11, 13]
    inputs = _inputs()
    expected = np.log(np.clip(inputs["lin_map_concat"] * 0.003, 1e-12, None))
    np.testing.assert_array_equal(dec["cache"].logpfmaps, expected)
    assert dec["accepted"] is None
    assert [d["session_pbe_index"] for d in rebuilt.decoded] == [0, 1, 2, 3]
    assert rebuilt.n_candidates == 4 and rebuilt.data == {"stub": True}
    assert rebuilt.detection_diag == {"baseline_mean": 1.0}
    np.testing.assert_array_equal(rebuilt.cell_ids, [7, 11, 13])
    assert rebuilt.traditional_ids is None


def test_buzsaki_provenance_mismatch_raises(buzsaki):
    summary = _buzsaki_summary()
    summary["pbe_provenance"] = {**summary["pbe_provenance"], "peak_z": -1.0}
    with pytest.raises(ValueError, match="pbe_provenance differs"):
        _buzsaki_rebuild(summary)


@pytest.mark.parametrize("key,value", [("n_decoded_pbes", 5), ("n_mb_trajectory_events", 3)])
def test_buzsaki_count_mismatch_raises(buzsaki, key, value):
    with pytest.raises(ValueError, match=key):
        _buzsaki_rebuild(_buzsaki_summary(**{key: value}))


def test_buzsaki_counts_unrecorded_are_not_checked(buzsaki):
    summary = _buzsaki_summary()
    del summary["n_decoded_pbes"], summary["n_mb_trajectory_events"]
    assert len(_buzsaki_rebuild(summary).decoded) == 4


def test_buzsaki_traditional_restriction(buzsaki):
    summary = _buzsaki_summary(event_source="traditional", n_decoded_pbes=2, n_mb_trajectory_events=2)
    restricted = _buzsaki_rebuild(summary, traditional_rows=_rows())
    assert buzsaki.calls["decode"]["accepted"] == {1, 3}
    assert [d["session_pbe_index"] for d in restricted.decoded] == [1, 3]
    assert restricted.traditional_ids == {1, 3}

    # Unrestricted (the viewer): every event decodes, and the count guard still holds the
    # table's events to the fit's counts.
    everything = _buzsaki_rebuild(summary, traditional_rows=_rows(), restrict_to_accepted=False)
    assert buzsaki.calls["decode"]["accepted"] is None
    assert [d["session_pbe_index"] for d in everything.decoded] == [0, 1, 2, 3]
    assert everything.traditional_ids == {1, 3}

    with pytest.raises(ValueError, match="n_decoded_pbes"):
        _buzsaki_rebuild(summary, traditional_rows=_rows(ids=(1,)), restrict_to_accepted=False)


def test_buzsaki_traditional_fit_needs_its_table(buzsaki):
    with pytest.raises(ValueError, match="traditional-replay fit"):
        _buzsaki_rebuild(_buzsaki_summary(event_source="traditional"))


def test_buzsaki_table_rows_are_aligned(buzsaki):
    rows = _rows()
    rows.loc[0, "event_start_s"] += 0.5
    with pytest.raises(ValueError, match="not reproducing the decode run"):
        _buzsaki_rebuild(_buzsaki_summary(), traditional_rows=rows)


def test_buzsaki_cross_scope_table_skips_the_count_guard(buzsaki):
    # A table the (mb_traj) fit was not trained on restricts the decode to events the fit's
    # counts do not describe, so the guard is not applied.
    rebuilt = _buzsaki_rebuild(_buzsaki_summary(), traditional_rows=_rows())
    assert [d["session_pbe_index"] for d in rebuilt.decoded] == [1, 3]


def test_buzsaki_traditional_rows_reads_the_fit_statuses(tmp_path):
    csv = tmp_path / "events.csv"
    pd.DataFrame({
        "session": ["Achilles_10252013", "Achilles_10252013", "Achilles_10252013", "Buddy_06272013"],
        "session_pbe_index": [3, 1, 2, 0],
        "event_start_s": [4.0, 2.0, 3.0, 1.0],
        "event_end_s": [4.1, 2.1, 3.1, 1.1],
        "status": ["accepted", "accepted", "rejected", "accepted"],
        "epoch": ["maze"] * 4,
        "n_windows": [10] * 4,
    }).to_csv(csv, index=False)
    rows = brb.traditional_rows(csv, "Achilles", "Achilles_10252013", {})
    assert rows["session_pbe_index"].tolist() == [1, 3]
    rows = brb.traditional_rows(csv, "Achilles", "Achilles_10252013",
                                {"traditional_statuses": ["accepted", "rejected"]})
    assert rows["session_pbe_index"].tolist() == [1, 2, 3]


# ── Pfeiffer ──────────────────────────────────────────────────────────────────

def _pfeiffer_summary(**overrides):
    summary = {"event_source": "mb_traj", "event_detection_source": "spike-density",
               "decode_tbin_s": 0.003, "decode_tbin_step_s": 0.003, "hmm_bin_width_cm": 5.0,
               "n_decoded_pbes": 4, "n_mb_trajectory_events": 8}
    summary.update(overrides)
    setting = prb.run_setting(summary)
    summary.setdefault("pbe_provenance", _json(setting.pbe_provenance()))
    summary.setdefault("traj_provenance", _json(setting.traj_provenance()))
    return summary


@pytest.fixture
def pfeiffer(monkeypatch):
    rec = Recorder()

    def candidate_events(data, setting):
        rec.calls["candidates"] = setting
        return _candidates(), {"grids": []}

    def iter_decoded_events(events, spikes, cell_ids, cache, bounds, setting, *,
                            accepted_indices=None, raise_if_accepted_undecodable=False, label=None):
        rec.calls["decode"] = dict(cell_ids=cell_ids, cache=cache, bounds=bounds, setting=setting,
                                   accepted=accepted_indices)
        for ev in events:
            if accepted_indices is not None and int(ev["session_pbe_index"]) not in accepted_indices:
                continue
            yield {"event": ev, "spike_traj": np.zeros((len(cell_ids), 5), dtype=int),
                   "segments": {0: [(0, 3)], 1: [(3, 5)]}}

    monkeypatch.setattr(prb.pfeiffer_events, "candidate_events", candidate_events)
    monkeypatch.setattr(prb, "iter_decoded_events", iter_decoded_events)
    monkeypatch.setattr(prb.loader, "cell_spike_times", lambda data: {})
    return rec


def _pfeiffer_rebuild(summary, inputs=None, **kw):
    return prb.rebuild_events("Rat1", "Linear1", summary, inputs or _inputs(),
                              label="Rat1/Linear1", data={"stub": True}, **kw)


def test_pfeiffer_setting_and_maps_come_from_the_fit(pfeiffer):
    rebuilt = _pfeiffer_rebuild(_pfeiffer_summary())
    setting = pfeiffer.calls["candidates"]
    assert setting.event_source == "spike-density"
    assert (setting.tbin_s, setting.tbin_step_s, setting.bin_width_cm) == (0.003, 0.003, 5.0)
    dec = pfeiffer.calls["decode"]
    assert dec["setting"] is setting
    np.testing.assert_array_equal(dec["cell_ids"], [7, 11, 13])
    np.testing.assert_array_equal(dec["bounds"], [0, N_BINS, 2 * N_BINS])
    expected = np.log(np.clip(_inputs()["lin_map_concat"] * 0.003, 1e-12, None))
    np.testing.assert_array_equal(dec["cache"].logpfmaps, expected)
    assert not dec["cache"].include_wheel
    assert rebuilt.setting is setting and rebuilt.n_candidates == 4
    assert [d["event"]["session_pbe_index"] for d in rebuilt.decoded] == [0, 1, 2, 3]


def test_pfeiffer_provenance_mismatch_raises(pfeiffer):
    summary = _pfeiffer_summary()
    summary["traj_provenance"] = {**summary["traj_provenance"], "trajectory_mass_thresh": 0.9}
    with pytest.raises(ValueError, match="traj_provenance differs"):
        _pfeiffer_rebuild(summary)


def test_pfeiffer_saved_event_times_are_checked(pfeiffer):
    inputs = {**_inputs(), "pbe_events_s": np.array([[s, s + 0.1] for s in STARTS[:3]])}
    with pytest.raises(ValueError, match="re-detected 4 candidate events but the fit recorded 3"):
        _pfeiffer_rebuild(_pfeiffer_summary(), inputs=inputs)


@pytest.mark.parametrize("key,value", [("n_decoded_pbes", 3), ("n_mb_trajectory_events", 7)])
def test_pfeiffer_count_mismatch_raises(pfeiffer, key, value):
    with pytest.raises(ValueError, match=key):
        _pfeiffer_rebuild(_pfeiffer_summary(**{key: value}))


def test_pfeiffer_traditional_restriction(pfeiffer):
    summary = _pfeiffer_summary(event_source="traditional", n_decoded_pbes=2, n_mb_trajectory_events=4)
    rows = _rows(rat="Rat1", session="Linear1")
    restricted = _pfeiffer_rebuild(summary, traditional_rows=rows)
    assert pfeiffer.calls["decode"]["accepted"] == {1, 3}
    assert [d["event"]["session_pbe_index"] for d in restricted.decoded] == [1, 3]

    everything = _pfeiffer_rebuild(summary, traditional_rows=rows, restrict_to_accepted=False)
    assert pfeiffer.calls["decode"]["accepted"] is None
    assert [d["event"]["session_pbe_index"] for d in everything.decoded] == [0, 1, 2, 3]
    assert everything.traditional_ids == {1, 3}

    with pytest.raises(ValueError, match="traditional-replay fit"):
        _pfeiffer_rebuild(summary)


def test_pfeiffer_traditional_rows_refuse_another_detector(tmp_path):
    run = tmp_path / "trad_tag"
    run.mkdir()
    csv = run / "traditional_decode_events.csv"
    pd.DataFrame({
        "rat": ["Rat1", "Rat1", "Rat2"], "session": ["Linear1", "Linear1", "Linear1"],
        "session_pbe_index": [3, 1, 0], "event_start_s": [4.0, 2.0, 1.0],
        "event_end_s": [4.1, 2.1, 1.1], "status": ["accepted"] * 3, "n_windows": [10] * 3,
    }).to_csv(csv, index=False)
    meta_path = run / "traditional_decode_metadata.json"
    meta_path.write_text(json.dumps({"event_source": "ripple"}))
    summary = {"event_detection_source": "spike-density"}
    with pytest.raises(ValueError, match="classified on event detection source 'ripple'"):
        prb.traditional_rows(csv, "Rat1", "Linear1", summary)
    meta_path.write_text(json.dumps({"event_source": "spike-density"}))
    assert prb.traditional_rows(csv, "Rat1", "Linear1", summary)["session_pbe_index"].tolist() == [1, 3]


# ── Pfeiffer jumps: the fit-set x jump-set cells ─────────────────────────────

def _jump_events(pfeiffer, monkeypatch, summary, *, jump_set, csv="events.csv"):
    import argparse

    from replay_hmm.datasets.pfeiffer import jumps as pj

    monkeypatch.setattr(pj.loader, "load_session_data", lambda sdir: {"stub": True})
    monkeypatch.setattr(prb, "traditional_rows",
                        lambda *a, **kw: _rows(rat="Rat1", session="Linear1"))
    args = argparse.Namespace(tradreplay_jumps=jump_set == "traditional",
                              all_pbe_jumps=jump_set == "all_pbe",
                              traditional_status=["accepted"])
    return pj.rebuild_jump_events("Rat1", "Linear1", "sdir", summary, _inputs(), args, csv)


def test_pfeiffer_jumps_read_the_fit_maps(pfeiffer, monkeypatch):
    ev = _jump_events(pfeiffer, monkeypatch, _pfeiffer_summary(), jump_set="mb_traj")
    np.testing.assert_array_equal(pfeiffer.calls["decode"]["cell_ids"], [7, 11, 13])
    np.testing.assert_array_equal(ev.lin_map_bounds, [0, N_BINS, 2 * N_BINS])
    assert [d.shape for d in ev.primary_datas] == [(3, N_UNITS), (2, N_UNITS)] * 4
    assert [d.shape for d in ev.whole_event_datas] == [(5, N_UNITS)] * 4

    ev = _jump_events(pfeiffer, monkeypatch, _pfeiffer_summary(event_source="all_pbe"),
                      jump_set="all_pbe")
    assert ev.primary_datas is ev.whole_event_datas and len(ev.primary_datas) == 4


def test_pfeiffer_traditional_fit_with_whole_event_jumps(pfeiffer, monkeypatch):
    """Every event is decoded, and the fit's recorded counts are held to its table's events."""
    summary = _pfeiffer_summary(event_source="traditional", n_decoded_pbes=2, n_mb_trajectory_events=4)
    ev = _jump_events(pfeiffer, monkeypatch, summary, jump_set="mb_traj")
    assert pfeiffer.calls["decode"]["accepted"] is None
    assert len(ev.whole_event_datas) == 4 and len(ev.primary_datas) == 8

    with pytest.raises(ValueError, match="n_decoded_pbes"):
        _jump_events(pfeiffer, monkeypatch, {**summary, "n_decoded_pbes": 4}, jump_set="all_pbe")


def test_pfeiffer_traditional_jumps_on_a_whole_event_fit(pfeiffer, monkeypatch):
    """The table restricts the decode; the fit's counts cover other events, so are not checked."""
    ev = _jump_events(pfeiffer, monkeypatch, _pfeiffer_summary(), jump_set="traditional")
    assert pfeiffer.calls["decode"]["accepted"] == {1, 3}
    assert ev.primary_datas is ev.whole_event_datas and len(ev.primary_datas) == 2

    with pytest.raises(ValueError, match="need an events CSV"):
        _jump_events(pfeiffer, monkeypatch, _pfeiffer_summary(), jump_set="traditional", csv=None)


# ── the shared guard ──────────────────────────────────────────────────────────

def test_check_event_counts_counts_every_span_before_length_filters():
    from replay_hmm.datasets.pfeiffer.session import PFEIFFER_DECODED_VIEW

    decoded = [{"event": {"session_pbe_index": i}, "spike_traj": np.zeros((2, 4)),
                "segments": {0: [(0, 1)], 1: [(1, 4)]}} for i in range(3)]
    tmes.check_event_counts({"n_decoded_pbes": 3, "n_mb_trajectory_events": 6}, decoded,
                            PFEIFFER_DECODED_VIEW, label="x")
    tmes.check_event_counts({"n_decoded_pbes": 1, "n_mb_trajectory_events": 2}, decoded,
                            PFEIFFER_DECODED_VIEW, label="x", fit_ids={2})
    with pytest.raises(ValueError, match=r"\[x\] rebuilt 3 decoded events but the fit recorded 2"):
        tmes.check_event_counts({"n_decoded_pbes": 2}, decoded, PFEIFFER_DECODED_VIEW, label="x")
