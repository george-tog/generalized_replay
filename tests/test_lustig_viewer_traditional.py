"""`replay-hmm viewer --dataset lustig --jump-scope traditional`, with no real data.

What these pin:
  * the records are the traditional event handles -- the fit's own geometry
    (`traditional_io.collect_traditional_event_handles`) -- in their order, and the scope's
    arrays are each handle's `spike_traj` exactly;
  * the windows are not bundle PBEs, so no trajectory segments are looked up for them, and
    their labels and file tags cannot collide with a PBE's;
  * the session-input cache keeps a traditional event set in its own file, keyed on the table,
    and a PBE cache's manifest is unchanged;
  * the fit's scope is read from `event_source`, then the legacy `use_all_pbes`.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from replay_hmm.datasets.lustig import traditional_io
from replay_hmm.decode import config, grid
from replay_hmm.viewer import engine as ev
from replay_hmm.viewer import input_cache
from replay_hmm.viewer import lustig as lv

CFG = config.DecodeConfig(decode_tbin_s=0.010, decode_tbin_step_s=0.010)
TRIALS = ((0, 2000), (2500, 4000))         # lap [start, end] in LFP samples (1250 Hz)


class _Bundle:
    """The SessionBundle fields the collector and the viewer read."""

    def __init__(self, n_units=3):
        self.rat, self.session, self.cfg = "Rat", "rec01", CFG
        self.trial_starts_lfp = np.array([t[0] for t in TRIALS])
        self.trial_ends_lfp = np.array([t[1] for t in TRIALS])
        self.trial_ids = [1, 2]
        self.trial_rate_trajs = {}
        for i, (ts, te) in enumerate(TRIALS, start=1):
            n = grid.decode_window_grid(ts, te, CFG)[0].size
            # Column j holds j + 100 * lap, so any slice names its own bins.
            self.trial_rate_trajs[i] = np.tile(np.arange(n, dtype=np.uint16) + 100 * i,
                                               (n_units, 1))
        self.lin_map_bounds = np.array([0, 5, 10])
        self.lin_map_concat = np.ones((n_units, 10))
        self.C_bins = self.L_bins = self.R_bins = 5


def _events():
    return pd.DataFrame([
        {"rat": "Rat", "session": "rec01", "trial": 2, "event_start_lfp": 2700,
         "event_end_lfp": 2800, "trial_pbe_index": 4, "status": "accepted"},
        {"rat": "Rat", "session": "rec01", "trial": 1, "event_start_lfp": 100,
         "event_end_lfp": 160, "trial_pbe_index": 0, "status": "accepted"},
        {"rat": "Rat", "session": "rec01", "trial": 1, "event_start_lfp": 900,
         "event_end_lfp": 1000, "trial_pbe_index": 3, "status": "accepted"},
    ]).sort_values(["trial", "event_start_lfp"]).reset_index(drop=True)


def _records(bundle):
    arrays = SimpleNamespace(trial_starts=bundle.trial_starts_lfp, trial_ends=bundle.trial_ends_lfp)
    grids = {tr: grid.decode_window_grid(ts, te, CFG)
             for tr, (ts, te) in zip(bundle.trial_ids, TRIALS)}
    z = {tr: np.linspace(0.0, 1.0, te + 1 - ts) for tr, (ts, te) in zip(bundle.trial_ids, TRIALS)}
    handles, _ = traditional_io.collect_traditional_event_handles(bundle, _events(), min_len=2)
    return handles, input_cache.build_traditional_records(arrays, z, grids, handles), grids


def test_records_are_the_handles_in_fit_order_and_score_their_counts():
    bundle = _Bundle()
    handles, records, grids = _records(bundle)
    assert [(r.trial, r.event_idx) for r in records] == [(1, 0), (1, 3), (2, 4)]
    for h, r in zip(handles, records):
        assert (r.s_bin, r.e_bin) == (h["s"], h["e"])
        assert r.t0_s == grids[r.trial][0][h["s"]] / grid.SPK_SAMP_RATE
        assert r.t1_s == grids[r.trial][1][h["e"] - 1] / grid.SPK_SAMP_RATE
        (arr,) = lv.event_spike_arrays(bundle, r, "traditional")
        assert arr.dtype == np.int32 and np.array_equal(arr, h["spike_traj"])


def test_traditional_windows_have_no_segments_and_their_own_labels():
    bundle = _Bundle()
    _h, records, _g = _records(bundle)
    inputs = SimpleNamespace(session_obj=bundle)
    src = lv.LustigSource(inputs, traditional=True)
    rec = records[1]
    assert src.traj_segments(rec) == []
    assert src.scope_spans(rec, "traditional") == [(rec.s_bin, rec.e_bin)]
    assert src.event_label(rec) == "lap 1 trad #3"
    assert src.event_file_tag(rec) == "lap1_trad3"
    pbe_src = lv.LustigSource(inputs)
    assert pbe_src.event_file_tag(rec) == "lap1_ev3"


def test_traditional_cache_is_its_own_file_keyed_on_the_table(tmp_path):
    npz, man = input_cache.cache_files(tmp_path, 20.0)
    npz_t, man_t = input_cache.cache_files(tmp_path, 20.0, "tradtag")
    assert npz.name == "session_inputs__pad20ms.npz"
    assert npz_t.name == "session_inputs__pad20ms__trad_tradtag.npz" and man_t != man

    csv = tmp_path / "tradtag" / "traditional_decode_events.csv"
    csv.parent.mkdir()
    csv.write_text("x")
    ident = input_cache.traditional_identity(csv, ("accepted",))
    assert ident["run_tag"] == "tradtag" and ident["statuses"] == ["accepted"]

    kw = dict(cfg=CFG, run_tag="tag", pad_ms=20.0, rat="Rat", session="rec01",
              data_root=tmp_path, maps_root=tmp_path, run_dir=tmp_path)
    assert "traditional" not in input_cache.cache_manifest(**kw)
    manifest = input_cache.cache_manifest(**kw, traditional=ident)
    assert manifest["traditional"] == ident

    # A cache written for another table (or for PBEs) is not reused.
    npz_t.parent.mkdir(parents=True)
    npz_t.write_bytes(b"")
    stale = dict(manifest, traditional=dict(ident, statuses=["no_candidate"]))
    man_t.write_text(json.dumps(stale, default=str))
    assert input_cache.read_inputs_cache(
        npz_path=npz_t, manifest_path=man_t, manifest=manifest, cfg=CFG, rat="Rat",
        session="rec01", requested_pad_ms=20.0,
    ) is None


def test_fit_scope_from_the_summary():
    assert ev.fit_jump_scope(lv.saved_event_source({"event_source": "traditional"})) == "traditional"
    assert ev.fit_jump_scope(lv.saved_event_source({"use_all_pbes": False})) == "traj"
    assert ev.fit_jump_scope(lv.saved_event_source({"use_all_pbes": True})) == "full-pbe"
    # A traditional fit also records use_all_pbes False; event_source wins.
    assert ev.fit_jump_scope(lv.saved_event_source(
        {"event_source": "traditional", "use_all_pbes": False})) == "traditional"
    assert lv.saved_event_source({}) is None
    assert ev.fit_jump_scope("running") is None
