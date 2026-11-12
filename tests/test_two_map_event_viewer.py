"""The two-map (Pfeiffer, Buzsaki) sources for the per-event browser.

What these pin, with no real data:
  * `count_windows` is each dataset's own counting rule: it reproduces Pfeiffer's
    `build_spike_traj_for_event` and Buzsaki's `decode_pbe` counts on the same windows;
  * padding keeps the fit's counts as the window interior, counts the pad windows on the fit's
    own grid extended by whole steps, and refuses an event its counter cannot reproduce;
  * record coordinates, trajectory spans and scope arrays line up with the fit's counts;
  * position lands on the same bin in both maps, NaN reads as untracked, and a fill is named;
  * scope: traditional fits map to `--jump-scope traditional`, whose records are the decoded
    records the table names (in fit order, every named event required) and whose spans are
    whole windows; the backdrop directory is chosen by its metadata;
  * the per-dataset plumbing: roots and inputs file pinned to the PPC's (and so the fit CLIs').
"""

import json
from pathlib import Path

import numpy as np
import pytest

from replay_hmm.decode import poisson
from replay_hmm.viewer import engine as ev
from replay_hmm.viewer import two_map as tme
from replay_hmm.io import names

N = 10                                    # bins per map
BOUNDS = np.array([0, N, 2 * N])          # [outbound | inbound]
TBIN = STEP = 0.01
BIN_CM = 4.0


def _spikes(rng, cell_ids, t0=100.0, t1=101.0, rate=60.0):
    return {int(c): np.sort(rng.uniform(t0, t1, rng.poisson(rate * (t1 - t0)))) for c in cell_ids}


def _session(rng, *, n_windows=12, segments=((0, 2, 9),), pos_cm=None, pos_observed=None):
    cell_ids = np.array([3, 5, 8, 13])
    spikes = _spikes(rng, cell_ids)
    starts = 100.3 + STEP * np.arange(n_windows)
    event = tme.TwoMapEvent(
        index=7, starts_s=starts, counts=tme.count_windows(spikes, cell_ids, starts, TBIN),
        segments=list(segments),
    )
    pos_ts = np.linspace(99.0, 102.0, 301)
    return tme.TwoMapSession(
        events=[event], spike_times=spikes, cell_ids=cell_ids,
        density_grids=[(np.arange(99.0, 102.0, 0.001), np.ones(3000), np.ones(3000, bool))],
        baseline_mean=0.0, baseline_sd=1.0, z_thresholds=(3.0, 0.0),
        pos_ts=pos_ts,
        pos_cm=np.full(pos_ts.size, 13.0) if pos_cm is None else pos_cm,
        bin_edges_cm=BIN_CM * np.arange(N + 1),
        pos_observed=pos_observed, pos_fill=None if pos_observed is None else "hold_last",
    )


def _source(session, pad_ms=20.0):
    inputs = {"lin_map_bounds": BOUNDS, "lin_map_concat": np.ones((4, 2 * N))}
    summary = {"decode_tbin_s": TBIN, "decode_tbin_step_s": STEP, "hmm_bin_width_cm": BIN_CM}
    return tme.TwoMapSource(
        name="R/S", name_spaced="R / S", session=session, inputs=inputs, summary=summary,
        pad_ms=pad_ms, signed_jumps=lambda z, b: np.diff(z),
    )


# --------------------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------------------

def test_count_windows_is_pfeiffers_rule():
    from replay_hmm.datasets.pfeiffer import events as pfeiffer_events

    rng = np.random.default_rng(0)
    cell_ids = np.array([1, 2, 4])
    spikes = _spikes(rng, cell_ids)
    traj, starts, _centers = pfeiffer_events.build_spike_traj_for_event(spikes, cell_ids, 100.2, 100.7, TBIN, STEP)
    assert np.array_equal(tme.count_windows(spikes, cell_ids, starts, TBIN), traj.T)


def test_count_windows_is_buzsakis_rule():
    from replay_hmm.decode import two_map
    from replay_hmm.decode.grid import DecodeGrid

    rng = np.random.default_rng(1)
    cell_ids = [1, 2, 4]
    spikes = _spikes(rng, cell_ids)
    grid = DecodeGrid(tbin_s=TBIN, tbin_step_s=STEP)
    cache = poisson.build_decode_cache(rng.uniform(0.5, 5.0, (3, 2 * N)), BOUNDS.tolist(), TBIN,
                                   include_wheel=False)
    pbe = {"start_s": 100.2, "end_s": 100.7, "duration_s": 0.5, "peak_pfr": 1.0,
           "pfr_trace": np.zeros(1), "pfr_times": np.zeros(1)}
    dec = two_map.decode_pbe(pbe, spikes, cell_ids, cache, N, decode_grid=grid)
    # The Buzsaki viewer's (viewer.buzsaki) reconstruction of decode_pbe's window starts.
    starts = np.arange(pbe["start_s"], pbe["end_s"] - grid.tbin_s + 1e-9, grid.tbin_step_s)
    assert np.array_equal(starts + 0.5 * TBIN, dec["win_centers_s"])
    assert np.array_equal(tme.count_windows(spikes, cell_ids, starts, TBIN),
                          dec["rate_traj"].T.astype(np.int32))


# --------------------------------------------------------------------------------------
# padding and coordinates
# --------------------------------------------------------------------------------------

def test_padding_keeps_the_fit_counts_and_extends_the_grid_by_whole_steps():
    session = _session(np.random.default_rng(2))
    src = _source(session, pad_ms=20.0)
    (rec,) = src.records()
    event = session.events[0]
    k = src.pad_bins
    assert k == 2 and (rec.s_bin, rec.e_bin) == (k, k + 12)
    assert src.n_bins_total(rec) == 12 + 2 * k

    padded = src.counts(rec, 0, src.n_bins_total(rec))
    assert padded.dtype == np.int32
    assert np.array_equal(padded[k:k + 12], event.counts)
    centers = src.bin_centers_s(rec, 0, src.n_bins_total(rec))
    assert np.array_equal(centers[k:k + 12], event.starts_s + 0.5 * TBIN)
    assert np.allclose(np.diff(centers), STEP)
    left = event.starts_s[0] - STEP * np.arange(k, 0, -1)
    right = event.starts_s[-1] + STEP * np.arange(1, k + 1)
    count = lambda s: tme.count_windows(session.spike_times, session.cell_ids, s, TBIN)  # noqa: E731
    assert np.array_equal(padded[:k], count(left))
    assert np.array_equal(padded[k + 12:], count(right))
    assert rec.t0_s == event.starts_s[0]
    assert np.isclose(rec.duration_ms, 12 * STEP * 1000.0)


def test_an_event_its_counter_cannot_reproduce_is_refused():
    session = _session(np.random.default_rng(3))
    session.events[0].counts = session.events[0].counts + 1
    src = _source(session)
    with pytest.raises(AssertionError, match="count_windows"):
        src.counts(src.records()[0], 0, 1)


def test_scope_arrays_are_the_fit_counts_at_both_scopes():
    session = _session(np.random.default_rng(4), segments=((1, 0, 4), (0, 6, 11)))
    src = _source(session)
    (rec,) = src.records()
    k, counts = src.pad_bins, session.events[0].counts
    assert src.traj_segments(rec) == [(1, k, k + 4), (0, k + 6, k + 11)]
    traj = src.scope_arrays(rec, "traj")
    assert [a.shape[0] for a in traj] == [4, 5]
    assert np.array_equal(traj[1], counts[6:11])
    (whole,) = src.scope_arrays(rec, "full-pbe")
    assert np.array_equal(whole, counts)
    # The engine's bold Viterbi slices the padded window by the same spans.
    for (s, e), a in zip(src.scope_spans(rec, "traj"), traj):
        assert np.array_equal(src.counts(rec, s, e), a)


# --------------------------------------------------------------------------------------
# position
# --------------------------------------------------------------------------------------

def test_position_is_drawn_on_the_same_bin_in_both_maps():
    src = _source(_session(np.random.default_rng(5)))            # 13 cm, 4 cm bins -> bin 3
    (rec,) = src.records()
    lines, desc = src.position_lines(rec, -20.0, 140.0)
    assert len(lines) == 2 and desc == "13 cm"
    assert set(np.unique(lines[0][1])) == {3.0} and set(np.unique(lines[1][1])) == {N + 3.0}


def test_untracked_position_and_named_fill():
    rng = np.random.default_rng(6)
    nan_pos = np.full(301, np.nan)
    src = _source(_session(rng, pos_cm=nan_pos))
    lines, desc = src.position_lines(src.records()[0], -20.0, 140.0)
    assert desc == "untracked" and not any(np.isfinite(y).any() for _t, y in lines)

    src = _source(_session(rng, pos_observed=np.zeros(301, bool)))
    assert src.position_lines(src.records()[0], -20.0, 140.0)[1] == "13 cm (hold_last fill)"


# --------------------------------------------------------------------------------------
# scope and backdrop
# --------------------------------------------------------------------------------------

def test_fit_scope_maps_event_sources():
    assert tme.fit_scope({"event_source": "mb_traj"}, "x") == "traj"
    assert tme.fit_scope({"event_source": "all_pbe"}, "x") == "full-pbe"
    assert tme.fit_scope({"use_all_pbes": True}, "x") == "full-pbe"
    assert tme.fit_scope({"event_source": "traditional"}, "x") == "traditional"
    with pytest.raises(SystemExit, match="unrecognised"):
        tme.fit_scope({"event_source": "running"}, "x")


def _records(indices):
    return [ev.EventRecord(trial=0, event_idx=i, s_bin=2, e_bin=2 + 3 + i % 2, t0_s=float(i),
                           t1_s=float(i) + 0.05, peak_z=1.0) for i in indices]


def test_traditional_records_are_the_named_events_in_fit_order():
    import pandas as pd

    records = _records([0, 2, 3, 5, 8])
    rows = pd.DataFrame({"session_pbe_index": [8, 2, 5]})
    got = tme.traditional_records(records, rows, "x")
    assert [r.event_idx for r in got] == [2, 5, 8]
    assert all(r is records[i] for r, i in zip(got, (1, 3, 4)))
    with pytest.raises(ValueError, match="no decoded record"):
        tme.traditional_records(records, pd.DataFrame({"session_pbe_index": [2, 4]}), "x")


def test_traditional_fit_guards_run_on_the_named_events():
    import pandas as pd

    rng = np.random.default_rng(0)
    session = _session(rng)
    src = _source(session)
    rec = src.records()
    trad = tme.traditional_records(rec, pd.DataFrame({"session_pbe_index": [7]}), "x")
    seqs = tme.fitted_sequences(src, trad, "traditional")
    # Whole window, not the trajectory segment.
    assert len(seqs) == 1 and np.array_equal(seqs[0], session.events[0].counts)
    tme.check_event_set({"n_decoded_pbes": 1, "n_cv_events": 1}, len(trad), seqs, "x")
    assert src.scope_spans(rec[0], "traditional") == src.scope_spans(rec[0], "full-pbe")


def test_traditional_table_resolution(tmp_path):
    import argparse

    local = tmp_path / "tradtag"
    local.mkdir()
    (local / ev.TRADITIONAL_EVENTS_NAME).write_text("x")
    summary = {"event_source": "traditional", "traditional_run_tag": "tradtag",
               "traditional_events_csv": "/cluster/tradtag/traditional_decode_events.csv",
               "traditional_statuses": ["accepted"]}
    args = argparse.Namespace(traditional_events_csv=None, traditional_status=None)
    csv, statuses = ev.resolve_traditional_table(
        summary, args, fit_scope="traditional", local_dir=tmp_path, rat="R", session="S")
    assert csv == local / ev.TRADITIONAL_EVENTS_NAME and statuses == ("accepted",)
    # A different verdict set than the fit's is a different event set.
    args.traditional_status = ["no_candidate"]
    with pytest.raises(SystemExit, match="trained on statuses"):
        ev.resolve_traditional_table(summary, args, fit_scope="traditional",
                                     local_dir=tmp_path, rat="R", session="S")
    # A table from another traditional run tag is refused.
    other = tmp_path / "other"
    other.mkdir()
    (other / ev.TRADITIONAL_EVENTS_NAME).write_text("x")
    args = argparse.Namespace(traditional_events_csv=other / ev.TRADITIONAL_EVENTS_NAME,
                              traditional_status=None)
    with pytest.raises(SystemExit, match="traditional run tag"):
        ev.resolve_traditional_table(summary, args, fit_scope="traditional",
                                     local_dir=tmp_path, rat="R", session="S")
    # Any other fit records no table: the flag is required, verdicts default to accepted.
    args = argparse.Namespace(traditional_events_csv=None, traditional_status=None)
    with pytest.raises(SystemExit, match="needs --traditional-events-csv"):
        ev.resolve_traditional_table({}, args, fit_scope="traj", local_dir=tmp_path,
                                     rat="R", session="S")
    args.traditional_events_csv = other / ev.TRADITIONAL_EVENTS_NAME
    assert ev.resolve_traditional_table({}, args, fit_scope="traj", local_dir=tmp_path,
                                        rat="R", session="S")[1] == ("accepted",)


def _write_jumps(d: Path, source: str, key: str, jumps) -> None:
    d.mkdir(parents=True)
    np.savez(d / "jump_distributions.npz", **{key: np.asarray(jumps)})
    (d / "jump_distribution_metadata.json").write_text(json.dumps({"jump_event_source": source}))


def test_backdrop_directory_is_chosen_by_its_recorded_scope(tmp_path):
    base = tmp_path / "jump_distribution"
    _write_jumps(base / "tag", "mb_called_trajectories", "k__S", [1, 2])
    _write_jumps(base / "tag__allpbe_jumps", "all_pbe", "k__S", [3, 4, 5])
    got = tme.load_backdrop(results_root=tmp_path, run_tag="tag", scope="full-pbe", key="k__S")
    assert got.tolist() == [3, 4, 5]
    assert tme.load_backdrop(results_root=tmp_path, run_tag="tag", scope="traj",
                             key="k__S").tolist() == [1, 2]


def test_backdrop_finds_the_tradreplay_jumps_directory(tmp_path):
    base = tmp_path / "jump_distribution"
    _write_jumps(base / "tag", "mb_traj", "k__S", [1, 2])
    _write_jumps(base / "tag__tradreplay_jumps", "traditional", "k__S", [6, 7])
    assert tme.load_backdrop(results_root=tmp_path, run_tag="tag", scope="traditional",
                             key="k__S").tolist() == [6, 7]


def test_backdrop_of_the_other_scope_warns(tmp_path):
    _write_jumps(tmp_path / "jump_distribution" / "tag", "mb_traj", "k__S", [1])
    with pytest.warns(RuntimeWarning, match="not directly comparable"):
        tme.load_backdrop(results_root=tmp_path, run_tag="tag", scope="full-pbe", key="k__S")
    with pytest.raises(SystemExit, match="no entry"):
        tme.load_backdrop(results_root=tmp_path, run_tag="tag", scope="traj", key="missing")


# --------------------------------------------------------------------------------------
# per-dataset plumbing
# --------------------------------------------------------------------------------------

def test_the_pipeline_imports_keep_the_viewers_backend(tmp_path):
    # The Pfeiffer viewer imports its pipeline in load_session, after the engine imported
    # pyplot, so no pipeline module may switch the backend at import (the backend is chosen
    # by `replay-hmm`, util/mpl). A fresh interpreter
    # on svg (it stands in for an interactive backend; always available) imports the engine and
    # then everything load_session does, and must still be on svg.
    import os
    import subprocess
    import sys

    code = (
        "import matplotlib, matplotlib.pyplot as plt\n"
        "from replay_hmm.viewer import engine, two_map, pfeiffer\n"
        "from replay_hmm.datasets.pfeiffer import jumps, adapters, loader, mb_jump, events, fit\n"
        "from replay_hmm.datasets.pfeiffer import rebuild, session\n"
        "from replay_hmm.analyses.ppc import pfeiffer as ppc\n"
        "print(matplotlib.get_backend())\n"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ, PYTHONPATH=str(src), MPLBACKEND="svg")
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "svg"


def test_roots_and_inputs_match_the_ppc_modules():
    from replay_hmm.viewer import buzsaki as bev
    from replay_hmm.analyses.ppc import buzsaki as bz
    from replay_hmm.viewer import pfeiffer as pev
    from replay_hmm.analyses.ppc import pfeiffer as pf

    for viewer, ppc_module, inputs_name in (
        (pev, pf, "PFEIFFER_INPUTS_NAME"), (bev, bz, "BUZSAKI_INPUTS_NAME"),
    ):
        assert viewer.RESULTS_ROOT == ppc_module.RESULTS_ROOT
        assert viewer.FIGURES_ROOT == ppc_module.FIGURES_ROOT
        # Both read the fit's inputs file through the one io.names constant.
        assert viewer.DATASET.inputs_name == getattr(names, inputs_name)
        assert f"inputs_name=names.{inputs_name}," in Path(ppc_module.__file__).read_text()


def test_session_keys_and_backdrop_keys():
    from replay_hmm.viewer import buzsaki as bev
    from replay_hmm.viewer import pfeiffer as pev

    class Args:
        rat, session = "Rat1", "Linear1"

    assert pev.DATASET.session_key(Args) == ("Rat1", "Linear1")
    assert pev.DATASET.session_rel("Rat1", "Linear1") == Path("Rat1") / "Linear1"
    assert pev.DATASET.backdrop_key("k", "Rat1", "Linear1") == "k__Rat1__Linear1"

    Args.session = "Gatsby_08022013"
    assert bev.DATASET.session_key(Args) == ("Gatsby", "Gatsby_08022013")
    assert bev.DATASET.session_rel("Gatsby", "Gatsby_08022013") == Path("Gatsby_08022013")
    assert bev.DATASET.backdrop_key("k", "Gatsby", "Gatsby_08022013") == "k__Gatsby_08022013"
