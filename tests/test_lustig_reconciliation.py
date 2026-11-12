"""Invariants that keep the Lustig MB-jump and HMM pipelines on one event set.

Run from the repo root with:  python -m pytest tests/test_lustig_reconciliation.py -v
or standalone:  python tests/test_lustig_reconciliation.py

Every test that builds a session bundle or reads a map archive is a Lucky/rec01 data test and
is skipped unless RUN_LUSTIG_DATA_TESTS=1 is set and both data/lee_lab/Lucky and
results/Lucky/rec01/lustig_maps_abs_bw8.npz exist. The data-free tests always run.

These are deliberately few and load-bearing. Each one pins a property whose silent failure
would invalidate results rather than just annoy someone.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from replay_hmm.datasets.lustig import bundle as lustig_bundle  # noqa: E402
from replay_hmm.datasets.lustig import event_sets  # noqa: E402
from replay_hmm.datasets.lustig import events as lustig_events  # noqa: E402
from replay_hmm.decode import grid  # noqa: E402
from replay_hmm.events import runs  # noqa: E402
from replay_hmm.decode import config  # noqa: E402

RAT, SESSION = "Lucky", "rec01"
BIN_WIDTH_PX = 8

# Absolute roots, so the tests do not depend on the working directory.
RESULTS_ROOT = ROOT / "results"
DATA_ROOT = ROOT / "data" / "lee_lab"

_HAS_DATA = (
    os.environ.get("RUN_LUSTIG_DATA_TESTS") == "1"
    and (RESULTS_ROOT / RAT / SESSION / f"lustig_maps_abs_bw{BIN_WIDTH_PX}.npz").exists()
    and (DATA_ROOT / RAT).exists()
)
_DATA_REASON = (
    "Lucky/rec01 data test; set RUN_LUSTIG_DATA_TESTS=1 with data/ and the bw8 map present"
)
requires_data = pytest.mark.skipif(not _HAS_DATA, reason=_DATA_REASON)


def _bundle(**cfg_overrides):
    if not _HAS_DATA:
        pytest.skip(_DATA_REASON)
    cfg = config.DecodeConfig(bin_width_px=BIN_WIDTH_PX, **cfg_overrides)
    return lustig_bundle.build_session_bundle(
        rat=RAT, session=SESSION, cfg=cfg, maps_root=RESULTS_ROOT, data_root=DATA_ROOT,
    )


def _pbe_runs(bundle):
    return [
        (t, s, e)
        for t in bundle.trial_ids
        for s, e in runs.runs_from_mask(bundle.trial_pbe_nonrun_masks[t])
    ]


def _traj_events(bundle):
    n_maps = len(bundle.lin_map_bounds) - 1
    wheel = n_maps - 1 if bundle.include_wheel else None
    out = set()
    for t in bundle.trial_ids:
        for _ev, by_map in bundle.traj_segments_by_pbe[t].items():
            for m, segs in by_map.items():
                if m == wheel:
                    continue
                out.update((t, int(s), int(e)) for s, e in segs if e - s >= 2)
    return out


@pytest.fixture(scope="module")
def gated():
    return _bundle(pbe_merge_gated=True)


@pytest.fixture(scope="module")
def ungated():
    """The exact configuration the saved HMM fits were trained on.

    BOTH flags are needed. `pbe_merge_gated=False` restores the unconditional merge, but the
    decode-timebase duration filter would then remove the very runs that merge splits -- so
    reproducing the saved fits' event set also requires switching that filter off.
    """
    return _bundle(pbe_merge_gated=False, enforce_min_duration_on_decode_runs=False)


@requires_data
def test_speed_threshold_actually_reaches_the_bundle(ungated):
    """--speed-thresh-m-s must change the bundle, not only be recorded in the summaries.

    A recorded-but-inert parameter fails silently, so the guard is that it HAS an effect.
    """
    strict = _bundle(pbe_merge_gated=False, speed_thresh_m_s=0.01)
    assert len(_pbe_runs(strict)) != len(_pbe_runs(ungated)), (
        "speed_thresh_m_s had no effect on the PBE event set -- it is being ignored"
    )


@requires_data
@pytest.mark.parametrize("tbin,step", [(0.010, 0.005), (0.020, 0.005)])
def test_no_events_below_the_minimum_duration_on_any_grid(tbin, step):
    """No PBE run is shorter than the minimum duration, on any decode grid.

    Two independent causes, one invariant: the PBE merge, and window-width edge truncation.
    Parameterised over grids deliberately: a wider decode window drops more trailing windows
    and clips harder at the lap boundary, so PBEs near a lap end lose decode bins -- visible
    only at tbin/tbin_step > 2.
    """
    bundle = _bundle(pbe_merge_gated=True, decode_tbin_s=tbin, decode_tbin_step_s=step)
    min_bins = int(np.ceil(bundle.cfg.pbe_min_duration_s / bundle.cfg.decode_tbin_step_s))
    too_short = [(t, s, e) for t, s, e in _pbe_runs(bundle) if (e - s) < min_bins]
    assert not too_short, (
        f"tbin={tbin}/{step}: {len(too_short)} PBE runs below the {min_bins}-bin "
        f"minimum: {too_short[:5]}"
    )


@requires_data
def test_gated_merge_makes_the_decode_timebase_gate_a_noop(gated):
    """The invariant the gate buys: re-applying eligibility cannot split anything."""
    for t in gated.trial_ids:
        before = runs.runs_from_mask(gated.trial_pbe_masks[t])
        after = runs.runs_from_mask(gated.trial_pbe_nonrun_masks[t])
        assert len(after) <= len(before), (
            f"trial {t}: the non-run AND split events ({len(before)} -> {len(after)}); "
            "the merge is bridging ineligible time"
        )


@requires_data
def test_ungated_merge_still_splits_events(ungated):
    """The unconditional merge (the saved fits' configuration) demonstrably splits events.

    Pinned as a property rather than a count, so a legitimate change elsewhere in the
    detector does not break it.
    """
    min_bins = int(np.ceil(ungated.cfg.pbe_min_duration_s / ungated.cfg.decode_tbin_step_s))
    too_short = [(t, s, e) for t, s, e in _pbe_runs(ungated) if (e - s) < min_bins]
    assert too_short, (
        "the ungated merge produced no sub-threshold runs; either the split mechanism is "
        "gone or this session no longer exercises it"
    )


@requires_data
def test_gate_only_removes_events(gated, ungated):
    """Gating the merge must be subtractive: it may drop events, never invent them."""
    a, b = _traj_events(ungated), _traj_events(gated)
    assert not b - a, f"gating should only remove events, but added {b - a}"
    assert len(b) <= len(a), f"gated ({len(b)}) exceeds ungated ({len(a)})"


@requires_data
def test_spike_density_smoothing_is_not_integer_truncated():
    """The Gaussian smoothing must run in floating point.

    np.histogram returns int64 and gaussian_filter1d preserves the input dtype, so smoothing
    an integer array truncates: one spike over sigma=25 bins peaks at ~0.016 and becomes 0.
    Left uncast, the "smoothed population firing rate" collapses to 2-3 distinct values. This
    asserts the signal the PBE detector thresholds on is genuinely continuous.
    """
    from scipy.ndimage import gaussian_filter1d

    counts = np.zeros(4000, dtype=np.int64)
    counts[2000] = 1
    sigma = config.DecodeConfig().pbe_smooth_sigma_s * grid.LFP_SAMP_RATE
    assert gaussian_filter1d(counts, sigma=sigma).sum() == 0, (
        "scipy no longer truncates integer input; this test's premise needs revisiting"
    )

    bundle = _bundle()
    n_levels = len(
        np.unique(np.concatenate([
            np.asarray(bundle.trial_rate_trajs[t]).ravel() for t in bundle.trial_ids[:3]
        ]))
    )
    assert n_levels > 3, "spike counts look quantised; check the smoothing dtype"


def test_pbe_filter_modes_are_nested():
    """arm_end_reward_speed subset of nonrun_nonwheel subset of speed."""
    rng = np.random.default_rng(0)
    speed = rng.uniform(0, 200, size=20000)
    section = rng.integers(0, 12, size=20000)
    masks = {
        mode: lustig_events.pbe_eligible_mask(speed, section, config.DecodeConfig(pbe_filter_mode=mode))
        for mode in ("arm_end_reward_speed", "nonrun_nonwheel", "speed")
    }
    assert np.all(masks["arm_end_reward_speed"] <= masks["nonrun_nonwheel"])
    assert np.all(masks["nonrun_nonwheel"] <= masks["speed"])


def test_degenerate_pbe_baseline_raises_instead_of_yielding_zero_events():
    """An empty non-run baseline raises: its NaN z-scores would make every comparison False
    and silently yield zero PBEs for the whole session."""
    with pytest.raises(ValueError, match="baseline"):
        lustig_events._pbe_baseline_stats(
            np.zeros(10), np.zeros(10, dtype=bool), where="test", strict=True
        )


def test_trajectory_mass_threshold_rejects_zero():
    """At 0 an all-zero (whitened-out) posterior row would pass segment assignment."""
    with pytest.raises(ValueError, match="trajectory_mass_threshold"):
        config.DecodeConfig(trajectory_mass_threshold=0.0)


@requires_data
def test_mb_jump_and_hmm_consume_the_same_event_set(gated):
    """The backbone of the MB / HMM comparison.

    MB_jump and the HMM pipeline must be looking at byte-identical events: the same wheel
    position in the state vector, unit selection, PBE threshold and baseline scope, speed
    gate, PBE->decode-bin rule and both trajectory thresholds. Otherwise any agreement
    between the two analyses is coincidence.
    """
    from replay_hmm.datasets.lustig import mb_jump as mb

    setting = mb.RunSetting(
        decode_tbin_s=gated.cfg.decode_tbin_s,
        decode_tbin_step_s=gated.cfg.decode_tbin_step_s,
        hover_bins=0,
    )

    hmm_events = len(
        event_sets.collect_mb_traj_event_datas(
            gated, min_len=2, include_wheel=False, dedupe_across_maps=True
        )
    )

    mb_events = 0
    for trial in gated.trial_ids:
        mb_events += int(mb._compute_trial_metrics(trial, gated, setting)["event_segment_idx"].size)

    assert mb_events == hmm_events, (
        f"MB_jump sees {mb_events} events, the HMM path sees {hmm_events}; "
        "the two pipelines have drifted apart"
    )


@requires_data
def test_inbound_slots_are_even_indexed(gated):
    """The sign convention. Getting this wrong inverts every reported asymmetry.

    Asserted against occupancy, which cannot be misread: inbound running dominates outbound
    in this dataset, so the inbound slots must carry more occupancy.
    """
    from replay_hmm.datasets.lustig.sessions import load_session_mat

    maps = np.load(
        RESULTS_ROOT / RAT / SESSION / f"lustig_maps_abs_bw{BIN_WIDTH_PX}.npz", allow_pickle=True
    )["Maps"][0]["linear"]
    occ = np.asarray(maps["occ"])
    even = sum(float(np.nansum(np.asarray(occ[i], dtype=float))) for i in (0, 2, 4))
    odd = sum(float(np.nansum(np.asarray(occ[i], dtype=float))) for i in (1, 3, 5))

    mat = load_session_mat(RAT, SESSION, data_root=DATA_ROOT)
    raw = mat["Track"]
    inb = np.asarray(raw["inbound"]).ravel() == 1
    outb = np.asarray(raw["outbound"]).ravel() == 1
    assert (inb.sum() > outb.sum()) == (even > odd), (
        f"even slots hold {even:.1f}s vs odd {odd:.1f}s, but inbound={inb.sum()} "
        f"outbound={outb.sum()} samples -- even index is NOT inbound, so the jump "
        "sign convention is inverted"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--no-header", "-x"]))
