"""Parity and geometry tests for the traditional-criteria jump distribution.

The module's (`datasets.lustig.traditional_jumps`) whole claim is that it changes the
CANDIDATE WINDOWS and nothing else. The load-bearing test here is
`test_jumps_from_traj_segments_matches_mb_compute_trial_metrics`: it feeds the module's
extractor MB's own event set and asserts the jumps come out identical. Everything else in
the module is plumbing around that.
"""

from pathlib import Path

import numpy as np
import pytest


import matplotlib

matplotlib.use("Agg")

from replay_hmm.datasets.lustig import mb_jump as mbj
from replay_hmm.datasets.lustig import bundle
from replay_hmm.decode import poisson
from replay_hmm.events import runs
from replay_hmm.decode import config
from replay_hmm.datasets.lustig import traditional_jumps as tj


N_UNITS = 36
SEG_SIZES = [9, 9, 11, 11, 8, 8, 6]  # six linear segments + wheel last, as SEGMENT_ORDER
N_TRIALS = 3
T_TRIAL = 80
MIN_PROB_FACTOR = config.DecodeConfig().min_traj_prob_factor


# Which map each lap's position walks within. Confining the walk to one map is what makes
# the posterior clear `decode.poisson.confident_windows`' in-map threshold of
# min_traj_prob_factor / n_states_in_map; a walk across the whole state space leaves every
# individual map's slice below it, and every jump would be gated away.
TRIAL_MAP = {1: 0, 2: 1, 3: 2}


def _synthetic_bundle(seed: int = 0) -> bundle.SessionBundle:
    """A SessionBundle whose posterior is actually decodable.

    Gaussian place fields plus spike counts generated from a position that advances one
    state per bin inside a single map, so the decoded argmax moves and the confidence gate
    admits the windows. Random maps will NOT do: `_event_jump_diffs` gates each window on
    its in-map peak reaching `min_traj_prob_factor / n_states_in_map`, and a diffuse
    posterior clears nothing -- which would make the parity assertion below vacuously true.
    Verified: at these parameters every window in a walked map is confident and the argmax
    tracks the walk exactly.

    Trajectory segments are written by hand rather than called, so these tests exercise the
    jump extractor rather than the trajectory criterion.
    """
    bounds = np.concatenate([[0], np.cumsum(SEG_SIZES)]).astype(int)
    n_states = int(bounds[-1])

    cfg = config.DecodeConfig(decode_tbin_s=0.020, decode_tbin_step_s=0.005)

    states = np.arange(n_states, dtype=float)
    centers = np.linspace(0, n_states - 1, N_UNITS)
    lin_map_concat = 400.0 * np.exp(
        -((states[None, :] - centers[:, None]) ** 2) / (2.0 * 1.5**2)
    ) + 0.1

    def _counts_walking_in_map(map_idx: int, length: int) -> np.ndarray:
        """Expected counts for a position advancing one state per bin within one map.

        The walk wraps at the map's far end, which puts one large negative jump into each
        cycle -- useful, since it means the extractor is exercised on both signs.
        """
        lo, hi = int(bounds[map_idx]), int(bounds[map_idx + 1])
        pos = lo + (np.arange(length) % (hi - lo))
        return np.rint(lin_map_concat[:, pos] * cfg.decode_tbin_s).astype(np.uint16)

    trial_rate_trajs = {
        t: _counts_walking_in_map(TRIAL_MAP[t], T_TRIAL) for t in TRIAL_MAP
    }

    # trial -> event -> map -> [(s, e)]. Each lap's spans include the map it actually walks
    # (so real jumps come out) plus decoys: a wheel-map span (index 6), a sub-2-bin span,
    # and one span claimed under two maps at once -- exercising the wheel skip, the length
    # floor and the dedupe key inside the parity assertion.
    traj_segments_by_pbe = {
        1: {
            0: {0: [(5, 20)], 6: [(5, 20)]},
            1: {3: [(30, 31)], 0: [(34, 50)]},
        },
        2: {
            0: {1: [(2, 25)]},
            1: {1: [(40, 60)], 5: [(40, 60)]},
        },
        3: {0: {2: [(10, 12), (60, 78)]}},
    }

    return bundle.SessionBundle(
        rat="Synthetic",
        session="rec00",
        lin_map_concat=lin_map_concat,
        lin_map_bounds=bounds,
        decode_tbin_s=cfg.decode_tbin_s,
        include_wheel=True,
        min_traj_prob_factor=cfg.min_traj_prob_factor,
        trial_ids=sorted(TRIAL_MAP),
        trial_rate_trajs=trial_rate_trajs,
        trial_pbe_masks={t: np.zeros(T_TRIAL, dtype=bool) for t in trial_rate_trajs},
        trial_pbe_nonrun_masks={t: np.zeros(T_TRIAL, dtype=bool) for t in trial_rate_trajs},
        traj_segments_by_pbe=traj_segments_by_pbe,
        C_bins=SEG_SIZES[2],
        L_bins=SEG_SIZES[0],
        R_bins=SEG_SIZES[4],
        cfg=cfg,
        trial_starts_lfp=np.array([0, 100_000, 200_000], dtype=int),
        trial_ends_lfp=np.array([40_000, 140_000, 240_000], dtype=int),
    )


def _posterior_for(sess: bundle.SessionBundle, trial: int) -> np.ndarray:
    cache = poisson.build_decode_cache(
        sess.lin_map_concat,
        sess.lin_map_bounds,
        sess.decode_tbin_s,
        sess.include_wheel,
        cfg=sess.cfg,
    )
    return poisson.decode_posterior(np.asarray(sess.trial_rate_trajs[trial]), cache)


# --------------------------------------------------------------------------------------
# The decisive parity test
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("hover_bins", [0, 1])
def test_jumps_from_traj_segments_matches_mb_compute_trial_metrics(hover_bins):
    """Fed MB's own event set, the traditional extractor must reproduce MB's jumps exactly.

    This is what licenses the claim that only the candidate-window set differs. If MB's
    wheel exclusion, length floor, dedupe key or sign convention ever changes and this
    module's copy does not, this fails.
    """
    sess = _synthetic_bundle()
    setting = mbj.RunSetting(
        decode_tbin_s=sess.cfg.decode_tbin_s,
        decode_tbin_step_s=sess.cfg.decode_tbin_step_s,
        hover_bins=hover_bins,
    )

    for trial in sess.trial_ids:
        mb = mbj._compute_trial_metrics(trial, sess, setting)
        ours = tj._jumps_from_traj_segments(
            _posterior_for(sess, trial),
            sess,
            sess.traj_segments_by_pbe[trial],
            hover_bins,
            setting.min_traj_prob_factor,
        )

        np.testing.assert_array_equal(mb["jumps"], ours["jumps"])
        # Step sizes ride the same loop, so they must agree too -- and, being hover-free,
        # must not move with `hover_bins`.
        np.testing.assert_array_equal(mb["steps"], ours["steps"])
        for seg in tj.LINEAR_SEGMENTS:
            np.testing.assert_array_equal(
                mb["jumps_by_section"][seg], ours["jumps_by_section"][seg]
            )

    # A non-trivial amount of work actually happened, so the equality is not vacuous.
    total = sum(
        tj._jumps_from_traj_segments(
            _posterior_for(sess, t),
            sess,
            sess.traj_segments_by_pbe[t],
            hover_bins,
            setting.min_traj_prob_factor,
        )["jumps"].size
        for t in sess.trial_ids
    )
    assert total > 0


def test_wheel_map_contributes_no_jumps():
    """Wheel bins are time-since-onset, not position; a difference is not a displacement."""
    sess = _synthetic_bundle()
    posterior = _posterior_for(sess, 1)
    wheel_map = len(sess.lin_map_bounds) - 2  # last map index

    only_wheel = tj._jumps_from_traj_segments(posterior, sess, {0: {wheel_map: [(5, 40)]}}, 0, MIN_PROB_FACTOR)
    assert only_wheel["jumps"].size == 0
    assert only_wheel["spans"] == []


def test_spans_shorter_than_two_bins_are_skipped():
    sess = _synthetic_bundle()
    posterior = _posterior_for(sess, 1)

    assert tj._jumps_from_traj_segments(posterior, sess, {0: {0: [(5, 6)]}}, 0, MIN_PROB_FACTOR)["spans"] == []
    assert len(tj._jumps_from_traj_segments(posterior, sess, {0: {0: [(5, 7)]}}, 0, MIN_PROB_FACTOR)["spans"]) == 1


def test_dedupe_key_is_event_and_span_excluding_map_index():
    """Pins `_compute_trial_metrics`' key exactly: `(event_id, s, e)`, with no map index.

    That means one event claiming the SAME span under two maps yields one span, not two --
    MB's own comment notes this is a no-op above a 0.5 mass threshold, where only one map
    can qualify, and exists only to stay equal if the threshold ever drops. Adding map_idx
    to the key here would be more "correct" in isolation and would silently diverge from
    the run this is compared against.
    """
    sess = _synthetic_bundle()
    posterior = _posterior_for(sess, 1)

    # Same event, same span, two maps -> deduped to one, because map_idx is not in the key.
    two_maps = tj._jumps_from_traj_segments(posterior, sess, {0: {0: [(5, 20)], 1: [(5, 20)]}}, 0, MIN_PROB_FACTOR)
    assert len(two_maps["spans"]) == 1

    # Same span repeated within one map is likewise deduped.
    repeated = tj._jumps_from_traj_segments(posterior, sess, {0: {0: [(5, 20), (5, 20)]}}, 0, MIN_PROB_FACTOR)
    assert len(repeated["spans"]) == 1

    # Different spans under one event both survive.
    distinct = tj._jumps_from_traj_segments(posterior, sess, {0: {0: [(5, 20), (25, 40)]}}, 0, MIN_PROB_FACTOR)
    assert len(distinct["spans"]) == 2

    # The same span under two different events is NOT deduped -- event_id is in the key.
    two_events = tj._jumps_from_traj_segments(posterior, sess, {0: {0: [(5, 20)]}, 1: {0: [(5, 20)]}}, 0, MIN_PROB_FACTOR)
    assert len(two_events["spans"]) == 2


# --------------------------------------------------------------------------------------
# Trajectory calling inside externally supplied windows
# --------------------------------------------------------------------------------------


def test_call_traj_segments_in_windows_matches_build_session_bundle_inline_block():
    """The window path must apply the pipeline's criterion, not a lookalike.

    Mirrors `datasets.lustig.bundle.build_session_bundle`'s inline per-map block bin for bin,
    including the conversion of event-local spans back to full trial indices.
    """
    sess = _synthetic_bundle()
    posterior = _posterior_for(sess, 1)
    bounds = np.asarray(sess.lin_map_bounds, dtype=int)
    mass_threshold, merge_gap_bins, min_duration_bins = 0.60, 6, 5
    windows = [(0, 5, 40), (1, 45, 78)]

    ours = tj.call_traj_segments_in_windows(
        posterior,
        bounds,
        windows,
        mass_threshold=mass_threshold,
        merge_gap_bins=merge_gap_bins,
        min_duration_bins=min_duration_bins,
    )

    num_maps = bounds.size - 1
    per_map = np.stack(
        [posterior[:, bounds[m]:bounds[m + 1]].sum(axis=1) for m in range(num_maps)], axis=0
    )
    expected: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for event_id, s, e in windows:
        by_map = {}
        for m in range(num_maps):
            above = per_map[m, s:e] >= mass_threshold
            segs = runs.merge_runs(runs.runs_from_mask(above), merge_gap_bins)
            segs = [(a, b) for a, b in segs if (b - a) >= min_duration_bins]
            if segs:
                by_map[m] = [(s + int(a), s + int(b)) for a, b in segs]
        expected[event_id] = by_map

    assert ours == expected


def test_spans_are_confined_to_their_candidate_window():
    """A trajectory may never leak outside the window that produced it."""
    sess = _synthetic_bundle()
    posterior = _posterior_for(sess, 1)
    windows = [(0, 10, 30), (1, 50, 70)]

    called = tj.call_traj_segments_in_windows(
        posterior,
        sess.lin_map_bounds,
        windows,
        mass_threshold=0.30,
        merge_gap_bins=6,
        min_duration_bins=2,
    )
    bounds_by_event = {ev: (s, e) for ev, s, e in windows}
    for event_id, by_map in called.items():
        lo, hi = bounds_by_event[event_id]
        for spans in by_map.values():
            for s, e in spans:
                assert lo <= s < e <= hi


# --------------------------------------------------------------------------------------
# Candidate-window overlap
# --------------------------------------------------------------------------------------


def test_assert_spans_disjoint_counts_overlaps_within_a_lap_only():
    """Traditional windows can overlap where MB's runs_from_mask spans cannot."""
    disjoint = [
        {"trial": 1, "s": 0, "e": 10},
        {"trial": 1, "s": 10, "e": 20},  # touching, not overlapping
        {"trial": 2, "s": 5, "e": 15},
    ]
    assert tj.assert_spans_disjoint(disjoint, "test") == 0

    overlapping = disjoint + [{"trial": 1, "s": 18, "e": 30}]
    assert tj.assert_spans_disjoint(overlapping, "test") == 1

    # Spans on different laps share an index space but not a posterior, so they never clash.
    cross_lap = [{"trial": 1, "s": 0, "e": 10}, {"trial": 2, "s": 0, "e": 10}]
    assert tj.assert_spans_disjoint(cross_lap, "test") == 0


# --------------------------------------------------------------------------------------
# Settings / tagging
# --------------------------------------------------------------------------------------


def test_pyramidal_only_reaches_the_decode_config():
    """RunSetting is the only route by which this flag reaches the DecodeConfig."""
    for flag in (False, True):
        settings = tj.JumpSettings(pyramidal_only=flag)
        cfg = mbj._cfg_from_setting(settings.to_run_setting(), settings.bin_width_px)
        assert cfg.pyramidal_only is flag


def test_settings_round_trip_into_decode_config():
    settings = tj.JumpSettings(
        bin_width_px=4,
        tbin_s=0.030,
        tbin_step_s=0.010,
        hover_thresh_bins=2,
        trajectory_mass_thresh=0.75,
        trajectory_min_duration_s=0.050,
        trajectory_merge_gap_s=0.020,
        min_active_cells=3,
    )
    cfg = mbj._cfg_from_setting(settings.to_run_setting(), settings.bin_width_px)
    assert cfg.decode_tbin_s == 0.030
    assert cfg.decode_tbin_step_s == 0.010
    assert cfg.hover_bins == 2
    assert cfg.trajectory_mass_threshold == 0.75
    assert cfg.trajectory_min_duration_s == 0.050
    assert cfg.trajectory_merge_gap_s == 0.020
    assert cfg.min_active_cells == 3
    assert cfg.bin_width_px == 4


def test_run_tag_carries_bin_width_and_the_mb_tag():
    """RunSetting.tag omits bin_width_px, so the prefix is what keeps runs from colliding."""
    settings = tj.JumpSettings()
    tag = tj.run_tag(settings)
    assert tag == f"bw8__{settings.to_run_setting().tag}"
    assert tj.run_tag(tj.JumpSettings(bin_width_px=4)) != tag


def test_validate_rejects_out_of_range_settings():
    with pytest.raises(ValueError):
        tj.JumpSettings(bin_width_px=0).validate()
    with pytest.raises(ValueError):
        tj.JumpSettings(trajectory_mass_thresh=0.0).validate()
    with pytest.raises(ValueError):
        tj.JumpSettings(trajectory_mass_thresh=1.5).validate()
    with pytest.raises(ValueError):
        tj.JumpSettings(hover_thresh_bins=-1).validate()


# --------------------------------------------------------------------------------------
# Pooling convention
# --------------------------------------------------------------------------------------


def test_pool_jumps_for_figures_rescales_each_session_onto_the_reference_bin():
    """MB's recipe: per-session cm, pool, then express in reference-bin units."""
    processed = [
        {"jumps": np.array([1.0, 2.0]), "bin_width_cm": 4.0},
        {"jumps": np.array([1.0]), "bin_width_cm": 5.0},
        {"jumps": np.array([-1.0]), "bin_width_cm": 6.0},
    ]
    pooled, ref_bin_mm = tj.pool_jumps_for_figures(processed)

    assert ref_bin_mm == pytest.approx(50.0)  # median of 40, 50, 60 mm
    # 1 bin at 4 cm is 0.8 reference bins; at 6 cm it is 1.2.
    np.testing.assert_allclose(pooled, [0.8, 1.6, 1.0, -1.2])

    # Pooling raw bins without per-session scaling gives a different answer.
    naive = np.concatenate([r["jumps"] for r in processed])
    assert not np.allclose(pooled, naive)


def test_jump_stats_is_fed_integer_bins_and_reports_signs():
    stats = tj.jump_stats(np.array([2.0, 2.0, 2.0, -1.0, 0.0]))
    assert stats["n_jumps"] == 5
    assert stats["n_pos"] == 3
    assert stats["n_neg"] == 1
    assert stats["n_nonzero"] == 4


# --------------------------------------------------------------------------------------
# Pairing with the MB run
# --------------------------------------------------------------------------------------


def test_tj_mb_pairing_probe_looks_under_the_results_root():
    from replay_hmm.datasets.lustig import traditional_jumps as tj
    src = Path(tj.__file__).read_text()
    probe = src[src.index("mb_name = f\"MB_jump_results_"):src.index("MB pairing:")]
    assert "paths.RESULTS_ROOT / rat / session / \"mb_jump\" / mb_name" in probe
    assert "args.maps_root" not in probe
