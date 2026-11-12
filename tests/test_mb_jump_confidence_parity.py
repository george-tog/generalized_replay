"""The MB-jump per-window confidence criterion is ONE criterion across all datasets.

A decoded window contributes a jump only when its peak posterior within the map it was
assigned to is at least `min_traj_prob_factor / n_states_in_map`, and a jump needs BOTH
of its endpoint windows to clear that bar.

Different rules per dataset would silently make the three jump histograms three different
estimators: an ungated in-map argmax, or a threshold divided by the states in ALL maps
rather than the assigned one (2x too lenient), each changes what counts as a jump. Per-dataset
tests cannot see that. This file is the cross-dataset invariant: the four implementations must
agree window-for-window on one shared posterior. `poisson.confident_windows` is the single
definition they all route through, and the denominator is read off the map slice itself so it
cannot be got wrong per call site.
"""

import inspect

import numpy as np
import pytest

from replay_hmm.datasets.buzsaki import mb_jump as bmj
from replay_hmm.datasets.lustig import mb_jump as lmj
from replay_hmm.jumps import mb as lmd
from replay_hmm.decode import poisson
from replay_hmm.decode import config
from replay_hmm.datasets.pfeiffer import ratemaps
from replay_hmm.datasets.pfeiffer import session as pfeiffer_session
from replay_hmm.datasets.pfeiffer import events as pfeiffer_events

N_BINS = 20
FACTOR = 5.0
BOUNDS = np.array([0, N_BINS, 2 * N_BINS], dtype=int)


def _joint_posterior(path, unconfident=()):
    """A 2-map joint posterior whose FIRST map walks `path`, one bin per window.

    Windows listed in `unconfident` are flattened within the first map instead of peaked,
    so their in-map peak falls below the threshold. Rows are normalised over BOTH maps,
    which is what the real decoder produces -- the point being that the criterion's
    denominator is the assigned map's width, not the row's.
    """
    T = len(path)
    post = np.zeros((T, 2 * N_BINS))
    for t, b in enumerate(path):
        if t in unconfident:
            post[t, :N_BINS] = 0.9 / N_BINS
        else:
            post[t, b] = 0.9
            post[t, :N_BINS] += 0.1 / N_BINS
        post[t, N_BINS:] = (1.0 - post[t, :N_BINS].sum()) / N_BINS
    return post / post.sum(axis=1, keepdims=True)


def _lmd_cache():
    return poisson.DecodeCache(
        logpfmaps=np.zeros((1, 2 * N_BINS)),
        sumpfmaps=np.zeros(2 * N_BINS),
        map_slices=[slice(0, N_BINS), slice(N_BINS, 2 * N_BINS)],
        include_wheel=False,
    )


def _all_four(post, factor=FACTOR):
    """Jumps from each dataset's own entry point, read over the SAME map slice.

    Lustig and the `jumps.mb` helpers (imported here as `lmd`) negate even map indices (their
    inbound-first layout), so
    map 0 comes back sign-flipped there; that convention is pinned elsewhere and is
    undone here so the four are comparable as magnitudes.
    """
    T = post.shape[0]
    pf = pfeiffer_session.compute_jumps(
        post,
        {ratemaps.MAP_OUTBOUND: [(0, T)]},
        BOUNDS,
        pfeiffer_events.RunSetting(min_traj_prob_factor=factor, hover_thresh_bins=0),
    )[ratemaps.MAP_OUTBOUND]
    bz = lmd.compute_jumps({"map_posterior": post[:, :N_BINS]}, min_prob_factor=factor)
    lu = -lmj._event_jump_diffs(post, BOUNDS, 0, (0, T), 0, factor)
    ref = -lmd.jumps_by_map_from_segments(post, {0: [(0, T)]}, _lmd_cache(), factor)[0]
    return {"pfeiffer": pf, "buzsaki": bz, "lustig": lu, "lmd": ref.astype(float)}


def test_all_datasets_agree_when_every_window_is_confident():
    got = _all_four(_joint_posterior([2, 4, 6, 8, 10]))
    for name, jumps in got.items():
        np.testing.assert_allclose(jumps, [2.0, 2.0, 2.0, 2.0], err_msg=name)


def test_an_unconfident_window_breaks_the_step_in_every_dataset():
    """Window 2 is flat within the map, so it yields no jump on EITHER side of itself.

    The 4 -> 8 step must not appear: bridging across an undecodable window would invent a
    displacement of double the real step size, which is exactly the kind of artefact the
    gate exists to suppress.
    """
    got = _all_four(_joint_posterior([2, 4, 6, 8, 10], unconfident={2}))
    for name, jumps in got.items():
        np.testing.assert_allclose(jumps, [2.0, 2.0], err_msg=name)
        assert 4.0 not in np.abs(jumps), f"{name} bridged across the unconfident window"


def test_the_denominator_is_the_assigned_map_not_the_whole_posterior():
    """The case this file exists for.

    A window whose in-map peak lands between 5/(2N) and 5/N is confident under an all-states
    denominator and unconfident under the assigned-map one. Every dataset must call it
    unconfident.
    """
    per_map = FACTOR / N_BINS
    all_states = FACTOR / (2 * N_BINS)
    peak = 0.5 * (per_map + all_states)
    assert all_states < peak < per_map

    post = np.zeros((3, 2 * N_BINS))
    for t, b in enumerate([4, 6, 8]):
        post[t, b] = peak
        post[t, N_BINS:] = (1.0 - peak) / N_BINS
    np.testing.assert_allclose(post.sum(axis=1), 1.0)
    np.testing.assert_allclose(post[:, :N_BINS].max(axis=1), peak)

    for name, jumps in _all_four(post).items():
        assert jumps.size == 0, f"{name} accepted a window below {FACTOR}x in-map uniform"


def test_confident_windows_reads_its_denominator_off_the_slice():
    """`poisson.confident_windows` cannot be handed the wrong denominator by a caller.

    The width comes from `seg.shape[1]`, so a call site that slices one map correctly
    gets the right threshold with no separate argument to keep in sync -- the mechanism
    that keeps the four implementations above from drifting apart.
    """
    for n in (7, 20, 63):
        seg = np.full((1, n), 1.0 / n)
        assert not poisson.confident_windows(seg, FACTOR)[0]
        seg = np.zeros((1, n))
        seg[0, 0] = FACTOR / n
        assert poisson.confident_windows(seg, FACTOR)[0], f"boundary is >=, n={n}"


@pytest.mark.parametrize("shape", [(0, N_BINS), (4, 0)])
def test_confident_windows_handles_degenerate_slices(shape):
    mask = poisson.confident_windows(np.zeros(shape), FACTOR)
    assert mask.shape == (shape[0],)
    assert mask.dtype == bool
    assert not mask.any()


def test_buzsaki_takes_the_factor_as_an_argument():
    """The factor reaches `compute_jumps` as an argument, not a module global a caller could
    override inside its joblib workers; leaving it out is the default, 5x in-map uniform, and
    nothing in the module namespace can change that.

    The ungated numbers show what the gate buys: with the threshold effectively off, the flat
    window is still argmaxed, and `np.argmax` on a tied row returns bin 0, injecting a
    spurious -4 and +8 into a trajectory whose every real step is +2 -- a bias towards large
    jumps that no downstream averaging removes.
    """
    post = _joint_posterior([2, 4, 6, 8, 10], unconfident={2})
    traj = {"map_posterior": post[:, :N_BINS]}
    np.testing.assert_allclose(lmd.compute_jumps(traj, 1e-9), [2.0, -4.0, 8.0, 2.0])  # gate off
    np.testing.assert_allclose(lmd.compute_jumps(traj, FACTOR), [2.0, 2.0])
    assert lmd.DEFAULT_MIN_TRAJ_PROB_FACTOR == FACTOR
    np.testing.assert_allclose(lmd.compute_jumps(traj), [2.0, 2.0])
    np.testing.assert_allclose(lmd.compute_jumps(traj, None), [2.0, 2.0])
    assert not hasattr(bmj, "MIN_TRAJ_PROB_FACTOR")

def test_a_non_default_factor_does_not_collide_with_the_default_run_on_disk():
    """A `--min-traj-prob-factor` sweep must not overwrite the default run's outputs.

    The factor changes every jump, so two settings sharing an output path would leave npz
    and figures that are indistinguishable by filename -- the failure the conditional
    suffix idiom exists to prevent. The suffix must be EMPTY at the default, so nothing
    already on disk is renamed. Pfeiffer and Lustig carry it in the run tag; Buzsaki, which
    has no run tag, in an output subdirectory.
    """
    assert bmj.min_traj_prob_factor_subdir(lmd.DEFAULT_MIN_TRAJ_PROB_FACTOR) == ""
    assert bmj.min_traj_prob_factor_subdir(2.5) == "mpf_2p5"

    pf_default = pfeiffer_events.RunSetting()
    pf_swept = pfeiffer_events.RunSetting(min_traj_prob_factor=2.5)
    assert "mpf" not in pf_default.tag
    assert pf_default.tag != pf_swept.tag
    assert pf_swept.tag.startswith(pf_default.tag)

    cfg = config.DecodeConfig()
    kw = dict(
        decode_tbin_s=cfg.decode_tbin_s,
        decode_tbin_step_s=cfg.decode_tbin_step_s,
        hover_bins=0,
    )
    lu_default = lmj.RunSetting(**kw)
    lu_swept = lmj.RunSetting(**kw, min_traj_prob_factor=2.5)
    assert "mpf" not in lu_default.tag
    assert lu_default.tag != lu_swept.tag

    # The Buzsaki subdirectory token is the one the Pfeiffer tag uses for the same factor.
    assert pf_swept.tag[len(pf_default.tag):].lstrip("_") == bmj.min_traj_prob_factor_subdir(2.5)


def test_lustig_forwards_a_swept_factor_into_the_shared_config():
    """A swept factor reaches the DecodeConfig; dropping it would let the flag parse and the
    tag look right while the run proceeds at the default.
    """
    cfg = config.DecodeConfig()
    setting = lmj.RunSetting(
        decode_tbin_s=cfg.decode_tbin_s,
        decode_tbin_step_s=cfg.decode_tbin_step_s,
        hover_bins=0,
        min_traj_prob_factor=3.0,
    )
    assert lmj._cfg_from_setting(setting, cfg.bin_width_px).min_traj_prob_factor == 3.0
    assert "min_traj_prob_factor" in lmj.SENSITIVITY_PARAM_ORDER


# ── Step sizes: |Δ argmax| with self-transitions kept ─────────────────────────────────

def _all_three_steps(post, factor=FACTOR, hover=0):
    """Step sizes from each dataset's own entry point, over the same map-0 slice.

    Lustig has no step wrapper of its own: `_compute_trial_metrics` (and the traditional
    path) call `decode.poisson.step_sizes_from_map_posterior` on the span's map slice
    directly, so that is what is read.
    `hover` is threaded through every dataset's hover knob to show the steps ignore it.
    Buzsaki's step sizes take no hover argument at all (its hover threshold is a
    `compute_jumps` argument), so there is no knob to thread it through.
    """
    T = post.shape[0]
    lu = poisson.step_sizes_from_map_posterior(post[:, :N_BINS], factor)
    assert "hover_thresh_bins" not in inspect.signature(lmd.compute_step_sizes).parameters
    bz = lmd.compute_step_sizes({"map_posterior": post[:, :N_BINS]}, min_prob_factor=factor)
    pf = pfeiffer_session.compute_step_sizes(
        post,
        {ratemaps.MAP_OUTBOUND: [(0, T)]},
        BOUNDS,
        pfeiffer_events.RunSetting(min_traj_prob_factor=factor, hover_thresh_bins=hover),
    )
    return {"lustig": lu, "buzsaki": bz, "pfeiffer": pf}


def test_step_sizes_keep_self_transitions_and_drop_the_sign():
    for path, want in (
        ([2, 2, 4, 4, 4, 7], [0.0, 2.0, 0.0, 0.0, 3.0]),
        ([7, 4, 4, 2], [3.0, 0.0, 2.0]),
    ):
        for name, steps in _all_three_steps(_joint_posterior(path)).items():
            np.testing.assert_allclose(steps, want, err_msg=f"{name} {path}")


def test_an_unconfident_window_breaks_the_step_in_every_dataset():
    """Window 2 is flat, so neither the 2->? nor the ?->4 step exists -- no bridging."""
    got = _all_three_steps(_joint_posterior([2, 2, 4, 4, 4, 7], unconfident={2}))
    for name, steps in got.items():
        np.testing.assert_allclose(steps, [0.0, 0.0, 3.0], err_msg=name)


def test_nonzero_steps_are_exactly_the_jump_magnitudes():
    """The defining invariant: steps minus the zeros == |jumps| at hover 0, per dataset."""
    post = _joint_posterior([2, 2, 4, 4, 4, 7, 7, 3, 9, 9, 8], unconfident={5})
    jumps = _all_four(post)
    for name, steps in _all_three_steps(post).items():
        np.testing.assert_allclose(
            np.sort(steps[steps > 0]), np.sort(np.abs(jumps[name])), err_msg=name
        )
        assert np.sum(steps == 0) > 0, f"{name} dropped the self-transitions"


def test_step_sizes_ignore_the_hover_threshold():
    post = _joint_posterior([2, 3, 3, 5, 6])
    base = _all_three_steps(post, hover=0)
    hover1 = _all_three_steps(post, hover=1)
    for name in base:
        np.testing.assert_allclose(base[name], [1.0, 0.0, 2.0, 1.0], err_msg=name)
        np.testing.assert_allclose(hover1[name], base[name], err_msg=name)


@pytest.mark.parametrize("shape", [(0, N_BINS), (1, N_BINS), (4, 0)])
def test_step_sizes_handle_degenerate_slices(shape):
    steps = poisson.step_sizes_from_map_posterior(np.zeros(shape), FACTOR)
    assert steps.shape == (0,)
