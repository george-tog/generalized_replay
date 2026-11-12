"""Buzsaki traversal segmentation (`analyses.diagnostics.traversals`) and the decoders' cell set."""

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

try:
    from replay_hmm.analyses.diagnostics.traversals import _mask_from_runs, segment_traversals
    from replay_hmm.datasets.buzsaki.ratemaps import SPEED_THRESH, directional_rate_maps
except Exception as exc:  # pragma: no cover - import guard for local envs
    segment_traversals = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

pytestmark = pytest.mark.skipif(
    segment_traversals is None, reason=f"import failed: {_IMPORT_ERROR!r}"
)


# ── Traversal segmentation ────────────────────────────────────────────────────

def _synthetic_track(n_laps: int = 4, samples: int = 50, track_len: float = 1.6):
    """Alternating full-track sweeps separated by stationary pauses."""
    pos, direction = [], []
    for lap in range(n_laps):
        sweep = np.linspace(0.0, track_len, samples)
        if lap % 2:
            sweep = sweep[::-1]
        pos.append(sweep)
        direction.append(np.full(samples, 1.0 if lap % 2 == 0 else -1.0))
        pos.append(np.full(5, sweep[-1]))          # pause at the reward end
        direction.append(np.zeros(5))
    return np.concatenate(pos), np.concatenate(direction)


def test_segment_traversals_recovers_full_sweeps():
    pos, direction = _synthetic_track(n_laps=4)
    kept, rejected = segment_traversals(
        direction, pos, dt=0.025, track_len=1.6, dir_value=1.0,
        merge_gap_s=0.0, min_span_frac=0.6,
    )
    assert len(kept) == 2 and rejected == []
    for s, e in kept:
        assert np.all(direction[s:e] == 1.0)
        assert pos[s:e].max() - pos[s:e].min() >= 0.6 * 1.6


def test_segment_traversals_rejects_short_wobbles():
    # One full sweep plus a 10 cm wobble; only the sweep should survive.
    pos = np.concatenate([np.linspace(0.0, 1.6, 50), np.zeros(3), np.linspace(0.2, 0.3, 8)])
    direction = np.concatenate([np.ones(50), np.zeros(3), np.ones(8)])
    kept, rejected = segment_traversals(
        direction, pos, dt=0.025, track_len=1.6, dir_value=1.0,
        merge_gap_s=0.0, min_span_frac=0.6,
    )
    assert len(kept) == 1 and len(rejected) == 1


def test_segment_traversals_merges_across_short_pause():
    # A single sweep interrupted by a 2-sample pause: merged at a 1 s gap
    # tolerance (dt=0.5 s -> max_gap=2), split when the tolerance is zero.
    pos = np.concatenate([np.linspace(0.0, 0.8, 25), np.full(2, 0.8),
                          np.linspace(0.8, 1.6, 25)])
    direction = np.concatenate([np.ones(25), np.zeros(2), np.ones(25)])
    merged, _ = segment_traversals(
        direction, pos, dt=0.5, track_len=1.6, dir_value=1.0,
        merge_gap_s=1.0, min_span_frac=0.6,
    )
    split, rejected = segment_traversals(
        direction, pos, dt=0.5, track_len=1.6, dir_value=1.0,
        merge_gap_s=0.0, min_span_frac=0.6,
    )
    assert len(merged) == 1
    assert len(split) == 0 and len(rejected) == 2


def test_mask_from_runs_is_end_exclusive():
    mask = _mask_from_runs(10, [(2, 5), (7, 8)])
    assert mask.tolist() == [False, False, True, True, True,
                             False, False, True, False, False]


def test_speed_threshold_is_the_shared_constant():
    assert SPEED_THRESH == pytest.approx(0.05)


# ── Cell set must match the decoders ──────────────────────────────────────────

def test_decoders_use_unfiltered_pyr_ids():
    """The Buzsaki decoders build their maps from every pyramidal cell, unfiltered.

    The three decoders share one rate-map function, so the source check is on it, and each
    decoder is checked to build its maps with it (and not a copy).
    """
    import inspect

    from replay_hmm.datasets.buzsaki import fit, mb_jump, traditional

    src = inspect.getsource(directional_rate_maps)
    assert 'pyr_ids = sorted(set(data["pyr_ids"].tolist()))' in src
    assert "MIN_SPIKES" not in src
    assert "place_cell" not in src
    for caller in (mb_jump.analyse_replay, traditional.process_session, fit.run_one):
        module = inspect.getmodule(caller)
        assert module.directional_rate_maps is directional_rate_maps, module.__name__
        assert "directional_rate_maps(" in inspect.getsource(caller), caller.__qualname__
        assert "_occ_and_rate" not in inspect.getsource(module), module.__name__
