"""Decode grids: `DecodeGrid`, the window spec each dataset's `grids.py` defaults are written in;
the Lustig sliding decode windows on the 20 kHz spike timebase; and event-bin selection.

`decode_window_grid` is the only place the Lustig grid is built. `SPK_SAMP_RATE` and
`LFP_SAMP_RATE` are the two timebases it converts between; they live here rather than in
`datasets.lustig.constants` because this generic layer may not import a dataset.
"""
from dataclasses import dataclass

import numpy as np

from replay_hmm.decode.config import DecodeConfig

SPK_SAMP_RATE = 20000.0 # Hz
LFP_SAMP_RATE = 1250.0 # Hz


@dataclass(frozen=True)
class DecodeGrid:
    """One decode's sliding windows: `tbin_s` wide, one every `tbin_step_s` (seconds).

    Passed explicitly, whole or as one of its fields, to everything that tiles or measures
    decode windows (`decode_pbe`, `events_from_ripples`' two-window floor, the
    `tbin_step_s` of `call_trajectories`' one-window step) and as the `tbin_s` of the
    Poisson cache's window length.
    """
    tbin_s: float
    tbin_step_s: float


def lfp_to_spk_index(lfp_index):
    """LFP-timebase (1250 Hz) sample index -> 20 kHz spike-timebase index."""
    return int(np.floor(lfp_index / LFP_SAMP_RATE * SPK_SAMP_RATE))


def decode_window_grid(trial_start_lfp, trial_end_lfp, cfg: DecodeConfig):
    """Sliding decode windows for one lap, in the 20 kHz spike timebase.

    Returns ``(starts_20k, ends_20k, centers_20k)``; each window spans the half-open
    interval ``[start, end)`` and successive starts are one stride apart, so width and
    stride are independent (10 ms windows every 5 ms overlap 2x; 3 ms every 3 ms do not).

    This is the only place the decode grid is constructed. `build_session_bundle` bins its
    spike counts on it and anything plotting a posterior gets its time axis from it, so a
    figure cannot end up drawn against a different axis from the one the counts were binned
    on.
    """
    bin_w_20k = int(round(cfg.decode_tbin_s * SPK_SAMP_RATE)) # decode window width in 20 kHz samples
    step_20k = int(round(cfg.decode_tbin_step_s * SPK_SAMP_RATE)) # decode window stride in 20 kHz samples
    if bin_w_20k <= 0 or step_20k <= 0:
        raise ValueError("decode_tbin_s and decode_tbin_step_s must both be > 0.")
    start_20k = lfp_to_spk_index(trial_start_lfp)
    end_20k = lfp_to_spk_index(trial_end_lfp)

    # Sliding decode windows with independent width and stride.
    # start[i] -> end[i] spans one decode bin.
    dec_starts = np.arange(start_20k, end_20k - bin_w_20k + 1, step_20k, dtype=int)
    dec_ends = dec_starts + bin_w_20k
    centers_20k = dec_starts + 0.5 * bin_w_20k # centers in 20KHz time base
    return dec_starts, dec_ends, centers_20k


def traditional_window_grid(trial_start_lfp, trial_end_lfp, tbin_s, tbin_step_s, *, dtype):
    """The traditional detector's sliding windows for one lap.

    Returns ``(starts_20k, ends_20k, starts_lfp, ends_lfp)``: half-open 20 kHz windows of
    `tbin_s` every `tbin_step_s`, and each edge floored onto the 1250 Hz LFP timebase, all of
    element type `dtype`. The windows are `decode_window_grid`'s, without its config and its
    positivity check; the LFP edges are what `datasets.lustig.traditional.process_session`
    slices an event with and what `analyses.ppc.traditional` finds a saved event's windows by. A
    lap shorter than one window has no windows: empty arrays, which `process_session` skips.
    """
    bin_w_20k = int(round(tbin_s * SPK_SAMP_RATE))
    step_20k = int(round(tbin_step_s * SPK_SAMP_RATE))
    start_20k = lfp_to_spk_index(trial_start_lfp)
    end_20k = lfp_to_spk_index(trial_end_lfp)
    starts_20k = np.arange(start_20k, end_20k - bin_w_20k + 1, step_20k, dtype=dtype)
    ends_20k = starts_20k + bin_w_20k
    starts_lfp = np.floor(starts_20k / SPK_SAMP_RATE * LFP_SAMP_RATE).astype(dtype)
    ends_lfp = np.floor(ends_20k / SPK_SAMP_RATE * LFP_SAMP_RATE).astype(dtype)
    return starts_20k, ends_20k, starts_lfp, ends_lfp


def select_event_bins(
    dec_starts_20k: np.ndarray,
    dec_ends_20k: np.ndarray,
    event_start_lfp: int,
    event_end_lfp: int,
) -> np.ndarray:
    """Indices of the lap's decode bins belonging to one externally supplied event.

    A bin belongs to the event when at least half its span lies inside it, the
    `coverage_half` rule `build_session_bundle` uses to reduce a PBE mask onto the decode
    timebase. The grid is built for the whole lap and then sliced, so these are the same bins
    the MB/HMM pipeline would produce for this lap; starting a fresh grid at the event start
    would be off-phase from the pipeline for no gain.

    Lives here, next to `decode_window_grid`, so the event-span-to-decode-bin rule has one
    definition shared by every analysis that takes its event set from a times table.
    """
    start_20k = float(np.floor(event_start_lfp / LFP_SAMP_RATE * SPK_SAMP_RATE))
    end_20k = float(np.floor(event_end_lfp / LFP_SAMP_RATE * SPK_SAMP_RATE))
    starts = np.asarray(dec_starts_20k, dtype=float)
    ends = np.asarray(dec_ends_20k, dtype=float)
    widths = ends - starts
    overlap = np.clip(np.minimum(ends, end_20k) - np.maximum(starts, start_20k), 0.0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.where(widths > 0, overlap / widths, 0.0)
    return np.flatnonzero(coverage >= 0.5)


def assert_contiguous_selection(selected: np.ndarray, context: str) -> None:
    """The event span is contiguous, so its bins must be too; a gap means a grid/phase bug."""
    if selected.size == 0:
        return
    if not np.array_equal(selected, np.arange(int(selected[0]), int(selected[-1]) + 1)):
        raise ValueError(
            f"{context}: event decode bins are not contiguous ({selected.size} bins spanning "
            f"{int(selected[-1]) - int(selected[0]) + 1} slots); the lap decode grid and the "
            "event span are out of phase."
        )
