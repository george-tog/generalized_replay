"""The interactive per-event HMM browser, shared by every dataset (`replay-hmm viewer`).

Time-aligned panels for one PBE at a time:

    z-PFR       z-scored population spike density, the trace PBEs were called from
    posterior   HMM smoothed posterior (forward-backward) over the stacked maps, one row
                per kernel, with the Viterbi path overlaid (and, with --highlight-mb-traj,
                the called trajectory segments shaded)
    raster      spike raster of the decode units, sorted by field peak on the
                event's dominant map

Beside them: an info block, a zoom square per kernel on its dominant map and, with a single
kernel only, this event's Viterbi jumps over a faint session distribution.

Controls: left/right arrows, n/p or the Previous/Next buttons step events, the slider seeks,
`s` or the Save button saves the current figure.

`EventViewer` and `run_viewer` read a session only through a `ViewerSource`;
`replay_hmm.viewer.lustig` builds Lustig's (`LustigSource`) and `replay_hmm.viewer.two_map`
Pfeiffer's and Buzsaki's.
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.widgets import Button, Slider
from joblib import Parallel, delayed

from replay_hmm.hmm import constants
from replay_hmm.jumps import stats
from replay_hmm.plots.style import COLOR_PALETTE
from replay_hmm.plots.labels import kernel_label
from replay_hmm.jumps.signed import signed_jumps_from_state_sequence


# --jump-scope names, keyed by the `jump_event_source` a `replay-hmm jumps` run records.
JUMP_SCOPE_BY_EVENT_SOURCE = {
    "mb_called_trajectories": "traj",
    "all_pbes": "full-pbe",
    "traditional": "traditional",
}

# --jump-scope values. `traditional` is the traditional-replay event set: the events a
# traditional classifier table names, each scored over its whole window -- what a __tradreplay
# fit is trained on. Its records are a different event list from the PBE scopes', so the
# dataset CLIs build them; the engine only needs to know its spans are whole windows.
JUMP_SCOPES = ("traj", "full-pbe", "traditional")
WHOLE_WINDOW_SCOPES = ("full-pbe", "traditional")
# --jump-scope by the `event_source` a fit CLI records for its training set.
FIT_JUMP_SCOPE_BY_EVENT_SOURCE = {
    "mb_traj": "traj",
    "all_pbe": "full-pbe",
    "traditional": "traditional",
}


def fit_jump_scope(event_source) -> str | None:
    """The --jump-scope matching a fit's recorded `event_source`; None for any other set."""
    return FIT_JUMP_SCOPE_BY_EVENT_SOURCE.get(event_source)

DPI = 400
DEFAULT_KERNEL = "sticky_TCMSR"

# Figure geometry. The layout and every font size below are set for a 12 in wide figure; text
# scales with FIG_W / 12.
FIG_W = 12.0
FIG_H_ONE = 6.0
FIG_H_TWO = 8.00
FONT_SCALE = FIG_W / 12.0


def _fs(points: float) -> float:
    """A point size from the 12 in design, scaled to the current figure width."""
    return points * FONT_SCALE

# Posterior colour ceiling, FIXED so one event's heatmap can be read against another's and
# one kernel's against another's; per-event scaling would make every panel look equally
# confident. 0.10 sits above the typical per-event 99th percentile, so the typical peak is not
# clipped; individual bins can reach 1.0 and saturate. Override with --posterior-vmax.
POSTERIOR_VMAX = 0.10

POSTERIOR_CMAP = LinearSegmentedColormap.from_list(
    "kernel",
    ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#6e016b'],
)

# The Viterbi overlay sits on the purple posterior heatmap, so it is drawn in black for contrast.
VITERBI_COLOR = 'black'
TRAJ_SPAN_COLOR = COLOR_PALETTE[2]
ZPFR_COLOR = COLOR_PALETTE[0]
JUMP_COLOR = COLOR_PALETTE[1]
# Must not read as the Viterbi path, which is black.
POSITION_COLOR = COLOR_PALETTE[4]


# --------------------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------------------

@dataclass
class EventRecord:
    """One PBE-and-nonrun run, in decode-bin indices local to its lap.

    The two-map datasets reuse it with bins local to the event's own padded window, `trial`
    unused (0) and `event_idx` the session_pbe_index.
    """
    trial: int
    event_idx: int          # index within the lap, matching traj_segments_by_pbe[trial]
    s_bin: int
    e_bin: int
    t0_s: float             # absolute start time, seconds
    t1_s: float
    peak_z: float

    @property
    def n_bins(self) -> int:
        return self.e_bin - self.s_bin

    @property
    def duration_ms(self) -> float:
        return (self.t1_s - self.t0_s) * 1000.0


# --------------------------------------------------------------------------------------
# jumps
# --------------------------------------------------------------------------------------

def event_loglik(arrays_list: list[np.ndarray], hmm) -> float:
    """Total log-likelihood this event contributes under `hmm`, in nats.

    Summed over the arrays `event_spike_arrays` returned, which at `traj` scope is one per
    called trajectory segment and at `full-pbe` scope is the single whole window. Summing
    this over a session's events reproduces the fit's saved `loglik` exactly, so a per-event
    difference between two kernels is an exact decomposition of the session-level comparison
    rather than a proxy for it.

    NaN when the event contributes nothing (no called segments), which must stay
    distinguishable from a genuine 0.0 -- those events have no opinion about the kernels and
    have to sort apart from the ones that do.
    """
    parts = [np.asarray(a) for a in arrays_list if np.asarray(a).shape[0] >= 1]
    if not parts:
        return float("nan")
    return float(sum(hmm.log_likelihood([p]) for p in parts))


def warn_if_sigma_capped(kernel: str, summary_row: dict, sigma_cap: float | None = None) -> None:
    """Flag a continuous_EM fit whose Gaussian width sat on its cap.

    At the cap the kernel wanted a wider Gaussian than it was allowed, so a likelihood gap
    partly measures the cap rather than the kernel family.

    `sigma_cap` is the cap THIS fit was built with, which is not the module default for a
    non-default-cap fit -- comparing against `constants.SIGMA_CAP` there would report such a
    fit as having exceeded a cap it was never fitted under.
    """
    cap = float(constants.SIGMA_CAP if sigma_cap is None else sigma_cap)
    for name in ("sigma_arm", "sigma_wheel"):
        raw = summary_row.get(name)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        if abs(val - cap) <= 0.01 * cap:
            warnings.warn(
                f"{kernel}: {name}={val:.4f} is at sigma_cap={cap}. The kernel is "
                "clipped, so a likelihood difference here partly reflects the cap rather "
                "than the transition family.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif val > cap:
            warnings.warn(
                f"{kernel}: {name}={val:.4f} exceeds sigma_cap={cap}, so this fit was made "
                "under a different cap. Do not compare its likelihoods against fits made "
                "under this one.",
                RuntimeWarning,
                stacklevel=2,
            )


def _frac_in_map(dec: dict, bounds: np.ndarray, m: int) -> float:
    """Fraction of the padded Viterbi path lying inside map `m` -- i.e. how much of the event
    the arm-zoom panel actually shows."""
    z = np.asarray(dec["z_pad"])
    if z.size == 0:
        return float("nan")
    return float(((z >= int(bounds[m])) & (z < int(bounds[m + 1]))).mean())


def asymmetry_text(jumps: np.ndarray) -> str:
    """`N=..., asym=...` for one event -- the index only, no permutation test.

    Reuses `stats.asymmetry_2sided_pvalue` rather than recomputing the index, so the two can
    never disagree, but with `n_perm=1`: the index does not depend on the permutations, and
    drawing 50k of them per event to then discard the p-value is pure waste.
    """
    asym, _pvalue, n = stats.asymmetry_2sided_pvalue(
        np.asarray(jumps, dtype=float), n_perm=1, seed=0,
    )
    if not np.isfinite(asym):
        return f"N={n}, asym=NA"
    return f"N={n}, asym={asym:.3g}"


def jumps_for_arrays(
    arrays_list: list[np.ndarray], hmm, lin_map_bounds, signed_jumps=None,
) -> np.ndarray:
    """Viterbi jumps over each array independently, concatenated.

    `signed_jumps(z, bounds)` is Lustig's `signed_jumps_from_state_sequence` unless a dataset
    passes its own (`ViewerSource.signed_jumps`).
    """
    signed_jumps = signed_jumps or signed_jumps_from_state_sequence
    chunks = [
        signed_jumps(hmm.most_likely_states(a), lin_map_bounds)
        for a in arrays_list if a.shape[0] >= 2
    ]
    chunks = [c for c in chunks if c.size]
    return np.concatenate(chunks) if chunks else np.array([], dtype=int)


# --------------------------------------------------------------------------------------
# dataset hook
# --------------------------------------------------------------------------------------

@dataclass
class DensityTrace:
    """The detector trace over one panel's time range, for the z-PFR row."""
    t_ms: np.ndarray                    # relative to the event start
    values: np.ndarray
    thresholds: tuple                   # (peak, boundary): dashed, then solid; may be empty
    ineligible_ms: list                 # [(a_ms, b_ms)] spans no event may start in
    ylabel: str = "z spike density"


class ViewerSource:
    """What `EventViewer` reads from a dataset. Returns data only; the viewer does the drawing.

    `viewer.lustig.LustigSource` is Lustig's, `viewer.two_map.TwoMapSource` serves Pfeiffer and
    Buzsaki. Record bins index whatever array `counts` slices -- a lap for Lustig, the event's
    own padded window for the two-map datasets.
    """

    name: str                           # "TinyTim/rec01": suptitle and the compact info block
    name_spaced: str                    # "TinyTim / rec01": the single-kernel info block
    decode_tbin_s: float
    decode_tbin_step_s: float
    bounds: np.ndarray                  # map boundaries over the stacked state axis
    lin_map_concat: np.ndarray          # (n_units, K) Hz; the raster sorts on it
    map_labels: tuple
    arm_maps: tuple                     # a prefix of the map order: dominant-map candidates
    cm_per_bin: float
    max_map_bins: int                   # widest map in bins: the jump histogram's half-range
    jump_axis_labels: tuple = ("- inbound", "+ outbound")

    def n_bins_total(self, rec: EventRecord) -> int:
        raise NotImplementedError

    def counts(self, rec: EventRecord, lo: int, hi: int) -> np.ndarray:
        """(hi - lo, n_units) int32 spike counts."""
        raise NotImplementedError

    def bin_centers_s(self, rec: EventRecord, lo: int, hi: int) -> np.ndarray:
        """Absolute decode-bin centre times over [lo, hi), seconds."""
        raise NotImplementedError

    def traj_segments(self, rec: EventRecord) -> list[tuple[int, int, int]]:
        """Called trajectory segments `[(map, s, e)]`, in the fit collector's order."""
        raise NotImplementedError

    def scope_spans(self, rec: EventRecord, scope: str) -> list[tuple[int, int]]:
        """The bin spans the jumps and log-likelihood come from, per `--jump-scope`."""
        if scope in WHOLE_WINDOW_SCOPES:
            return [(rec.s_bin, rec.e_bin)]
        return [(s, e) for _m, s, e in self.traj_segments(rec)]

    def scope_arrays(self, rec: EventRecord, scope: str) -> list[np.ndarray]:
        """The spike-count arrays this event contributes at `scope`, one per span."""
        return [self.counts(rec, s, e) for s, e in self.scope_spans(rec, scope)]

    def signed_jumps(self, z: np.ndarray, bounds) -> np.ndarray:
        raise NotImplementedError

    def density(self, rec: EventRecord, t_lo_ms: float, t_hi_ms: float) -> DensityTrace | None:
        raise NotImplementedError

    def raster_ms(self, row: int, rec: EventRecord, t_lo_ms: float, t_hi_ms: float) -> np.ndarray:
        """Unit `row`'s spike times inside the window, in ms from the event start."""
        raise NotImplementedError

    def position_lines(self, rec: EventRecord, t_lo_ms: float, t_hi_ms: float):
        """`([(t_ms, y_state), ...], description)`: the animal's position on the state axis.

        Every line is returned, NaN where it does not apply; the viewer skips all-NaN ones.
        The description goes in the info block, so an absent line is interpretable.
        """
        raise NotImplementedError

    def event_label(self, rec: EventRecord) -> str:
        raise NotImplementedError

    def event_file_tag(self, rec: EventRecord) -> str:
        raise NotImplementedError


class EventViewer:
    """Slider-driven browser over one session's events."""

    def __init__(
        self,
        *,
        source: ViewerSource,
        hmm,
        kernel: str,
        run_tag: str,
        records: list[EventRecord],
        scope: str,
        backdrop: np.ndarray | None,
        pad_ms: float,
        initial_index: int,
        save_dir: Path,
        show_position: bool = True,
        highlight_mb_traj: bool = False,
        jumps_cache: dict[int, np.ndarray] | None = None,
        compare_hmms: list | None = None,
        compare_kernels: list[str] | None = None,
        ll_cache: dict[int, tuple[float, ...]] | None = None,
        posterior_vmax: float = POSTERIOR_VMAX,
    ) -> None:
        self.source = source
        self.hmm = hmm
        self.kernel = kernel
        # Base kernel first, then the compare kernels in the order given. Everything
        # per-kernel is indexed by position in these two lists.
        self.compare_hmms = list(compare_hmms or [])
        self.compare_kernels = list(compare_kernels or [])
        self.hmms = [hmm, *self.compare_hmms]
        self.kernels = [kernel, *self.compare_kernels]
        self.n_kernels = len(self.hmms)
        self.multi = self.n_kernels > 1
        self.posterior_vmax = float(posterior_vmax)
        # One LL per kernel per event, in `self.kernels` order. Memoised as the tuple rather
        # than as a difference so the info panel can print every kernel without rescoring.
        self.ll_cache: dict[int, tuple[float, ...]] = dict(ll_cache or {})
        self.run_tag = run_tag
        self.records = records
        self.scope = scope
        self.backdrop = backdrop
        self.pad_ms = float(pad_ms)
        self.save_dir = Path(save_dir)
        self.show_position = bool(show_position)
        self.highlight_mb_traj = bool(highlight_mb_traj)
        self.bounds = np.asarray(source.bounds, dtype=int)
        self.n_maps = len(self.bounds) - 1
        self.map_labels = tuple(source.map_labels)
        self.cm_per_bin = source.cm_per_bin
        self.pad_bins = int(np.ceil((self.pad_ms / 1000.0) / source.decode_tbin_step_s))

        self.decode_caches: list[dict[int, dict]] = [{} for _ in range(self.n_kernels)]
        self.decode_cache = self.decode_caches[0]
        self.jumps_cache: dict[int, np.ndarray] = dict(jumps_cache or {})
        self.current_index = int(np.clip(initial_index, 0, len(records) - 1))

        max_arm = int(source.max_map_bins)
        # Histogram at the native one-position-bin resolution, but drawn on a cm axis: the
        # edges are rescaled, not rebinned, so no counts move between bars.
        self.jump_bins = np.arange(-max_arm, max_arm + 2) - 0.5
        self.jump_bins_cm = self.jump_bins * self.cm_per_bin
        pool = [j for j in (self.backdrop,) if j is not None and np.size(j)]
        pool += [j for j in self.jumps_cache.values() if np.size(j)]
        span = max((float(np.abs(j).max()) for j in pool), default=0.0)
        self.jump_lim_cm = (
            max(5.0, 1.1 * span) if span else float(max_arm)
        ) * self.cm_per_bin

        # Comparison mode inserts a heatmap row per compare kernel.
        # Tick labels, axis labels, the legend and the widget captions all take their size
        # from rcParams, and ax.clear() in every render resets them -- so the scale has to be
        # in force during render(), not just here. `medium` and friends resolve off font.size,
        # so scaling that one entry carries the rest.
        self._rc = {"font.size": matplotlib.rcParams["font.size"] * FONT_SCALE}
        with matplotlib.rc_context(self._rc):
            self._build_figure(records, initial_index)

    def _build_figure(self, records, initial_index: int) -> None:
        """One heatmap row per kernel, each with a square zoom beside it.

        Column 0 is z-PFR / one posterior per kernel / raster. Column 1 is the info block,
        then a zoom square per kernel, and -- with a single kernel only -- the jump histogram.
        Comparing kernels drops the histogram: the interesting contrast is between the
        posteriors, and the jumps come from the base kernel alone so one histogram beside
        three heatmaps would invite reading it as belonging to all of them.
        """
        n = self.n_kernels
        multi = self.multi
        # Rows: z-PFR, one posterior per kernel, raster.
        height_ratios = (0.667, *(3.2,) * n, 2.0)
        n_rows = len(height_ratios)
        raster_row = n_rows - 1
        fig_h = FIG_H_ONE if n == 1 else FIG_H_TWO + 2.4 * (n - 2)
        self.fig = plt.figure(figsize=(FIG_W, fig_h), dpi=110)
        grid = self.fig.add_gridspec(
            n_rows, 2,
            # Column 1 is widened when comparing: the squares are the point there, and with
            # several kernels the heatmap rows are taller than the narrow column, which would
            # make the squares width-limited and so unable to sit flush with their rows.
            width_ratios=(3.6, 1.0) if multi else (4.2, 1.0),
            height_ratios=height_ratios,
            hspace=0.12, wspace=0.30,
            left=0.085, right=0.93,
            top=0.94 if multi else 0.93, bottom=0.11 if multi else 0.13,
        )
        self.ax_zpfr = self.fig.add_subplot(grid[0, 0])
        self.ax_posts = [
            self.fig.add_subplot(grid[1 + i, 0], sharex=self.ax_zpfr) for i in range(n)
        ]
        self.ax_post = self.ax_posts[0]
        self.ax_raster = self.fig.add_subplot(grid[raster_row, 0], sharex=self.ax_zpfr)

        # Squares sit beside their own kernel's heatmap row. With one kernel there is no
        # per-kernel row to align to, so it keeps its place beside the raster and the
        # histogram takes the posterior row.
        #
        # The info block goes wherever column 1 has a free cell. With one kernel that is row 0,
        # where it overflows harmlessly into the spacer above the histogram. When comparing,
        # every heatmap row is taken by a square, so it moves to the raster row -- row 0 is far
        # too short for the per-kernel logLik table and the text would run over the first
        # square and its colourbar.
        info_row = raster_row if multi else 0
        self.ax_info = self.fig.add_subplot(grid[info_row, 1])
        self.ax_info.axis("off")
        if multi:
            zoom_rows = [1 + i for i in range(n)]
            self.ax_jump = None
        else:
            zoom_rows = [raster_row]
            # Histogram in the upper-middle of its row, with a spacer band below for its own
            # tick labels and xlabel: the zoom under it is flush with the raster and leaves no
            # slack, so without the spacer the xlabel is clipped off the figure.
            jump_cell = grid[1, 1].subgridspec(
                3, 1, height_ratios=(0.55, 1.0, 0.45), hspace=0.0,
            )
            self.ax_jump = self.fig.add_subplot(jump_cell[1])

        # Deliberately NOT sharex: Axes.sharex shares the major/minor Tickers, so a locator
        # coarse enough for these narrow squares would also strip ticks off the wide panels.
        # Each square's time range is set explicitly in _plot_arm_zoom instead.
        #
        # Each takes its whole cell so its top and bottom sit flush with the row it belongs
        # to: the square is height-limited (the cell is wider than it is tall), so filling the
        # cell height is exactly what aligns it. It is anchored west so the colourbar has room
        # inside the cell rather than off the figure edge.
        self.ax_zooms = []
        self.zoom_caxes = []
        for i, r in enumerate(zoom_rows):
            # Each square gets a sub-cell that reserves a band on its right for the colourbar
            # and a band below for its own time axis, so its xlabel stays clear of the next
            # square and the bar never has to overlap the heatmap.
            zoom_cell = grid[r, 1].subgridspec(
                2, 2, width_ratios=(1.0, 0.18), height_ratios=(1.0, 0.22),
                wspace=0.0, hspace=0.0,
            )
            ax = self.fig.add_subplot(zoom_cell[0, 0])
            # North-west: left, so the colourbar sits in its reserved band, and top, so the
            # square's top edge stays flush with the row it belongs to even when it is
            # width-limited and cannot fill the cell height.
            ax.set_anchor("NW")
            self.ax_zooms.append(ax)
            # Only the top square carries a bar. The range is fixed and identical everywhere,
            # so one is enough for the whole figure and the lower squares stay uncluttered.
            self.zoom_caxes.append(
                self._add_zoom_colorbar_axes(zoom_cell[0, 0], fig_h) if i == 0 else None
            )
        self.ax_zoom = self.ax_zooms[0]

        # The main heatmaps carry no colourbar: the scale is fixed and shared with the squares,
        # so the bar beside each square describes them too.
        self.zoom_colorbars: list = [None] * len(self.ax_zooms)

        slider_ax = self.fig.add_axes((0.24, 0.045, 0.50, 0.022))
        prev_ax = self.fig.add_axes((0.03, 0.028, 0.075, 0.045))
        next_ax = self.fig.add_axes((0.115, 0.028, 0.075, 0.045))
        save_ax = self.fig.add_axes((0.80, 0.028, 0.075, 0.045))
        self.slider = Slider(
            slider_ax, "event", 0, max(len(records) - 1, 1),
            valinit=self.current_index, valstep=1,
        )
        self.prev_button = Button(prev_ax, "Previous")
        self.next_button = Button(next_ax, "Next")
        self.save_button = Button(save_ax, "Save")
        self.slider.on_changed(self._on_slider)
        self.prev_button.on_clicked(lambda _e: self.step(-1))
        self.next_button.on_clicked(lambda _e: self.step(1))
        self.save_button.on_clicked(lambda _e: self.save_current())
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.render(self.current_index)

    def _add_zoom_colorbar_axes(self, spec, fig_h: float):
        """Colourbar axes to the right of a west-anchored square filling `spec`.

        The square's drawn width is not known from `ax.get_position()` at build time -- that
        returns the whole cell, and `set_box_aspect` only shrinks it at draw. Since the cell is
        wider than it is tall, the square is height-limited, so its width in figure fractions
        is its height converted through the figure's aspect. Computing that here avoids
        needing a draw before the colourbar can be placed.
        """
        cell = spec.get_position(self.fig)
        # Square side in figure fractions, in each axis. Whichever of the cell's two extents
        # is smaller binds, exactly as set_box_aspect will decide at draw time.
        side_w = min(cell.width, cell.height * fig_h / FIG_W)
        side_h = side_w * FIG_W / fig_h
        # Never clamped back inside `cell`: `spec` is the square's own sub-cell and the caller
        # reserves a band to its right for this bar; clamping would push the bar onto the
        # heatmap whenever the square fills its cell's width.
        x0 = cell.x0 + side_w + 0.008
        # The square is anchored NW, so it hangs from the top of the cell.
        return self.fig.add_axes((
            x0, cell.y1 - 0.75 * side_h, 0.009, 0.5 * side_h,
        ))

    # -- state -------------------------------------------------------------------------

    def _on_slider(self, value) -> None:
        idx = int(round(float(value)))
        if idx != self.current_index:
            self.render(idx)

    def step(self, delta: int) -> None:
        idx = int(np.clip(self.current_index + delta, 0, len(self.records) - 1))
        if idx != self.current_index:
            self.slider.set_val(idx)

    def _on_key(self, event) -> None:
        if event.key in ("right", "n"):
            self.step(1)
        elif event.key in ("left", "p"):
            self.step(-1)
        elif event.key == "s":
            self.save_current()

    # -- decode ------------------------------------------------------------------------

    def _decode(self, index: int) -> dict:
        """Base-kernel posterior and Viterbi for one event."""
        return self._decode_k(index, 0)

    def _decode_k(self, index: int, k: int) -> dict:
        """Posterior and Viterbi for one event under kernel `k` of `self.kernels`."""
        return self._decode_for(index, self.hmms[k], self.decode_caches[k])

    def _decode_for(self, index: int, hmm, cache: dict) -> dict:
        """Smoothed posterior and Viterbi for one event, memoized.

        Two passes. The padded window drives the heatmap and a faint Viterbi so the image
        is continuous across the event boundary; the jump-generating Viterbi comes from
        the unpadded scope arrays, so the pad stays cosmetic and cannot change a jump.
        """
        if index in cache:
            return cache[index]

        rec = self.records[index]
        src = self.source
        n_trial_bins = src.n_bins_total(rec)
        p_s = max(0, rec.s_bin - self.pad_bins)
        p_e = min(n_trial_bins, rec.e_bin + self.pad_bins)

        x_pad = src.counts(rec, p_s, p_e)
        Ez = hmm.expected_states(x_pad)[0]
        z_pad = hmm.most_likely_states(x_pad)

        segs = src.traj_segments(rec)
        bold = [
            (s, e, hmm.most_likely_states(src.counts(rec, s, e)))
            for s, e in src.scope_spans(rec, self.scope) if e - s >= 2
        ]

        # Which map to sort the raster against. The longest called segment wins when there
        # is one: that is the map the trajectory and every jump live on, so a replay through
        # it reads as a diagonal. Total smoothed mass is only the fallback -- it can name a
        # map that holds diffuse mass all event while the actual trajectory sits elsewhere.
        ev_lo, ev_hi = rec.s_bin - p_s, rec.e_bin - p_s
        mass = np.array([
            Ez[ev_lo:ev_hi, self.bounds[m]:self.bounds[m + 1]].sum()
            for m in src.arm_maps
        ])
        if segs:
            m_star = max(segs, key=lambda t: (t[2] - t[1], mass[t[0]]))[0]
            basis = "longest called segment"
        else:
            m_star = int(np.argmax(mass))
            basis = "most posterior mass"
        out = {
            "p_s": p_s, "p_e": p_e, "Ez": Ez, "z_pad": z_pad,
            "bold": bold, "segs": segs, "dominant_map": int(m_star),
            "sort_basis": basis, "map_mass": mass,
        }
        cache[index] = out
        return out

    def _lls(self, index: int) -> tuple[float, ...]:
        """One log-likelihood per kernel for this event, memoised, in `self.kernels` order.

        NaN throughout when the event contributes no sequences at this scope.
        """
        if index not in self.ll_cache:
            arrs = self.source.scope_arrays(self.records[index], self.scope)
            self.ll_cache[index] = tuple(event_loglik(arrs, h) for h in self.hmms)
        return self.ll_cache[index]

    def _dll(self, index: int) -> float:
        """Per-event ΔLL = LL(base kernel) - LL(first compare kernel), in nats.

        Oriented so POSITIVE favours the main `--kernel`: that is the model under test, so a
        positive number reading as "my kernel wins here" is the way round that does not need
        re-deriving every time the figure is read. With more than two kernels the ranking stays
        this one pair -- the others are reported in the info block but do not reorder the list.
        NaN when the event contributes no sequences at this scope.
        """
        lls = self._lls(index)
        if len(lls) < 2:
            return float("nan")
        return lls[0] - lls[1]

    def _jumps(self, index: int) -> np.ndarray:
        if index not in self.jumps_cache:
            rec = self.records[index]
            self.jumps_cache[index] = jumps_for_arrays(
                self.source.scope_arrays(rec, self.scope),
                self.hmm,
                self.bounds,
                self.source.signed_jumps,
            )
        return self.jumps_cache[index]

    def _times_ms(self, rec: EventRecord, lo: int, hi: int) -> np.ndarray:
        """Decode-bin centre times over [lo, hi), in ms relative to the event start."""
        return (self.source.bin_centers_s(rec, lo, hi) - rec.t0_s) * 1000.0

    # -- panels ------------------------------------------------------------------------

    def _plot_zpfr(self, rec: EventRecord, t_lo_ms: float, t_hi_ms: float) -> None:
        ax = self.ax_zpfr
        trace = self.source.density(rec, t_lo_ms, t_hi_ms)
        if trace is None:
            ax.text(0.5, 0.5, "no detector trace for this event source", ha="center",
                    va="center", transform=ax.transAxes, fontsize=_fs(7), color="0.4")
            ax.set_ylabel("z spike density")
            ax.tick_params(labelbottom=False)
            return
        ax.plot(trace.t_ms, trace.values, color=ZPFR_COLOR, lw=0.9)

        # Detection threshold dashed, event-boundary level solid.
        for level, style in zip(trace.thresholds, (dict(ls="--", lw=0.8, alpha=0.6),
                                                   dict(lw=0.7, alpha=0.4))):
            ax.axhline(level, color="k", **style)
        # PBE-ineligible time: no event may start here, whatever the density does.
        for a, b in trace.ineligible_ms:
            ax.axvspan(a, b, color="0.75", alpha=0.35, lw=0)
        ax.set_ylabel(trace.ylabel)
        ax.tick_params(labelbottom=False)

    def _plot_posterior(
        self, rec: EventRecord, dec: dict, *, ax=None, cax=None,
        kernel: str | None = None, legend: bool = True,
    ) -> tuple[float, float]:
        """One posterior heatmap. `ax` defaults to the base-kernel row.

        The colour ceiling is `self.posterior_vmax`, fixed for every event and every kernel, so
        heatmaps are comparable across both. It is not a per-event percentile, which would make
        a diffuse event, or a kernel with systematically lower peaks, look as confident as a
        sharp one.

        No colourbar of its own: the scale is fixed and identical to the zoom squares', so the
        bar beside each square serves these rows too.
        """
        ax = self.ax_post if ax is None else ax
        Ez = dec["Ez"]
        p_s, p_e = dec["p_s"], dec["p_e"]
        times_ms = self._times_ms(rec, p_s, p_e)
        dt_ms = self.source.decode_tbin_step_s * 1000.0
        t_lo, t_hi = float(times_ms[0] - 0.5 * dt_ms), float(times_ms[-1] + 0.5 * dt_ms)
        K = int(self.bounds[-1])

        ax.imshow(
            Ez.T, origin="lower", aspect="auto",
            extent=(t_lo, t_hi, -0.5, K - 0.5),
            cmap=POSTERIOR_CMAP, vmin=0.0, vmax=self.posterior_vmax,
            interpolation="nearest",
        )
        if kernel is not None:
            ax.text(
                0.005, 0.975, kernel_label(kernel), transform=ax.transAxes,
                fontsize=_fs(8), va="top", ha="left", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85),
            )

        # Called trajectory segments, each spanning only its own map's band. Off by default:
        # they come from the memoryless-Bayes posterior, so under an all-PBE fit they drive
        # neither the fit nor the jumps.
        if self.highlight_mb_traj:
            for i, (m, s, e) in enumerate(dec["segs"]):
                y0, y1 = self.bounds[m] - 0.5, self.bounds[m + 1] - 0.5
                x0 = self._times_ms(rec, s, s + 1)[0] - 0.5 * dt_ms
                x1 = self._times_ms(rec, e - 1, e)[0] + 0.5 * dt_ms
                ax.add_patch(Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=TRAJ_SPAN_COLOR, edgecolor=TRAJ_SPAN_COLOR,
                    alpha=0.18, lw=0.8, zorder=2,
                    label="MB-called trajectory" if i == 0 else None,
                ))

        desc = self._plot_position(rec, t_lo, t_hi, ax=ax)
        if ax is self.ax_post:
            self.last_position_desc = desc

        # Faint Viterbi over the padded window, bold over the jump-generating spans.
        ax.step(times_ms, dec["z_pad"], where="mid", color=VITERBI_COLOR,
                lw=0.6, alpha=0.22, zorder=3)
        for i, (s, e, z_seg) in enumerate(dec["bold"]):
            ax.step(
                self._times_ms(rec, s, e), z_seg, where="mid",
                color=VITERBI_COLOR, lw=1.1, alpha=0.7, zorder=4,
                label="Viterbi (jump source)" if i == 0 else None,
            )

        for b in self.bounds[1:-1]:
            ax.axhline(b - 0.5, color="k", lw=0.6, alpha=0.35)
        centers = (self.bounds[:-1] + self.bounds[1:]) / 2.0 - 0.5
        labels = self.map_labels[:self.n_maps]
        ax.set_yticks(centers, labels)
        ax.set_ylim(-0.5, K - 0.5)
        ax.set_ylabel("Linearized position")
        ax.tick_params(labelbottom=False)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=_fs(7), framealpha=0.85)
        return t_lo, t_hi

    def _plot_arm_zoom(self, rec: EventRecord, dec: dict, *, ax, cax, k: int) -> None:
        """Square heatmap over the dominant map's bins alone, for kernel `k`.

        The full stack spans every map (hundreds of states on Lustig's seven-map stack), so a
        replay confined to one map is a thin sliver there. Cropping the position axis to
        `dec["dominant_map"]` -- the map the trajectory and every jump live on, chosen in
        `_decode_for` -- makes the slope of the sweep readable, which is the thing the tall
        panel hides.

        Same time axis and the same fixed colour ceiling as the main heatmap, so this is a crop
        of it, not a separately normalised picture. The y axis is cm along the map, using the
        source's `cm_per_bin`, as the jump histogram does.

        Note each kernel picks its own dominant map from its own posterior, so two rows can
        crop to different maps -- which is itself informative when they disagree. The ylabel
        names the map, so it is never ambiguous which one is shown.
        """
        m = int(dec["dominant_map"])
        b0, b1 = int(self.bounds[m]), int(self.bounds[m + 1])
        n_bins = b1 - b0
        Ez = dec["Ez"][:, b0:b1]
        times_ms = self._times_ms(rec, dec["p_s"], dec["p_e"])
        dt_ms = self.source.decode_tbin_step_s * 1000.0
        t_lo = float(times_ms[0] - 0.5 * dt_ms)
        t_hi = float(times_ms[-1] + 0.5 * dt_ms)
        y_hi = n_bins * self.cm_per_bin

        im = ax.imshow(
            Ez.T, origin="lower", aspect="auto", extent=(t_lo, t_hi, 0.0, y_hi),
            cmap=POSTERIOR_CMAP, vmin=0.0, vmax=self.posterior_vmax,
            interpolation="nearest",
        )
        if cax is not None:
            cax.clear()
            cbar = self.fig.colorbar(im, cax=cax)
            cbar.set_label("Posterior", fontsize=_fs(7.5))
            cax.tick_params(labelsize=_fs(6.5))
            self.zoom_colorbars[k] = cbar

        # Viterbi, blanked wherever it leaves this map rather than clipped to the edge -- a
        # clipped path would draw a flat line at the boundary and read as dwelling here.
        z = np.asarray(dec["z_pad"], dtype=float)
        inside = (z >= b0) & (z < b1)
        ax.step(
            times_ms, np.where(inside, (z - b0 + 0.5) * self.cm_per_bin, np.nan),
            where="mid", color=VITERBI_COLOR, lw=1.1, alpha=0.7, zorder=4,
        )

        ax.set_box_aspect(1.0)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(0.0, y_hi)
        # Few enough ticks to read in a ~1.3 in box; the wide panels keep their own.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel("Time (ms)", fontsize=_fs(8))
        # The map is named on the axis, so the panel needs no annotation over the data -- these
        # squares go into the paper. cm is always the right unit here: Lustig's wheel map's bins
        # are time since wheel onset, not distance, but it can never be the crop target because
        # `event_traj_segments` skips it and `map_mass` only spans `source.arm_maps`.
        ax.set_ylabel(f"Position on {self.map_labels[m]} (cm)", fontsize=_fs(7.5))
        ax.tick_params(labelsize=_fs(7))

    def _plot_position(
        self, rec: EventRecord, t_lo_ms: float, t_hi_ms: float, *, ax=None,
    ) -> str:
        """Dashed line at the animal's actual position, as `source.position_lines` places it.

        Returns the source's short description of where the animal was, for the info box --
        an absent line is usually a real "unassigned", not a bug.
        """
        if not self.show_position:
            return "not loaded"
        ax = self.ax_post if ax is None else ax
        lines, desc = self.source.position_lines(rec, t_lo_ms, t_hi_ms)
        labelled = False
        for t_ms, y in lines:
            if not np.isfinite(y).any():
                continue
            ax.plot(
                t_ms, y, color=POSITION_COLOR, ls="--", lw=1.2, alpha=0.5, zorder=5,
                label="rat position" if not labelled else None,
            )
            labelled = True
        return desc

    def _plot_raster(self, rec: EventRecord, dec: dict, t_lo_ms: float, t_hi_ms: float) -> None:
        ax = self.ax_raster
        m = dec["dominant_map"]
        sub = np.nan_to_num(
            self.source.lin_map_concat[:, self.bounds[m]:self.bounds[m + 1]], nan=0.0,
        )
        peak_bin = sub.argmax(axis=1)
        peak_rate = sub.max(axis=1)
        # Silent-on-this-map units sink to the bottom; the rest ascend by field peak, so a
        # replay through this map reads as a diagonal.
        order = np.lexsort((peak_bin, peak_rate <= 0))

        for rank, row in enumerate(order):
            t_ms = self.source.raster_ms(int(row), rec, t_lo_ms, t_hi_ms)
            if t_ms.size:
                ax.vlines(t_ms, rank - 1.0, rank + 1.0, color="k", lw=0.8)
        ax.set_ylim(-0.5, order.size - 0.5)
        ax.set_ylabel(f"Decode units\n(by {self.map_labels[m]} field peak)")
        ax.set_xlabel("Time from event start (ms)")

    def _plot_jumps(self, index: int) -> None:
        ax = self.ax_jump
        jumps = self._jumps(index)
        counts, _ = np.histogram(jumps, bins=self.jump_bins)

        if self.backdrop is not None and self.backdrop.size:
            bg, _ = np.histogram(self.backdrop, bins=self.jump_bins)
            scale = (counts.max() / bg.max()) if bg.max() and counts.max() else 0.0
            if scale:
                ax.stairs(
                    bg * scale, self.jump_bins_cm,
                    color="0.55", lw=0.9, label="session (scaled)",
                )
        if jumps.size:
            ax.stairs(counts, self.jump_bins_cm, fill=True, color=JUMP_COLOR, alpha=0.55)
            ax.stairs(counts, self.jump_bins_cm, color=JUMP_COLOR, lw=1.0)
        else:
            ax.text(0.5, 0.5, "No jumps", ha="center", va="center", transform=ax.transAxes)

        ax.axvline(0, color="k", lw=0.7, alpha=0.35)
        ax.set_xlim(-self.jump_lim_cm, self.jump_lim_cm)
        ax.set_xlabel("Position jump (cm)")
        ax.set_ylabel("count")
        # Counts are counts: at most two ticks, both integers, so a 2-jump event does not get
        # an axis labelled 0.0/0.5/1.0.
        top = int(max(1, counts.max()))
        ax.set_ylim(0, top)
        ax.set_yticks([0, top] if top > 1 else [0, 1])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(round(v))}"))
        # Sign convention, in the bottom corners: the axis extremes carry no counts, and the
        # top-left is where loc="best" puts the legend.
        neg_label, pos_label = self.source.jump_axis_labels
        ax.annotate(neg_label, (0.02, 0.04), xycoords="axes fraction",
                    fontsize=_fs(6), ha="left", va="bottom", color="0.35")
        ax.annotate(pos_label, (0.98, 0.04), xycoords="axes fraction",
                    fontsize=_fs(6), ha="right", va="bottom", color="0.35")
        if self.backdrop is not None and self.backdrop.size:
            ax.legend(loc="best", fontsize=_fs(6), framealpha=0.85)

    def _plot_info(self, index: int, rec: EventRecord, dec: dict) -> None:
        jumps = self._jumps(index)
        n_seg = len(dec["segs"])
        total_mass = dec["map_mass"].sum()
        m = dec["dominant_map"]
        if total_mass:
            top = int(np.argmax(dec["map_mass"]))
            mass_line = (
                f"most mass  {self.map_labels[top]}"
                f" ({dec['map_mass'][top] / total_mass:.0%})"
            )
        else:
            mass_line = "most mass  n/a"
        # Two budgets. With one kernel the cell is only as tall as the z-PFR panel and the text
        # overflows into the spacer above the histogram, so ~10 lines is the ceiling. When
        # comparing, the block sits in the raster row and has to also carry a row per kernel,
        # so the event metadata is folded onto fewer lines and the two facts the squares
        # already display in their own corners (map, fraction on it) are dropped.
        src = self.source
        grid = f"{src.decode_tbin_s * 1000:g}/{src.decode_tbin_step_s * 1000:g} ms"
        if self.multi:
            lines = [
                f"{src.name}  "
                f"event {index + 1}/{len(self.records)}  {src.event_label(rec)}",
                f"{rec.duration_ms:.0f} ms, {rec.n_bins} bins @ {grid}"
                f"   peak z {rec.peak_z:.2f}",
                f"rat at {getattr(self, 'last_position_desc', 'n/a')}"
                f"   raster {self.map_labels[m]}   {n_seg} seg",
            ]
        else:
            lines = [
                f"{src.name_spaced}   "
                f"{kernel_label(self.kernel)}",
                f"event {index + 1} / {len(self.records)}   {src.event_label(rec)}",
                f"{rec.duration_ms:.0f} ms, {rec.n_bins} bins @ {grid}",
                f"peak z = {rec.peak_z:.2f}",
                "",
                f"rat at     {getattr(self, 'last_position_desc', 'n/a')}",
                f"raster     {self.map_labels[m]} ({dec['sort_basis']})",
                # How much of the event the zoom panel actually shows.
                f"zoom       {_frac_in_map(dec, self.bounds, m):.0%} of Viterbi on it",
                mass_line,
                f"segments   {n_seg} called",
                asymmetry_text(jumps),
            ]
        if self.multi:
            lls = self._lls(index)
            d = self._dll(index)
            arrs = src.scope_arrays(rec, self.scope)
            n_ll_bins = int(sum(np.asarray(a).shape[0] for a in arrs))
            lines += ["", f"{'':<10}{'logLik':>9}{'Δ(base-k)':>10}{'in zoom':>8}"]
            if np.isfinite(d):
                for k, name in enumerate(self.kernels):
                    # base - k, so positive means the main kernel explains it better.
                    delta = "" if k == 0 else f"{lls[0] - lls[k]:>+10.2f}"
                    # Only the fraction: which map each square crops is on its own y axis.
                    dk = self.decode_caches[k][index]
                    frac = _frac_in_map(dk, self.bounds, int(dk["dominant_map"]))
                    lines.append(
                        f"{kernel_label(name)[:9]:<10}{lls[k]:>9.1f}{delta:>8}{frac:>8.0%}"
                    )
                # Ranking uses the total ΔLL against the FIRST compare kernel, which grows
                # with event length; the per-bin figure keeps that confound visible.
                lines += [
                    f"{'ΔLL/bin':<10}{d / max(n_ll_bins, 1):>+9.3f}  vs "
                    f"{kernel_label(self.kernels[1])[:9]}",
                    f"{'over':<10}{n_ll_bins:>5} bins, {len(arrs)} seq",
                ]
            else:
                lines += ["n/a: event contributes", "no sequences at this scope"]
        self.ax_info.clear()
        self.ax_info.axis("off")
        # Smaller and nudged down when comparing: the block sits under the last square, and at
        # the single-kernel size its first lines would run into that square's time axis.
        self.ax_info.text(
            0.0, 0.90 if self.multi else 1.0, "\n".join(lines),
            transform=self.ax_info.transAxes,
            va="top", ha="left", fontsize=_fs(6.2) if self.multi else _fs(7.5),
            family="monospace",
        )

    # -- render ------------------------------------------------------------------------

    def render(self, index: int) -> None:
        with matplotlib.rc_context(self._rc):
            self._render(index)

    def _render(self, index: int) -> None:
        self.current_index = int(np.clip(index, 0, len(self.records) - 1))
        rec = self.records[self.current_index]
        multi = self.multi
        decs = [self._decode_k(self.current_index, k) for k in range(self.n_kernels)]
        dec = decs[0]

        panels = [self.ax_zpfr, self.ax_raster, *self.ax_posts, *self.ax_zooms]
        if self.ax_jump is not None:
            panels.append(self.ax_jump)
        for ax in panels:
            ax.clear()

        # Every heatmap uses the same fixed ceiling, so no shared-vmax bookkeeping here.
        t_lo, t_hi = self._plot_posterior(
            rec, dec, kernel=self.kernel if multi else None,
        )
        for k in range(1, self.n_kernels):
            self._plot_posterior(
                rec, decs[k], ax=self.ax_posts[k],
                kernel=self.kernels[k], legend=False,
            )
        self._plot_zpfr(rec, t_lo, t_hi)
        self._plot_raster(rec, dec, t_lo, t_hi)
        if self.ax_jump is not None:
            self._plot_jumps(self.current_index)
        # With one kernel there is one square, for the base kernel beside the raster; with
        # several there is one per kernel beside its own heatmap row.
        for k, (ax, cax) in enumerate(zip(self.ax_zooms, self.zoom_caxes)):
            self._plot_arm_zoom(rec, decs[k], ax=ax, cax=cax, k=k)
        self._plot_info(self.current_index, rec, dec)

        # Event bounds, dashed through every time-aligned panel.
        time_axes = [self.ax_zpfr, self.ax_raster, *self.ax_posts, *self.ax_zooms]
        for ax in time_axes:
            ax.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.7)
            ax.axvline(rec.duration_ms, color="k", ls="--", lw=0.8, alpha=0.7)
        self.ax_zpfr.set_xlim(t_lo, t_hi)

        if multi:
            d = self._dll(self.current_index)
            gap = f"ΔLL={d:+.2f} nats" if np.isfinite(d) else "ΔLL=n/a (no called segments)"
            title = (
                f"{self.source.name}  {self.run_tag}  "
                f"scope={self.scope}  "
                f"{kernel_label(self.kernel)} - {kernel_label(self.kernels[1])}: {gap}"
            )
            if self.n_kernels > 2:
                title += f"   (+{self.n_kernels - 2} more kernel"
                title += "s)" if self.n_kernels > 3 else ")"
        else:
            title = (
                f"{self.source.name}  "
                f"{self.run_tag}  {kernel_label(self.kernel)}  scope={self.scope}"
            )
        self.fig.suptitle(title, fontsize=_fs(9))
        self.fig.canvas.draw_idle()

    def save_current(self) -> str:
        from replay_hmm.plots.save import _savefig_with_retry

        rec = self.records[self.current_index]
        kern = self.kernel
        if self.multi:
            kern = "_vs_".join(self.kernels)
        name = (
            f"event{self.current_index:04d}_{self.source.event_file_tag(rec)}"
            f"_{kern}_{self.scope}.png"
        )
        path = _savefig_with_retry(self.fig, self.save_dir / name, dpi=DPI)
        print(f"saved: {path}")
        return path


# --------------------------------------------------------------------------------------
# CLI pieces every dataset's viewer shares
# --------------------------------------------------------------------------------------

SORT_KEYS = ("time", "duration", "peak_z", "n_jumps", "dll")


def add_viewer_args(parser: argparse.ArgumentParser) -> None:
    """The browser's own flags, shared by every dataset's CLI."""
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument(
        "--compare-kernel", nargs="+", default=None, metavar="KERNEL",
        help="One or more further kernels to score the same events against, e.g. "
             "`--compare-kernel continuous_EM fragmented`. Each gets its own posterior row "
             "and zoom square, and the jump histogram is dropped. Enables --sort-by dll, "
             "which ranks by ΔLL = LL(--kernel) - LL(FIRST compare kernel); positive means "
             "--kernel wins. Further kernels are reported but do not affect the order.",
    )
    parser.add_argument(
        "--posterior-vmax", type=float, default=POSTERIOR_VMAX,
        help=f"Fixed colour ceiling for every posterior heatmap (default {POSTERIOR_VMAX}). "
             "Fixed rather than per-event so events and kernels are comparable.",
    )
    parser.add_argument(
        "--jump-scope", choices=JUMP_SCOPES, default="traj",
        help="traj: Viterbi per MB-called trajectory segment (the pipeline default). "
             "full-pbe: one Viterbi over the whole PBE window. "
             "traditional: the traditional-replay events (the set a __tradreplay fit is "
             "trained on), one Viterbi over each event's whole window.",
    )
    parser.add_argument(
        "--traditional-events-csv", type=Path, default=None,
        help="Traditional-decode events table for --jump-scope traditional. A __tradreplay "
             "fit finds its own when this flag is omitted (the local copy under its "
             "traditional run tag, else the recorded path); any other fit needs this flag.",
    )
    parser.add_argument(
        "--traditional-status", nargs="+", default=None, metavar="STATUS",
        help="Classifier verdicts kept at --jump-scope traditional. Default: the fit's own "
             "statuses for a __tradreplay fit, else accepted.",
    )
    parser.add_argument(
        "--session-backdrop", choices=("saved", "compute", "none"), default="saved",
        help="Where the faint session jump distribution comes from.",
    )
    parser.add_argument(
        "--highlight-mb-traj", action="store_true",
        help="Shade the memoryless-Bayes called trajectory segments on the posterior. Off by "
             "default: under an all-PBE fit they drive neither the fit nor the jumps.",
    )
    parser.add_argument(
        "--no-position", action="store_true",
        help="Skip the animal's actual position line (drawing only).",
    )
    parser.add_argument("--sort-by", choices=SORT_KEYS, default="time")
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--pad-ms", type=float, default=20.0)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-initial", action="store_true")


def validate_kernel_args(args, rows_by_model: dict, summary_path: Path) -> list[str]:
    """The --kernel / --compare-kernel / --sort-by checks; returns the compare kernels."""
    compare_kernels = list(args.compare_kernel or [])
    for name in (args.kernel, *compare_kernels):
        if name not in rows_by_model:
            raise SystemExit(
                f"kernel {name!r} is not in {summary_path}; "
                f"available: {sorted(rows_by_model)}"
            )
    if args.kernel in compare_kernels or len(set(compare_kernels)) != len(compare_kernels):
        raise SystemExit(
            "--compare-kernel must list kernels distinct from each other and from --kernel."
        )
    if args.sort_by == "dll" and not compare_kernels:
        raise SystemExit("--sort-by dll requires --compare-kernel.")
    return compare_kernels


def warn_if_scope_mismatch(fit_scope: str, jump_scope: str, run_tag: str) -> None:
    """Warn (RuntimeWarning) when the fit's event set and --jump-scope disagree."""
    if fit_scope != jump_scope:
        warnings.warn(
            f"fit {run_tag} was trained on {fit_scope!r} events but --jump-scope is "
            f"{jump_scope!r}. The transition kernel was never fitted to the event "
            "set these jumps are harvested from.",
            RuntimeWarning,
            stacklevel=2,
        )


TRADITIONAL_EVENTS_NAME = "traditional_decode_events.csv"


def resolve_traditional_table(
    summary: dict, args, *, fit_scope: str | None, local_dir: Path, rat: str, session: str,
) -> tuple[Path, tuple[str, ...]]:
    """The traditional events table and verdicts `--jump-scope traditional` shows.

    A __tradreplay fit's own table: `--traditional-events-csv`, else the local copy under
    `local_dir/<the fit's traditional run tag>/` (fits made on the cluster record a cluster
    path), else the recorded path -- `fit_summary.resolve_traditional_events_csv`, which
    refuses a CSV from a different traditional run tag. Its verdicts are the fit's; a
    different `--traditional-status` would show a different event set under the same tag, so
    it is refused. Any other fit records no table, so the flag is required, and the verdicts
    default to accepted.
    """
    from replay_hmm.io import fit_summary

    passed = None if args.traditional_status is None else tuple(map(str, args.traditional_status))
    if fit_scope != "traditional":
        if args.traditional_events_csv is None:
            raise SystemExit(
                f"[{rat}/{session}] --jump-scope traditional on a fit not trained on traditional "
                "events needs --traditional-events-csv: the fit records no table to take the "
                "event set from."
            )
        return Path(args.traditional_events_csv), passed or ("accepted",)

    override = args.traditional_events_csv
    run_tag = summary.get("traditional_run_tag")
    if override is None and run_tag:
        local = Path(local_dir) / str(run_tag) / TRADITIONAL_EVENTS_NAME
        override = local if local.exists() else None
    try:
        csv_path = fit_summary.resolve_traditional_events_csv(
            summary, override, rat=rat, session=session,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    recorded = summary.get("traditional_statuses")
    recorded = tuple(map(str, recorded)) if recorded else None
    if passed is not None and recorded is not None and passed != recorded:
        raise SystemExit(
            f"[{rat}/{session}] --traditional-status is {list(passed)} but this fit was trained "
            f"on statuses {list(recorded)}; showing a different verdict set would not be the "
            "fitted event set."
        )
    return Path(csv_path), passed or recorded or ("accepted",)


def run_viewer(
    *, source: ViewerSource, records: list[EventRecord], hmms: list, kernels, args,
    run_tag: str, save_dir: Path, load_saved_backdrop,
) -> int:
    """Score, sort and browse -- everything after a dataset has built its source and HMMs.

    `hmms` are in `kernels` order, the base kernel first. `load_saved_backdrop()` returns the
    session's saved jumps for `--session-backdrop saved`, and is called only then.
    """
    hmm = hmms[0]
    compare_kernels = list(kernels[1:])
    print(f"  {len(records)} events, {source.bounds[-1]} states, "
          f"{source.lin_map_concat.shape[0]} decode units")

    # One full pass over the session, when the sort key or the backdrop needs it.
    jumps_cache: dict[int, np.ndarray] = {}
    ll_cache: dict[int, tuple[float, ...]] = {}
    want_jumps = args.sort_by == "n_jumps" or args.session_backdrop == "compute"
    want_dll = args.sort_by == "dll"
    if want_jumps or want_dll:
        # Threads, not processes: forward-backward and Viterbi spend their time in numpy with
        # the GIL released, and a process pool costs more (spawning, pickling the HMMs) than
        # the work. Capped at 4.
        n_jobs = min(4, max(1, (os.cpu_count() or 2) - 2))
        arrs_all = [source.scope_arrays(rec, args.jump_scope) for rec in records]

        def _score(i: int):
            arrs = arrs_all[i]
            j = (jumps_for_arrays(arrs, hmm, source.bounds, source.signed_jumps)
                 if want_jumps else None)
            ll = tuple(event_loglik(arrs, h) for h in hmms) if want_dll else None
            return j, ll

        what = "scoring" if want_dll else "decoding"
        print(f"  {what} {len(records)} events on {n_jobs} threads...")
        out = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_score)(i) for i in range(len(records))
        )
        for i, (j, ll) in enumerate(out):
            if want_jumps:
                jumps_cache[i] = j
            if want_dll:
                ll_cache[i] = ll
        if want_jumps:
            total = sum(int(j.size) for j in jumps_cache.values())
            print(f"  session total: {total} jumps at scope={args.jump_scope}")
        if want_dll:
            first = compare_kernels[0]
            d = np.array([ll_cache[i][0] - ll_cache[i][1] for i in range(len(records))])
            ok = np.isfinite(d)
            n_b, n_a = int((d[ok] > 0).sum()), int((d[ok] < 0).sum())
            print(f"  ΔLL = {args.kernel} - {first}, summed over "
                  f"{int(ok.sum())} scoring events ({int((~ok).sum())} contribute none)")
            print(f"    session ΔLL {d[ok].sum():+.2f} nats;  "
                  f"{n_b} events favour {args.kernel}, {n_a} favour {first}")
            if ok.any():
                print(f"    per-event range {d[ok].min():+.2f} .. {d[ok].max():+.2f} nats")
            # Every kernel's session total, so a third kernel is not invisible just because
            # the ranking only uses the first pair.
            for k, name in enumerate(kernels):
                tot = float(np.array([ll_cache[i][k] for i in range(len(records))])[ok].sum())
                print(f"    {name:<20} session logLik {tot:>14.2f}"
                      + ("   <- ranked against base" if k == 1 else ""))

    backdrop = None
    if args.session_backdrop == "saved":
        backdrop = load_saved_backdrop()
        print(f"  backdrop: {backdrop.size} saved jumps")
    elif args.session_backdrop == "compute":
        pooled = [j for j in jumps_cache.values() if j.size]
        backdrop = np.concatenate(pooled) if pooled else np.array([], dtype=int)

    sort_key = {
        "time": lambda r: r.t0_s,
        "duration": lambda r: -r.duration_ms,
        "peak_z": lambda r: -(r.peak_z if np.isfinite(r.peak_z) else -np.inf),
    }.get(args.sort_by)
    if sort_key is not None:
        order = sorted(range(len(records)), key=lambda i: sort_key(records[i]))
    elif args.sort_by == "dll":
        # Descending, so the list runs base-kernel-best -> compare-kernel-best. Events with
        # no ΔLL sort last rather than into the middle where a 0.0 would put them: they have
        # no opinion about the kernels and should not read as a tie.
        order = sorted(
            range(len(records)),
            key=lambda i: (
                not np.isfinite(ll_cache[i][0] - ll_cache[i][1]),
                -(ll_cache[i][0] - ll_cache[i][1])
                if np.isfinite(ll_cache[i][0] - ll_cache[i][1]) else 0.0,
            ),
        )
    else:
        order = sorted(range(len(records)), key=lambda i: -int(jumps_cache[i].size))
    records = [records[i] for i in order]
    jumps_cache = {new: jumps_cache[old] for new, old in enumerate(order) if old in jumps_cache}
    ll_cache = {new: ll_cache[old] for new, old in enumerate(order) if old in ll_cache}

    viewer = EventViewer(
        source=source, hmm=hmm, kernel=args.kernel, run_tag=run_tag, records=records,
        scope=args.jump_scope, backdrop=backdrop, pad_ms=args.pad_ms,
        initial_index=args.event_index, save_dir=save_dir, jumps_cache=jumps_cache,
        show_position=not args.no_position, highlight_mb_traj=args.highlight_mb_traj,
        compare_hmms=hmms[1:], compare_kernels=compare_kernels, ll_cache=ll_cache,
        posterior_vmax=args.posterior_vmax,
    )

    if args.save_initial:
        viewer.save_current()
    if args.no_show:
        plt.close(viewer.fig)
    else:
        plt.show()
    return 0
