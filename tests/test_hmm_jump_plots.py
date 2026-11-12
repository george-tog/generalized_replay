"""The shared jump-histogram renderers.

Assertions are on axis state, not pixels. What is pinned: the two axis policies stay
different on purpose (fixed for the pooled figure, data-derived for the grids), and the
Buzsaki two-line panel title is a parameter of the shared renderer.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

import matplotlib.pyplot as plt

from replay_hmm.plots import jumps as hjp

COLOR = "#8c4bd5"


# ── shared_jump_axis ──────────────────────────────────────────────────────────

def test_axis_is_unit_width_bins_out_to_the_pooled_max():
    edges, ticks = hjp.shared_jump_axis(np.array([-3, 1, 2, 7, -7]))
    np.testing.assert_allclose(edges, np.arange(-7.5, 8.5, 1))
    np.testing.assert_allclose(ticks, np.linspace(-7, 7, 7))
    assert edges.size == 16          # 15 unit-width bins spanning -7..7


def test_axis_ignores_hover_jumps_when_setting_the_range():
    """Zeros are excluded from every panel, so they must not set the axis either."""
    edges_with, _ = hjp.shared_jump_axis(np.array([0, 0, 0, 4]))
    edges_without, _ = hjp.shared_jump_axis(np.array([4]))
    np.testing.assert_allclose(edges_with, edges_without)


@pytest.mark.parametrize("ref", [np.array([], dtype=int), np.zeros(5, dtype=int)])
def test_axis_falls_back_to_one_for_an_empty_reference(ref):
    """A kernel that produced no jumps must render an empty axis, not raise."""
    edges, ticks = hjp.shared_jump_axis(ref)
    np.testing.assert_allclose(edges, np.arange(-1.5, 2.5, 1))
    assert ticks[0] == -1 and ticks[-1] == 1


def test_the_pooled_axis_is_fixed_not_derived():
    """The pooled figure's axis must NOT track the data -- that is what makes it
    overlayable across datasets."""
    assert hjp.AGG_BINS[0] == pytest.approx(-52.5)
    # 51.5, not 52.5: `arange(-52, 53) - 0.5` gives bin CENTRES -52..51, so the fixed axis
    # is asymmetric by one bin. That is the Lustig convention -- pinned here so nobody
    # "corrects" it and silently shifts every pooled histogram.
    assert hjp.AGG_BINS[-1] == pytest.approx(51.5)
    assert hjp.AGG_BINS.size == 105


# ── cm ticks ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("c_bin_cm", [3.465, 3.507, 3.557, 4.0])
def test_ticks_are_round_cm_whatever_the_bin_width(c_bin_cm):
    """Labels are fixed round cm and positions move with cm/bin, so MB (3.507 cm/bin) and
    HMM (3.557) figures label the same quantity identically."""
    locs, labels = hjp.cm_tick_locs(c_bin_cm, hjp.AGG_BINS)
    assert labels.tolist() == [-150, -100, -50, 0, 50, 100, 150]
    np.testing.assert_allclose(locs, labels / c_bin_cm)


def test_ticks_outside_the_drawn_range_are_dropped():
    """At 2 cm/bin the +-52-bin axis spans +-104 cm, so +-150 must not stretch it."""
    _locs, labels = hjp.cm_tick_locs(2.0, hjp.AGG_BINS)
    assert labels.tolist() == [-100, -50, 0, 50, 100]
    _locs, step_labels = hjp.cm_tick_locs(2.0, hjp.STEP_BINS, one_sided=True)
    assert step_labels.tolist() == [0, 50, 100]


def test_title_statistic_can_come_from_raw_bins_while_rescaled_values_are_drawn(tmp_path, monkeypatch):
    """Rescaled jumps like 0.99 would be truncated to 0 by the int-casting asymmetry test,
    so Lustig HMM hands the raw integer jumps in separately for the title."""
    titles = []
    real = hjp.jstats.asymmetry_title_text
    monkeypatch.setattr(
        hjp.jstats, "asymmetry_title_text",
        lambda j, **kw: titles.append(np.asarray(j).copy()) or real(j, **kw),
    )
    raw = np.array([1, 1, -1, 2])
    hjp.plot_aggregate_jump_histogram(
        jumps=raw * 0.99, stat_jumps=raw, out_path=tmp_path / "agg.png",
        c_bin_cm=3.5, n_perm=100, seed=42, hist_color=COLOR,
    )
    np.testing.assert_array_equal(titles[-1], raw)


def test_grid_keeps_rescaled_jumps_that_round_toward_zero(tmp_path, monkeypatch):
    """Panels are drawn as floats: casting to int would turn a rescaled 0.99 into a dropped 0."""
    drawn = []
    real_hist = plt.Axes.hist

    def spy(self, x, *a, **kw):
        drawn.append(np.asarray(x).copy())
        return real_hist(self, x, *a, **kw)

    monkeypatch.setattr(plt.Axes, "hist", spy)
    raw = np.array([1, -1, 3])
    hjp.plot_jump_panel_grid(
        panel_labels=["R1"], panel_jumps=[raw * 0.99], panel_stat_jumps=[raw],
        axis_jumps=raw * 0.99, out_path=tmp_path / "grid.png", suptitle="",
        c_bin_cm=3.5, n_perm=100, seed=42, hist_color=COLOR,
    )
    np.testing.assert_allclose(drawn[0], raw * 0.99)


# ── plot_jump_panel_grid ──────────────────────────────────────────────────────

def _grid(tmp_path, panel_labels, panel_jumps, axis_jumps=None, **kw):
    out = hjp.plot_jump_panel_grid(
        panel_labels=panel_labels,
        panel_jumps=panel_jumps,
        axis_jumps=np.concatenate(panel_jumps) if axis_jumps is None else axis_jumps,
        out_path=tmp_path / "grid.png",
        suptitle="TCM-SR",
        c_bin_cm=4.0,
        n_perm=100,
        seed=42,
        hist_color=COLOR,
        **kw,
    )
    return out


def test_grid_writes_the_file_and_makes_parent_dirs(tmp_path):
    out = hjp.plot_jump_panel_grid(
        panel_labels=["A", "ALL"],
        panel_jumps=[np.array([1, 2]), np.array([1, 2, -1])],
        axis_jumps=np.array([1, 2, -1]),
        out_path=tmp_path / "deep" / "nested" / "grid.png",
        suptitle="k", c_bin_cm=4.0, n_perm=100, seed=42, hist_color=COLOR,
    )
    assert out.exists() and out.stat().st_size > 0


def test_n_cols_is_clamped_to_the_panel_count(tmp_path, monkeypatch):
    seen = {}
    real = plt.subplots

    def spy(nrows=1, ncols=1, **kw):
        seen["shape"] = (nrows, ncols)
        return real(nrows=nrows, ncols=ncols, **kw)

    monkeypatch.setattr(plt, "subplots", spy)
    _grid(tmp_path, ["only"], [np.array([1, 2, 3])])
    assert seen["shape"] == (1, 1), "a single panel must not create a 1x2 grid"


def test_four_panels_are_two_by_two(tmp_path, monkeypatch):
    seen = {}
    real = plt.subplots

    def spy(nrows=1, ncols=1, **kw):
        seen["shape"] = (nrows, ncols)
        return real(nrows=nrows, ncols=ncols, **kw)

    monkeypatch.setattr(plt, "subplots", spy)
    _grid(tmp_path, ["R1", "R2", "R3", "ALL"], [np.array([1, -1])] * 4)
    assert seen["shape"] == (2, 2)


@pytest.mark.parametrize("sep,expect_newline", [(": ", False), ("\n", True)])
def test_title_sep_selects_the_one_or_two_line_panel_title(tmp_path, sep, expect_newline):
    """Buzsaki needs the two-line form; it is a parameter of the one shared renderer."""
    captured = []
    real_hist = plt.Axes.set_title

    def spy(self, label, *a, **kw):
        captured.append(label)
        return real_hist(self, label, *a, **kw)

    import matplotlib.axes
    orig = matplotlib.axes.Axes.set_title
    matplotlib.axes.Axes.set_title = spy
    try:
        _grid(tmp_path, ["Rat 1"], [np.array([1, 2, -1])], title_sep=sep)
    finally:
        matplotlib.axes.Axes.set_title = orig

    assert captured, "no panel title was set"
    assert ("\n" in captured[0]) is expect_newline
    assert "#jumps=" in captured[0] and "asym=" in captured[0] and "pval=" in captured[0]


def test_unused_axes_are_switched_off(tmp_path):
    """Three panels in a 2x2 grid must leave the fourth blank, not empty-but-framed."""
    figs_before = plt.get_fignums()
    _grid(tmp_path, ["a", "b", "c"], [np.array([1, -1])] * 3)
    assert plt.get_fignums() == figs_before, "the figure must be closed"


def test_grid_renders_with_an_all_empty_panel(tmp_path):
    """A kernel with no jumps in one session must not break the figure."""
    out = _grid(
        tmp_path,
        ["empty", "full", "ALL"],
        [np.array([], dtype=int), np.array([1, 2, -1]), np.array([1, 2, -1])],
    )
    assert out.exists()


# ── plot_aggregate_jump_histogram ─────────────────────────────────────────────

def test_aggregate_writes_the_file_and_closes_the_figure(tmp_path):
    figs_before = plt.get_fignums()
    out = hjp.plot_aggregate_jump_histogram(
        jumps=np.array([1, 2, -1, 3]),
        out_path=tmp_path / "agg.png",
        c_bin_cm=4.0, n_perm=100, seed=42, hist_color=COLOR,
    )
    assert out.exists() and out.stat().st_size > 0
    assert plt.get_fignums() == figs_before


def test_aggregate_handles_no_jumps(tmp_path):
    """`draw_jump_histogram`'s "No jumps" path -- exercised by sparse kernels in practice."""
    out = hjp.plot_aggregate_jump_histogram(
        jumps=np.array([], dtype=int),
        out_path=tmp_path / "empty.png",
        c_bin_cm=4.0, n_perm=100, seed=42, hist_color=COLOR,
    )
    assert out.exists()


def test_aggregate_drops_hover_and_nonfinite_jumps(tmp_path):
    """Both gates matter: zeros are hovers, and NaNs would poison the asymmetry title."""
    out = hjp.plot_aggregate_jump_histogram(
        jumps=np.array([0.0, np.nan, 1.0, -1.0]),
        out_path=tmp_path / "gated.png",
        c_bin_cm=4.0, n_perm=100, seed=42, hist_color=COLOR,
    )
    assert out.exists()


def test_the_two_figures_label_the_same_quantity_differently_on_purpose():
    """Both plot counts. The y-labels are pinned: changing either relabels every published
    pooled or grid figure."""
    assert hjp.AGG_YLABEL == "Frequency"
    assert hjp.GRID_YLABEL == "Count"
    assert hjp.XLABEL == "Position jump (cm)"


# ── plot_jump_histogram_row, step-size mode ─────────────────────────────────────────

def _render_row(tmp_path, monkeypatch, panel, **kw):
    """Render a one-panel row; return (axis, the values and bins handed to the drawer)."""
    drawn = {}
    real_draw = hjp.draw_jump_histogram

    def draw_spy(ax, jumps, bins, **dkw):
        drawn["vals"] = np.asarray(jumps, dtype=float).copy()
        drawn["bins"] = np.asarray(bins, dtype=float).copy()
        return real_draw(ax, jumps, bins, **dkw)

    closed = {}
    monkeypatch.setattr(hjp, "draw_jump_histogram", draw_spy)
    monkeypatch.setattr(plt, "close", lambda fig=None: closed.setdefault("fig", fig))
    hjp.plot_jump_histogram_row(
        panel_labels=["Rat 1"],
        panel_jumps=[panel],
        panel_colors=COLOR,
        out_path=tmp_path / "row.png",
        c_bin_cm=4.0,
        n_perm=100,
        seed=42,
        **kw,
    )
    ax = closed["fig"].axes[0]
    return ax, drawn


def test_step_row_keeps_the_zeros_and_draws_the_positive_half(tmp_path, monkeypatch):
    steps = np.array([0, 0, 0, 1, 2, 2, 5], dtype=float)
    ax, drawn = _render_row(tmp_path, monkeypatch, steps, step_sizes=True)

    np.testing.assert_array_equal(drawn["vals"], steps)  # no hover filter
    # The positive half of the jump axis exactly: AGG_BINS centres run -52..51, so 0..51.
    np.testing.assert_allclose(drawn["bins"], hjp.AGG_BINS[hjp.AGG_BINS >= -0.5])
    left_edge = min(p.get_xy()[:, 0].min() for p in ax.patches)
    assert left_edge == pytest.approx(-0.5)
    counts, _ = np.histogram(drawn["vals"], bins=drawn["bins"])
    assert counts[0] == 3, "the zero bin must hold every self-transition"

    assert ax.get_xlabel() == "Step size (cm)"
    assert "N=7" in ax.get_title() and "stay=43%" in ax.get_title()
    assert "asym" not in ax.get_title()


def test_step_row_ticks_are_the_jump_rows_positive_ticks(tmp_path, monkeypatch):
    ax_step, _ = _render_row(tmp_path, monkeypatch, np.array([0.0, 1.0]), step_sizes=True)
    ax_jump, _ = _render_row(tmp_path, monkeypatch, np.array([-1.0, 1.0]))

    step_ticks = list(ax_step.get_xticks())
    jump_ticks = [t for t in ax_jump.get_xticks() if t >= 0]
    assert sorted(step_ticks) == sorted(jump_ticks)
    labels = lambda ax: {
        float(t): lab.get_text() for t, lab in zip(ax.get_xticks(), ax.get_xticklabels())
    }
    for t in step_ticks:
        assert labels(ax_step)[float(t)] == labels(ax_jump)[float(t)]


def test_jump_row_still_drops_zeros(tmp_path, monkeypatch):
    """The default (jump) mode is untouched: hovers are filtered before drawing."""
    _, drawn = _render_row(tmp_path, monkeypatch, np.array([0.0, 0.0, 2.0, -3.0]))
    np.testing.assert_array_equal(drawn["vals"], [2.0, -3.0])
