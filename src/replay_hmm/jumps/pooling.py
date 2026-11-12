"""Reference-bin pooling of per-session jumps.

Each session decodes on its own bin width, so jumps are pooled across sessions in a physical
unit and then re-expressed in units of one reference bin (the median session width). The two
operation orders can differ in the last floating-point bit, so `to_reference_bins` offers both:
`values * width / ref_width` (MB jump and the traditional path, left to right) or
`values * (width / ref_width)` (`scale_first=True`, the HMM jump distribution). Units are the
caller's: a bin-to-cm conversion is a rescale onto a 1-cm "bin" of `MM_PER_CM` mm.
"""
import numpy as np


def to_reference_bins(values, bin_width, ref_bin_width, *, scale_first: bool = False):
    """`values`, counted in bins of `bin_width`, re-expressed in bins of `ref_bin_width`.

    ``values * bin_width / ref_bin_width``, evaluated left to right, or
    ``values * (bin_width / ref_bin_width)`` with `scale_first=True`.
    """
    if scale_first:
        return values * (bin_width / ref_bin_width)
    return values * bin_width / ref_bin_width


def reference_bin_width(widths, *, skip_nan: bool, empty: float | None):
    """The pooled figures' reference bin: the median of the per-session bin `widths`, as a float.

    `skip_nan=True` takes `np.nanmedian` (MB jump, traditional), otherwise `np.median` (the
    HMM jump distribution). With `empty` given, no widths gives `empty`; with None, the median
    of nothing (nan, with numpy's warning).
    """
    if empty is not None and len(widths) == 0:
        return empty
    return float(np.nanmedian(widths) if skip_nan else np.median(widths))
