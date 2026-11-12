"""Millimetres per tracking pixel, measured per Lustig session from the centre-arm-tip-to-wheel span.

Also the pixel bin width a cm target resolves to (`resolve_bin_width_px`) and cm per linear bin
of a built bundle (`bin_cm_from_bundle`), the single source for any cm axis.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from replay_hmm.datasets.lustig.constants import CENTER_ARM_LENGTH_MM, MM_PER_CM
from replay_hmm.datasets.lustig.sessions import load_session_mat
from replay_hmm.decode.config import DecodeConfig
from replay_hmm.io.matlab import _as_1d

# The tip-to-wheel span (see `constants.CENTER_ARM_LENGTH_MM`). Maps are binned in pixels, so
# the scale only affects cm axes, which go through `bin_cm_from_bundle`.
CENTER_ARM_TO_WHEEL_MM = CENTER_ARM_LENGTH_MM


def mm_per_pixel_from_tracking(
    x_pix, y_pix, centre_polyline, pct: float = 5.0
) -> float:
    """`mm_per_pixel` for callers that already hold the tracking arrays.

    The measurement itself; `mm_per_pixel` is the by-name wrapper that loads a session and
    calls this. Split so that code inside the map pipeline -- which has the tracking in hand
    already -- can get the right scale without re-reading the `.mat`, and so the landmark has
    exactly one definition rather than one per call site.
    """
    centre = np.asarray(centre_polyline, dtype=float)
    if centre.ndim != 2 or centre.shape[1] != 2 or centre.shape[0] < 2:
        raise ValueError("centre-arm polyline must be an (N, 2) array with N >= 2.")
    axis = centre[-1] - centre[0]
    axis = axis / np.linalg.norm(axis)

    x = _as_1d(x_pix, dtype=float)
    y = _as_1d(y_pix, dtype=float)
    # x == 0 marks a dropped frame rather than the left edge of the arena.
    ok = np.isfinite(x) & np.isfinite(y) & (x != 0)
    if not np.any(ok):
        raise ValueError("no finite tracking samples to measure a span from.")
    proj = (np.c_[x[ok], y[ok]] - centre[0]) @ axis
    span_px = float(np.percentile(proj, 100.0 - pct) - np.percentile(proj, pct))
    if span_px <= 0:
        raise ValueError(f"degenerate tip-to-wheel span {span_px} px.")
    return float(CENTER_ARM_TO_WHEEL_MM / span_px)


@lru_cache(maxsize=None)
def mm_per_pixel(
    rat: str, session: str, data_root: str | Path | None = None, pct: float = 5.0
) -> float:
    """Millimetres per tracking pixel for one session, from a known physical landmark.

    `Track.xMM` cannot answer this: it applies one hard-coded 3.769 mm/px to every file, but
    the camera was not in the same place for all sessions. The pixel scale differs by ~1.26x
    between TinyTim/Lucky/Ned/BB and Spot/Huey/Stinky.

    The span is measured by projecting every tracked sample onto the centre arm's own axis
    and taking a `pct` to `100 - pct` percentile range, which reaches from the far end of the
    centre arm back through the hub to the wheel -- the extent `CENTER_ARM_TO_WHEEL_MM`
    describes. The trim has to be generous (5%, not 1%): at 1% the span is outlier-driven in
    some sessions.
    """
    track = load_session_mat(rat, session, data_root=data_root)["Track"]
    x = _as_1d(track["xPix"], dtype=float)
    y = _as_1d(track["yPix"], dtype=float)
    centre = np.asarray(track["linLinesCenterArm"], dtype=float)
    if centre.ndim != 2 or centre.shape[1] != 2 or centre.shape[0] < 2:
        raise ValueError(f"{rat}/{session}: no usable centre-arm polyline to define an axis.")

    try:
        return mm_per_pixel_from_tracking(x, y, centre, pct=pct)
    except ValueError as exc:
        raise ValueError(f"{rat}/{session}: {exc}") from exc


# A resolved width this far from an integer means the target does not land cleanly on this
# session's pixel grid, and silently rounding would give it a bin of a different physical size
# from every other session -- the exact failure the cm target exists to prevent.
BIN_WIDTH_ROUNDING_TOL = 0.05


def resolve_bin_width_px(cfg: "DecodeConfig", rat: str, session: str, data_root=None) -> int:
    """The pixel bin width to use for one session, honouring a `bin_width_cm` target.

    With `cfg.bin_width_cm` unset this is just `cfg.bin_width_px`. With it set, the width is
    derived from the session's own calibration, so a 3.5 cm target resolves to 8 px across
    TinyTim/Lucky/Ned/BB and 10 px across Spot/Huey/Stinky.

    Raises rather than rounds when the target misses the pixel grid by more than
    `BIN_WIDTH_ROUNDING_TOL`. A session whose camera sat somewhere new would otherwise be
    handed a quietly mismatched bin, which is invisible downstream because every cm axis
    would still be individually correct.
    """
    if cfg.bin_width_cm is None:
        return int(cfg.bin_width_px)
    mm_px = mm_per_pixel(rat, session, data_root=data_root)
    exact = float(cfg.bin_width_cm) * MM_PER_CM / mm_px
    rounded = int(round(exact))
    if rounded < 1:
        raise ValueError(
            f"{rat}/{session}: bin_width_cm={cfg.bin_width_cm} is below one pixel "
            f"({exact:.3f} px at {mm_px:.3f} mm/px)."
        )
    if abs(exact - rounded) / rounded > BIN_WIDTH_ROUNDING_TOL:
        raise ValueError(
            f"{rat}/{session}: bin_width_cm={cfg.bin_width_cm} resolves to {exact:.3f} px "
            f"at {mm_px:.3f} mm/px, which rounds to {rounded} px "
            f"({abs(exact - rounded) / rounded:.1%} off, tolerance "
            f"{BIN_WIDTH_ROUNDING_TOL:.0%}). That bin would not match the other sessions' "
            "physical size; pick a target that lands on this session's pixel grid."
        )
    return rounded


def bin_cm_from_bundle(session_obj, data_root=None) -> float:
    """cm per linear bin for a SessionBundle -- the single source for any cm axis.

    Measured, not derived from a bin count. The bins are laid out in PIXELS, so their size in
    centimetres depends on the camera, and the camera moved between recording groups:
    `mm_per_pixel` returns ~4.4 mm/px for TinyTim/Lucky/Ned/BB and ~3.5 mm/px for
    Spot/Huey/Stinky, which is a 1.26x difference in what one bin spans. A dataset-wide
    constant cannot express that, and `Track.xMM` cannot either --
    it applies one hard-coded 3.769 mm/px to every file.

    `data_root=None` resolves through `constants().DATA_ROOT`, which is right for a default
    run; pass the real root wherever the caller has one.
    """
    # The RESOLVED width, not cfg.bin_width_px: under a cm target those differ, and reading
    # the config value would report 8 px for a session actually binned at 10.
    # 0 is the dataclass default, i.e. "not resolved" (e.g. a bundle rebuilt from the event
    # viewer's cache); resolve it from the config rather than return bin_cm 0.0.
    width_px = getattr(session_obj, "bin_width_px_resolved", None) or None
    if width_px is None:
        width_px = resolve_bin_width_px(
            session_obj.cfg, session_obj.rat, session_obj.session, data_root
        )
    mm_px = mm_per_pixel(session_obj.rat, session_obj.session, data_root=data_root)
    return float(width_px) * mm_px / MM_PER_CM
