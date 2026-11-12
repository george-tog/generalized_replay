"""Lustig (Lee lab) M-maze constants.

`constants()` is the canonical namespace (the Python version of `na_constants`): arm and slot
codes, sampling rates, map, smoothing, PBE and video parameters, and the raw-data root. A map
archive pickles it into `S["info"]["params"]`, so its fields are part of the saved format.
`LINEAR_SLOT_REINDEX` is the within-arm swap the map builder applies to the linear slots (see
`replay_hmm.datasets.lustig.labels` for what the stored order means). The module-level
unit conversions and maze-section codes are the decoder's (PBE eligibility, cm axes).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from replay_hmm import paths

# Applied by `maps.build.build_map_linear`; `labels.EXPECTED_SLOT_REINDEX` must equal it.
LINEAR_SLOT_REINDEX = np.array([1, 0, 3, 2, 5, 4], dtype=int)

# The landmark `calibration.mm_per_pixel` measures the camera scale from: the centre arm's far
# end back to the WHEEL, hub included (213 cm), not the centre arm alone. The centre-arm
# polyline is only ~74% of this span, so dividing the span by the centre-arm arc or bin count
# overstates the scale by ~1.35x; cm per bin is measured per session instead
# (`calibration.bin_cm_from_bundle`).
CENTER_ARM_LENGTH_MM = 2130.0

# Unit conversions used by the decoder.
MM_PER_M = 1000
MM_PER_CM = 10.0

# Maze-section codes. WHEEL_SECTIONS is excluded from PBE-eligible time; ARM_END_SECTIONS
# are the three reward-port zones, used only by pbe_filter_mode="arm_end_reward_speed".
WHEEL_SECTIONS = np.array([2, 3], dtype=int)
ARM_END_SECTIONS = np.array([7, 8, 9], dtype=int)


def constants() -> SimpleNamespace:
    """Return canonical constants (Python version of `na_constants`)."""
    return SimpleNamespace(
        ARM=SimpleNamespace(
            LEFT=1,
            CENTER=2,
            RIGHT=3,
            NAMES=["Left", "Center", "Right"],
            ABBREV=["L", "C", "R"],
        ),
        SLOTS=SimpleNamespace(
            L_OUT=1,
            L_IN=2,
            C_OUT=3,
            C_IN=4,
            R_OUT=5,
            R_IN=6,
            NAMES=["L-out", "L-in", "C-out", "C-in", "R-out", "R-in"],
        ),
        SRATE=SimpleNamespace(
            SPIKE=20000.0,
            LFP=1250.0,
        ),
        MAP=SimpleNamespace(
            MIN_SPEED=50.0,
            DIV_XY=6.0,
            N_LINEAR_BINS=52,
            N_WHEEL_BINS=64,
            WHEEL_BIN_MS=125.0,
            WHEEL_WINDOW_SEC=8.0,
        ),
        HPC_FILTER=SimpleNamespace(
            TINYTIM_MAX_SHANK=np.inf,
            LUCKY_MAX_SHANK=31,
            NED_MAX_SHANK=31,
        ),
        LINEAR=SimpleNamespace(
            MAX_DIST=40.0,
            NORMALIZE=False,
        ),
        SMOOTH=SimpleNamespace(
            SIGMA_2D=1.0,
            SIGMA_1D=1.0,
            MIN_OCC=0.1,
        ),
        MAZE=SimpleNamespace(
            WHEEL_SECTIONS=[2, 3],
            HUB_SECTIONS=[0, 1],
        ),
        PBE=SimpleNamespace(
            BIN_SIZE_MS=1.0,
            GAUSSIAN_SD_MS=20.0,
            STD_MULTIPLIER=2.0,
            LOWER_DIVISOR=2.0,
            MIN_DURATION_MS=50.0,
            MIN_PEAK_DIST_MS=5.0,
            SPEED_THRESH_BASELINE=50.0,
            SPEED_THRESH_DETECT=275.0,
            ARM_END_SECTIONS=[7, 8, 9],
            MIN_ACTIVE_NEURONS=4,
        ),
        VIDEO=SimpleNamespace(
            DEFAULT_FPS=10.0,
            OFFSET_RANGE=[-30.0, 60.0],
            SUBSAMPLE=5,
            FPS_SEARCH_RANGE=[9.95, 10.05],
            FPS_SEARCH_STEP=0.001,
        ),
        DATA_ROOT=paths.LUSTIG_DATA_DIR,
    )
