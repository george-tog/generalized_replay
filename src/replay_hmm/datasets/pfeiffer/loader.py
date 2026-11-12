"""Session discovery, raw ``.mat`` loading and the linear-axis choice for the Pfeiffer & Foster
linear-track sessions.

Mirrors ``replay_hmm.datasets.buzsaki.loader``. Speed and direction live in ``kinematics``,
spatial binning and the directional rate maps in ``ratemaps``, candidate events in
``events``, per-event decoding and MB jumps in ``session``, and the MB-jump command in
``mb_jump``.
The data root is ``replay_hmm.paths.PFEIFFER_DATA_DIR``.

Units are **centimetres and cm/s throughout**. ``Position_Data`` is already in cm, and the
Buzsaki constants are re-expressed in cm rather than converted at use. Every metadata JSON
written by a consumer records ``"length_units": "cm"``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from replay_hmm import paths
from replay_hmm.datasets import registry

PFEIFFER_DIR = paths.PFEIFFER_DATA_DIR

# The linear track must be clearly axis-aligned for the max-range rule below to be
# meaningful. The nine sessions' long/short range ratios are 3.9-5.2.
MIN_AXIS_RATIO = 3.0

# The session registry is replay_hmm.datasets.registry.PFEIFFER_SESSIONS.

REQUIRED_MAT_FILES = (
    "Position_Data.mat",
    "Spike_Data_With_Gamma_Phase.mat",
    "Experiment_Information.mat",
    "Ripple_Events.mat",
)


# ── Session discovery ─────────────────────────────────────────────────────────

def _natural_key(text: str) -> list:
    parts: list = []
    buf = ""
    is_digit = None
    for ch in text:
        ch_is_digit = ch.isdigit()
        if is_digit is None or ch_is_digit == is_digit:
            buf += ch
            is_digit = ch_is_digit
        else:
            parts.append(int(buf) if is_digit else buf.lower())
            buf = ch
            is_digit = ch_is_digit
    if buf:
        parts.append(int(buf) if is_digit else buf.lower())
    return parts


def session_dir(rat: str, session: str, data_dir: Path | None = None) -> Path:
    return Path(data_dir or PFEIFFER_DIR) / str(rat) / str(session)


def add_session_selection_args(parser: argparse.ArgumentParser, *, single: bool) -> None:
    """The session selection `discover_sessions` takes: --rats/--sessions and --data-dir.

    `single` first adds the fit CLI's --rat/--session (one name each, appended to the lists by
    its `selected_sessions`); the jump CLI passes False.
    """
    if single:
        parser.add_argument("--rat", default=None, help="Rat name, e.g. Rat1.")
        parser.add_argument("--session", default=None, help="Session name, e.g. Linear1.")
    parser.add_argument("--rats", nargs="+", default=None)
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--data-dir", type=Path, default=None)


def discover_sessions(
    data_dir: Path | None = None,
    rats: list[str] | None = None,
    sessions: list[str] | None = None,
) -> list[tuple[str, str, Path]]:
    """The registered linear sessions, filtered by ``rats`` / ``sessions``, with their directories.

    The registry is authoritative: a ``Linear*`` directory that appears on disk but is not
    in ``registry.PFEIFFER_SESSIONS`` is skipped with no output, so adding data cannot
    silently change what a sweep covers. A selected registered session missing any of
    ``REQUIRED_MAT_FILES`` raises FileNotFoundError.
    """
    root = Path(data_dir or PFEIFFER_DIR)
    rat_filter = {r.lower() for r in rats} if rats else None
    session_filter = {s.lower() for s in sessions} if sessions else None

    out: list[tuple[str, str, Path]] = []
    for rat, sess in registry.PFEIFFER_SESSIONS:
        if rat_filter is not None and rat.lower() not in rat_filter:
            continue
        if session_filter is not None and sess.lower() not in session_filter:
            continue
        sdir = root / rat / sess
        missing = [n for n in REQUIRED_MAT_FILES if not (sdir / n).exists()]
        if missing:
            raise FileNotFoundError(
                f"{rat}/{sess}: missing {', '.join(missing)} under {sdir}"
            )
        out.append((rat, sess, sdir))
    out.sort(key=lambda r: (_natural_key(r[0]), _natural_key(r[1])))
    return out


def eligible_sessions(
    names: list[str] | None = None, data_dir: Path | None = None
) -> list[tuple[str, str, Path]]:
    """Mirror of ``datasets.buzsaki.traditional.eligible_sessions``.

    ``names`` accepts either ``Rat1`` (all that rat's sessions) or ``Rat1/Linear1``.
    """
    all_sessions = discover_sessions(data_dir)
    if not names:
        return all_sessions

    wanted = {n.strip().lower() for n in names}
    out = []
    for rat, sess, sdir in all_sessions:
        if rat.lower() in wanted or f"{rat}/{sess}".lower() in wanted:
            out.append((rat, sess, sdir))
    unmatched = wanted - (
        {r.lower() for r, _, _ in all_sessions}
        | {f"{r}/{s}".lower() for r, s, _ in all_sessions}
    )
    if unmatched:
        raise ValueError(f"Unknown session selector(s): {sorted(unmatched)}")
    return out


def session_label(rat: str, session: str) -> str:
    """Flat, globally unique session name for CSV rows and figure titles.

    Pfeiffer's ``Linear1`` occurs under three rats, so unlike Buzsaki's
    ``Achilles_10252013`` the session name alone is not a key. Consumers write
    explicit ``rat`` and ``session`` columns; this is for display only.
    """
    return f"{rat}_{session}"


# ── Linearization ─────────────────────────────────────────────────────────────

def linear_axis(position_data: np.ndarray) -> int:
    """Column index (1 = x, 2 = y) of the track's long axis.

    Pfeiffer ships raw 2-D cm, not a pre-linearized coordinate like Buzsaki's
    ``OneDLocation``, and the long axis is **not** the same across sessions:
    Rat1/Linear4-6 run along x while the other six run along y; hardcoding column 2
    would turn those three sessions into a ~40 cm "track" of pure transverse jitter.
    The authors' own ``Field_Data`` shapes agree -- (21, 91, .) x-major for Linear4
    versus (88, 19, .) y-major for Linear1.

    A one-line max-range rule is used rather than PCA or track_linearization
    because the tracks are axis-aligned and this is auditable.
    """
    arr = np.asarray(position_data, dtype=float)
    x_range = float(np.ptp(arr[:, 1][np.isfinite(arr[:, 1])]))
    y_range = float(np.ptp(arr[:, 2][np.isfinite(arr[:, 2])]))
    lo, hi = sorted((x_range, y_range))
    if lo <= 0 or hi / lo < MIN_AXIS_RATIO:
        raise ValueError(
            f"Position is not clearly axis-aligned: x range {x_range:.1f} cm, "
            f"y range {y_range:.1f} cm (ratio {hi / max(lo, 1e-9):.2f} < "
            f"{MIN_AXIS_RATIO}). This loader is for linear tracks only."
        )
    return 1 if x_range > y_range else 2


def _as_2col_intervals(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size % 2 != 0:
        raise ValueError(f"Expected start/end pairs, got {arr.size} values.")
    return arr.reshape(-1, 2)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_session_data(sdir: Path) -> dict[str, Any]:
    """Load one Pfeiffer linear-track session.

    Returns the raw tables plus the derived linearization. Position samples with a
    non-increasing timestamp are dropped here (Rat2/Linear2 carries one duplicated
    timestamp), so every downstream consumer can assume a strictly increasing timebase.
    Near-zero but positive intervals (3 us on Rat2/Linear1) are kept; ``kinematics``
    differentiates with a per-interval median stride so they cannot spike the velocity.
    """
    sdir = Path(sdir)
    position_data = np.asarray(
        loadmat(sdir / "Position_Data.mat")["Position_Data"], dtype=float
    )

    # Enforce a strictly increasing timebase before anything differentiates it.
    t_raw = position_data[:, 0]
    keep = np.ones(t_raw.shape, dtype=bool)
    keep[1:] = np.diff(t_raw) > 0
    n_dropped = int((~keep).sum())
    position_data = position_data[keep]
    if position_data.shape[0] < 3:
        raise ValueError(f"{sdir}: fewer than 3 usable position samples.")

    spike_info = loadmat(sdir / "Spike_Data_With_Gamma_Phase.mat")
    spike_data = np.asarray(spike_info["Spike_Data"], dtype=float)
    excit_neurons = np.asarray(spike_info["Excitatory_Neurons"], dtype=int).reshape(-1)
    inhib_neurons = np.asarray(spike_info["Inhibitory_Neurons"], dtype=int).reshape(-1)

    analysis_intervals_s = _as_2col_intervals(
        loadmat(sdir / "Experiment_Information.mat")["Run_Times"]
    )
    analysis_intervals_s = analysis_intervals_s[
        np.argsort(analysis_intervals_s[:, 0])
    ]

    ripple_events = np.asarray(
        loadmat(sdir / "Ripple_Events.mat")["Ripple_Events"], dtype=float
    ).reshape(-1, 3)

    traj_path = sdir / "Trajectory_Events.mat"
    if traj_path.exists():
        trajectory_events = np.asarray(
            loadmat(traj_path)["Trajectory_Events"], dtype=float
        ).reshape(-1, 2)
    else:
        trajectory_events = np.zeros((0, 2), dtype=float)

    axis = linear_axis(position_data)
    pos_lin_cm = position_data[:, axis].astype(float)
    # Origin at the low end of the track, as datasets.buzsaki.ratemaps.make_bins assumes.
    pos_lin_cm = pos_lin_cm - float(np.nanmin(pos_lin_cm))
    track_len_cm = float(np.nanmax(pos_lin_cm))

    return {
        "position_data": position_data,
        "pos_ts": position_data[:, 0].astype(float),
        "pos_lin_cm": pos_lin_cm,
        "linear_axis": int(axis),
        "track_len_cm": track_len_cm,
        "n_dropped_position_samples": n_dropped,
        "spike_data": spike_data,
        "excit_neurons": excit_neurons,
        "inhib_neurons": inhib_neurons,
        "cell_ids": np.sort(np.unique(excit_neurons)),
        "analysis_intervals_s": analysis_intervals_s,
        "ripple_events": ripple_events,
        "trajectory_events": trajectory_events,
    }


def cell_spike_times(data: dict[str, Any]) -> dict[int, np.ndarray]:
    """Per-cell sorted spike times, built once per session.

    The counterpart of ``datasets.buzsaki.session.pyramidal_spike_times``, so per-event
    lookups need not re-scan the full ~600k-spike array.
    """
    spike_t = np.asarray(data["spike_data"][:, 0], dtype=float)
    spike_clu = np.asarray(data["spike_data"][:, 1], dtype=int)
    order = np.argsort(spike_t, kind="stable")
    spike_t = spike_t[order]
    spike_clu = spike_clu[order]
    out: dict[int, np.ndarray] = {}
    for cid in np.asarray(data["cell_ids"], dtype=int):
        out[int(cid)] = spike_t[spike_clu == int(cid)]
    return out


def mask_from_intervals(times_s: np.ndarray, intervals_s: np.ndarray | None) -> np.ndarray:
    """Boolean mask of samples falling inside any [start, end] interval."""
    times_s = np.asarray(times_s, dtype=float)
    if intervals_s is None:
        return np.ones(times_s.shape, dtype=bool)
    mask = np.zeros(times_s.shape, dtype=bool)
    for start_s, end_s in np.asarray(intervals_s, dtype=float).reshape(-1, 2):
        mask |= (times_s >= start_s) & (times_s <= end_s)
    return mask
