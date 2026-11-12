"""Buzsaki hc-11 session loading and the ripple-event tables.

`load_session` reads both .mat formats; `load_ripple_times` / `has_ripple_table` /
`ripple_sessions` find each session's SWR table. Shared by the Buzsaki pipeline
(`replay_hmm.datasets.buzsaki.{mb_jump,traditional,fit,jumps}`). The data root is
`replay_hmm.paths.BUZSAKI_DATA_DIR`.

Handles:
  - MATLAB v7.3 (HDF5): Achilles_10252013, Achilles_11012013, Buddy, Cicero
  - MATLAB v5: Gatsby (arrays differently transposed)
  - Circular mazes: Achilles_11012013, Cicero_09102014, Gatsby_08282013 (wrap-aware
    velocity; CW/CCW maps). None is in replay_hmm.datasets.registry.BUZSAKI_SESSIONS;
    all three are listed in its BUZSAKI_EXCLUDED_CIRCULAR_SESSIONS.
  - Linear mazes: 1.6 m (Achilles_10252013, Buddy, Gatsby_08022013, Cicero_09012014) and
    2 m (Cicero_09172014)
"""

from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from replay_hmm import paths
from replay_hmm.datasets import registry

BUZSAKI_DIR = paths.BUZSAKI_DATA_DIR

# The session registry is replay_hmm.datasets.registry.BUZSAKI_SESSIONS.

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_hdf5(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        pos_ts      = f["sessInfo/Position/TimeStamps"][:].flatten()
        pos_2d      = f["sessInfo/Position/TwoDLocation"][:]            # (2, N)
        pos_1d      = f["sessInfo/Position/OneDLocation"][:].flatten()
        maze_epoch  = f["sessInfo/Epochs/MazeEpoch"][:].flatten()
        pre_epoch   = f["sessInfo/Epochs/PREEpoch"][:].flatten()
        post_epoch  = f["sessInfo/Epochs/POSTEpoch"][:].flatten()
        spike_times = f["sessInfo/Spikes/SpikeTimes"][:].flatten()
        spike_ids   = f["sessInfo/Spikes/SpikeIDs"][:].flatten().astype(int)
        pyr_ids     = f["sessInfo/Spikes/PyrIDs"][:].flatten().astype(int)
        int_ids     = f["sessInfo/Spikes/IntIDs"][:].flatten().astype(int)
        maze_type   = "".join(
            chr(int(c)) for c in f["sessInfo/Position/MazeType"][:].flatten()
        )
    return dict(pos_ts=pos_ts, pos_2d=pos_2d, pos_1d=pos_1d,
                maze_epoch=maze_epoch, pre_epoch=pre_epoch, post_epoch=post_epoch,
                spike_times=spike_times,
                spike_ids=spike_ids, pyr_ids=pyr_ids, int_ids=int_ids,
                maze_type=maze_type)


def _load_v5(path: Path) -> dict:
    """Load from MATLAB v5 file (Gatsby: TwoDLocation stored as (N, 2) not (2, N))."""
    m    = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    sess = m["sessInfo"]
    raw  = np.asarray(sess.Position.TwoDLocation)
    pos_2d = raw.T if (raw.ndim == 2 and raw.shape[1] == 2) else raw
    return dict(
        pos_ts      = np.asarray(sess.Position.TimeStamps).flatten(),
        pos_2d      = pos_2d,
        pos_1d      = np.asarray(sess.Position.OneDLocation).flatten(),
        maze_epoch  = np.asarray(sess.Epochs.MazeEpoch).flatten(),
        pre_epoch   = np.asarray(sess.Epochs.PREEpoch).flatten(),
        post_epoch  = np.asarray(sess.Epochs.POSTEpoch).flatten(),
        spike_times = np.asarray(sess.Spikes.SpikeTimes).flatten(),
        spike_ids   = np.asarray(sess.Spikes.SpikeIDs).flatten().astype(int),
        pyr_ids     = np.asarray(sess.Spikes.PyrIDs).flatten().astype(int),
        int_ids     = np.asarray(sess.Spikes.IntIDs).flatten().astype(int),
        maze_type   = str(sess.Position.MazeType),
    )


def load_session(path: Path) -> dict:
    try:
        with h5py.File(path, "r"):
            pass
        return _load_hdf5(path)
    except OSError:
        return _load_v5(path)


# ── Ripple (SWR) event tables ─────────────────────────────────────────────────

# Preference order. `ripplesALL` is the all-sleep-states detection and is what we want;
# the WAKE/NREM files are the same detection split by sleep state, and that split is not
# consistent across sessions (Achilles' WAKE file does not reach the MAZE epoch at all,
# while Cicero_09012014's WAKE file contains every in-maze ripple). Taking ALL avoids
# depending on the split. The CSV fallback exists only for Gatsby.
_RIPPLE_MAT_PATTERNS = ("*.ripplesALL.event.mat",)
_RIPPLE_CSV_PATTERN = "ripplesWAKE_times.csv"


def _session_root(session_name: str) -> Path | None:
    """Top-level data directory for a session, e.g. <BUZSAKI_DIR>/Achilles_10252013."""
    for name, rel in registry.BUZSAKI_SESSIONS:
        if name == session_name:
            return BUZSAKI_DIR / Path(rel).parts[0]
    return None


def load_ripple_times(session_name: str) -> np.ndarray | None:
    """LFP ripple (SWR) start/end times in seconds, shape (N, 2), or None if absent.

    Only four of the five linear sessions ship a ripple table; Buddy_06272013 has none,
    which is why every ripple-sourced analysis drops it (see `ripple_sessions`).

    Two quirks of the files on disk:

      * Achilles carries duplicate copies of its three ripple .mat files at two directory
        levels. They are the same detection, so the first match wins and the sort makes
        that deterministic.
      * Gatsby's only ripple file is named ``ripplesWAKE_times.csv``, but the name is a
        misnomer: 6039 of its 9790 rows fall inside NREM intervals, so it is an all-states
        detection comparable to the other sessions' ``ripplesALL``. Treating it as
        wake-only would wrongly exclude Gatsby from every POST-sleep analysis.

    The buzcode .mat files are v7.3 (HDF5) and their single top-level group is named after
    the detection (``ripples`` for ALL, ``ripplesWAKE`` for WAKE), so the group is found by
    exclusion rather than by name.
    """
    root = _session_root(session_name)
    if root is None or not root.exists():
        return None

    for pattern in _RIPPLE_MAT_PATTERNS:
        matches = sorted(root.rglob(pattern))
        if not matches:
            continue
        with h5py.File(matches[0], "r") as f:
            group = next(k for k in f.keys() if k != "#refs#")
            times = np.asarray(f[f"{group}/times"], dtype=float)
        # Stored (2, N) in MATLAB column-major order; some exports come back (N, 2).
        if times.ndim == 2 and times.shape[0] == 2 and times.shape[1] != 2:
            times = times.T
        return np.ascontiguousarray(times.reshape(-1, 2))

    matches = sorted(root.rglob(_RIPPLE_CSV_PATTERN))
    if matches:
        times = np.loadtxt(matches[0], delimiter=",", dtype=float)
        return np.ascontiguousarray(times.reshape(-1, 2))

    return None


def has_ripple_table(session_name: str) -> bool:
    """Whether `session_name` ships a ripple table, without reading it."""
    root = _session_root(session_name)
    if root is None or not root.exists():
        return False
    if any(sorted(root.rglob(p)) for p in _RIPPLE_MAT_PATTERNS):
        return True
    return bool(sorted(root.rglob(_RIPPLE_CSV_PATTERN)))


def ripple_sessions() -> list[str]:
    """Registered session names that ship a ripple table, in registry order.

    A function rather than a module constant so that adding a downloaded ripple file is
    picked up without re-importing, and so importing this module never touches the disk.
    """
    return [name for name, _ in registry.BUZSAKI_SESSIONS if has_ripple_table(name)]
