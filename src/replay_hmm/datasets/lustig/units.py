"""Which `Clu` units of a Lustig session are hippocampal, per rat, and which units the decoder uses.

`_select_decode_units_and_maps` applies the directional, HPC and pyramidal filters to a map
archive's rows; `build_spike_dict20k` collects the selected units' spike times.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from replay_hmm.datasets.lustig.constants import constants
from replay_hmm.io.matlab import _as_1d

# Rats recorded with an HPC-only probe: every unit is hippocampal, so there is no shank
# cut to apply (see data/lee_lab/SESSION_SUMMARIES.txt). TinyTim/Lucky/Ned are not listed
# here because they are covered by HPC_FILTER in `constants()` (TinyTim's threshold is +inf,
# i.e. keep all).
HPC_KEEP_ALL_RATS: frozenset[str] = frozenset({"bb", "lb", "spot", "ren", "huey", "stinky"})


def hpc_unit_mask(clu_shank: Any, rat_name: str) -> np.ndarray:
    """Boolean mask over `Clu` units selecting the hippocampal ones.

    The single implementation of this rule: `load_session` (the map builder) and
    `_hpc_unit_mask_from_shank` (the decoder) both delegate here, so one session never gets
    two different masks.

    An unrecognised rat keeps every unit and warns. That is the safe direction: an included
    non-HPC unit shows up as a unit with no place field, whereas silently discarding a
    seventh of the population does not show up at all.
    """
    shank = _as_1d(clu_shank, dtype=int)
    key = str(rat_name).strip().lower()
    c = constants()
    shank_max = {
        "tinytim": c.HPC_FILTER.TINYTIM_MAX_SHANK,
        "lucky": c.HPC_FILTER.LUCKY_MAX_SHANK,
        "ned": c.HPC_FILTER.NED_MAX_SHANK,
    }
    if key in shank_max:
        return shank <= float(shank_max[key])
    if key not in HPC_KEEP_ALL_RATS:
        warnings.warn(
            f'Unknown animal "{rat_name}" - keeping all units as HPC.',
            RuntimeWarning,
            stacklevel=2,
        )
    return np.ones(shank.size, dtype=bool)


def _directional_row_mask(maps, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    n_units_maps = int(np.asarray(maps["linear"]["maps"]).shape[0])
    if mode == "all":
        return np.ones(n_units_maps, dtype=bool)
    if mode not in {"directional", "nondirectional"}:
        raise ValueError(f'Unknown DIRECTIONAL_UNIT_MODE="{mode}". Use "all", "directional", or "nondirectional".')

    try:
        is_directional = np.asarray(maps["linearDirectionality"]["isDirectional"], dtype=bool).reshape(-1)
    except Exception as exc:
        raise KeyError(
            "Directionality labels are missing from Maps. Rebuild maps with computeDirectionality enabled."
        ) from exc

    if is_directional.size != n_units_maps:
        raise ValueError(
            f"Directionality mask length ({is_directional.size}) does not match map units ({n_units_maps})."
        )
    return is_directional if mode == "directional" else ~is_directional


def _hpc_unit_mask_from_shank(clu_shank_arr: np.ndarray, rat_name: str) -> np.ndarray:
    """Boolean mask over Clu units for HPC selection; an alias of `hpc_unit_mask`."""
    return hpc_unit_mask(clu_shank_arr, rat_name)


def _clu_is_intern(mat) -> np.ndarray | None:
    """`Clu.isIntern` as a bool mask, or None when the field is absent."""
    try:
        return np.asarray(mat["Clu"]["isIntern"]).ravel().astype(bool)
    except Exception:
        return None


def _select_decode_units_and_maps(
    maps,
    directional_mode: str,
    *,
    hpc_only: bool = False,
    clu_shank: np.ndarray | None = None,
    rat_name: str | None = None,
    pyramidal_only: bool = False,
    clu_is_intern: np.ndarray | None = None,
):
    unit_idx_all = np.asarray(maps["linear"]["unitIdx"], dtype=int).reshape(-1)
    row_mask = _directional_row_mask(maps, directional_mode)
    if row_mask.size != unit_idx_all.size:
        raise ValueError(
            f"Directional row mask size mismatch ({row_mask.size} vs unitIdx size {unit_idx_all.size})."
        )

    if hpc_only:
        if clu_shank is None or rat_name is None:
            raise ValueError("hpc_only=True requires clu_shank and rat_name.")
        hpc_unit_mask = _hpc_unit_mask_from_shank(np.asarray(clu_shank), str(rat_name))
        if unit_idx_all.size and (unit_idx_all.max() >= hpc_unit_mask.size or unit_idx_all.min() < 0):
            raise ValueError(
                f"unitIdx out of range for Clu arrays: min={unit_idx_all.min()}, "
                f"max={unit_idx_all.max()}, n_clu={hpc_unit_mask.size}"
            )
        row_mask = row_mask & hpc_unit_mask[unit_idx_all]

    if pyramidal_only:
        # Note the effect is large: interneurons carry most of the rate-map mass, so enabling
        # this substantially changes each state's expected total spike count (the summed
        # rate-map term of the Poisson likelihood).
        if clu_is_intern is None:
            raise ValueError("pyramidal_only=True requires Clu.isIntern, which is absent from this session.")
        is_intern = np.asarray(clu_is_intern, dtype=bool).ravel()
        if unit_idx_all.size and unit_idx_all.max() >= is_intern.size:
            raise ValueError(
                f"unitIdx out of range for Clu.isIntern: max={unit_idx_all.max()}, n={is_intern.size}"
            )
        row_mask = row_mask & ~is_intern[unit_idx_all]

    if not np.any(row_mask):
        msg = f"No units left after applying directional_unit_mode='{directional_mode}'"
        if hpc_only:
            msg += ", hpc_only"
        if pyramidal_only:
            msg += ", pyramidal_only"
        raise ValueError(msg + " filter(s).")

    linear_maps = np.asarray(maps["linear"]["maps"])
    wheel_maps = np.asarray(maps["wheel"]["maps"], dtype=float)
    if wheel_maps.shape[0] != row_mask.size:
        raise ValueError(
            f"wheel map unit count ({wheel_maps.shape[0]}) does not match linear/directionality ({row_mask.size})."
        )

    return unit_idx_all[row_mask], linear_maps[row_mask], wheel_maps[row_mask], row_mask


def build_spike_dict20k(unit_idx: np.ndarray, clu_totclu: np.ndarray, spike_totclu: np.ndarray, spike_res20k: np.ndarray):
    spk_dict20k = {}
    for row_i, clu_idx in enumerate(np.asarray(unit_idx, dtype=int).reshape(-1)):
        clu_id = int(clu_totclu[int(clu_idx)])
        spk_dict20k[row_i] = spike_res20k[spike_totclu == clu_id]
    return spk_dict20k
