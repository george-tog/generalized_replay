"""Reading MATLAB .mat files into plain Python dicts, lists and arrays, v7 and v7.3 alike.

`_loadmat_native` is the one reader: v7 files go through `scipy.io.loadmat(simplify_cells=True)`,
v7.3 (HDF5) files through h5py, and both come back in the same shapes. `read_mat` is its
public wrapper for a path already resolved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def _as_1d(arr: Any, dtype: Any | None = None) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 0:
        x = x.reshape(1)
    else:
        x = x.reshape(-1)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
    return x


def _struct_to_native(obj: Any) -> Any:
    """Recursively convert scipy loadmat legacy structs/cells into Python-native."""
    if isinstance(obj, dict):
        return {k: _struct_to_native(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if obj.ndim == 0:
                return _struct_to_native(obj.item())
            if obj.ndim == 1:
                return [_struct_to_native(v) for v in obj]
            return [[_struct_to_native(v) for v in row] for row in obj]
        return obj
    if hasattr(obj, "_fieldnames"):
        return {name: _struct_to_native(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: _struct_to_native(obj[name]) for name in obj.dtype.names}
    return obj


# --------------------------------------------------------------------------- #
# MATLAB v7.3 (HDF5) reading
# --------------------------------------------------------------------------- #
# Some sessions (Ren_20160719_02, both Hueys and Stinky_20150722_03) are saved as MATLAB
# v7.3, which is HDF5 and which `scipy.io.loadmat` refuses outright with
# NotImplementedError. The helpers below read those files through h5py
# and return the SAME Python shapes `loadmat(..., simplify_cells=True)` returns, so nothing
# downstream of `_loadmat_native` can tell the two formats apart.
#
# The subtlety that matters: MATLAB writes column-major while HDF5 reports its dimensions in
# row-major order, so every array arrives with its axes REVERSED relative to what loadmat
# hands back. `Track.linLinesLeftArm` is a 500x2 polyline in MATLAB and arrives here as
# (2, 500). Getting this wrong silently transposes the arm geometry rather than failing, so
# every array is transposed on read and then squeezed to match the `squeeze_me=True` that
# `simplify_cells=True` implies.


def _h5_matlab_class(obj: Any) -> str:
    """The node's `MATLAB_class` attribute as a str, or `""` when absent.

    Absence is load-bearing, not a fallback: a field of a struct ARRAY is an
    object-reference dataset carrying no MATLAB_class, whereas a cell-valued field of a
    scalar struct carries `"cell"`. Both are object-reference datasets, and this attribute
    is the only thing that distinguishes them -- see `_h5_struct_to_native`.
    """
    raw = obj.attrs.get("MATLAB_class", b"")
    if isinstance(raw, (bytes, np.bytes_)):
        return raw.decode("utf-8", "replace")
    return str(raw)


def _h5_field_names(grp: Any) -> list[str]:
    """Struct field names in MATLAB's own field order.

    `MATLAB_fields` stores each name as an array of single bytes. It is preferred over
    `grp.keys()`, which HDF5 returns in alphabetical order: callers like
    `_extract_linear_slot_maps` read slots positionally, so field order is not cosmetic.
    Names listed in the attribute but missing from the group are dropped, since there is
    nothing to read for them.
    """
    raw = grp.attrs.get("MATLAB_fields", None)
    if raw is None:
        return list(grp.keys())
    names: list[str] = []
    for entry in raw:
        chars = [
            c.decode("utf-8", "replace") if isinstance(c, (bytes, np.bytes_)) else str(c)
            for c in np.asarray(entry).ravel()
        ]
        names.append("".join(chars))
    return [n for n in names if n in grp]


def _h5_deref(ref: Any, f: Any) -> Any:
    """Follow one object reference. A null reference is MATLAB's empty cell element."""
    if not ref:
        return np.array([], dtype=float)
    return _h5_to_native(f[ref], f)


def _h5_cell_to_native(refs: np.ndarray, f: Any) -> Any:
    """Cell array -> list, nested for 2-D, mirroring `_struct_to_native`'s object handling.

    `refs` is already transposed and squeezed, so a MATLAB 1xN or Nx1 cell (every cell in
    these files, e.g. `Laps.WhlLfpIndStartCW`) is 1-D here and yields a flat list of N
    arrays -- which is what `_as_index_list` consumes.
    """
    if refs.ndim == 0:
        return _h5_deref(refs[()], f)
    if refs.ndim == 1:
        return [_h5_deref(r, f) for r in refs]
    return [[_h5_deref(r, f) for r in row] for row in refs]


def _h5_struct_to_native(grp: Any, f: Any) -> Any:
    """Scalar struct -> dict; struct array -> list of dicts (as `simplify_cells` gives)."""
    import h5py

    names = _h5_field_names(grp)

    # A struct ARRAY writes each field as one object-reference dataset holding a reference
    # per element, with no MATLAB_class (a cell-valued field would say "cell"). `xml.AnatGrps`
    # is a 29-element example. Scalar structs -- Clu, Spike, Track, Laps, i.e. everything
    # `load_session` actually reads -- store their fields directly and take the fast path.
    def _is_struct_array_field(d: Any) -> bool:
        return (
            isinstance(d, h5py.Dataset)
            and d.dtype.kind == "O"
            and _h5_matlab_class(d) == ""
        )

    n_elem = 0
    for n in names:
        if _is_struct_array_field(grp[n]):
            n_elem = max(n_elem, int(grp[n].size))

    if n_elem > 1:
        out: list[dict[str, Any]] = []
        flat_refs = {
            n: np.asarray(grp[n]).T.reshape(-1)
            for n in names
            if _is_struct_array_field(grp[n])
        }
        for i in range(n_elem):
            item: dict[str, Any] = {}
            for n in names:
                if n in flat_refs:
                    refs = flat_refs[n]
                    item[n] = (
                        _h5_deref(refs[i], f) if i < refs.size else np.array([], dtype=float)
                    )
                else:
                    # A field shared by every element rather than split across them.
                    item[n] = _h5_to_native(grp[n], f)
            out.append(item)
        return out

    return {n: _h5_to_native(grp[n], f) for n in names}


def _h5_to_native(obj: Any, f: Any) -> Any:
    """Convert one HDF5 node to the value `loadmat(..., simplify_cells=True)` would return."""
    import h5py

    if isinstance(obj, h5py.Group):
        return _h5_struct_to_native(obj, f)

    # MATLAB writes `[]` as a dataset holding its own dimensions, flagged by this attribute;
    # reading the payload would give the dims back as if they were data.
    if int(obj.attrs.get("MATLAB_empty", 0)):
        return np.array([], dtype=float)

    cls = _h5_matlab_class(obj)

    if cls == "char":
        # uint16 code points, one column per string after the column-major write.
        data = np.asarray(obj).T
        rows = data.reshape(1, -1) if data.ndim < 2 else data
        strings = ["".join(chr(int(c)) for c in row if int(c) != 0) for row in rows]
        return strings[0] if len(strings) == 1 else strings

    if cls == "cell":
        return _h5_cell_to_native(np.asarray(obj).T.squeeze(), f)

    arr = np.asarray(obj)
    if cls == "logical":
        arr = arr.astype(bool, copy=False)
    arr = arr.T.squeeze()
    # loadmat with squeeze_me hands back a scalar, not a 0-d array, for a 1x1.
    return arr[()] if arr.ndim == 0 else arr


def _loadmat_native(path: Path) -> dict[str, Any]:
    """Load a .mat into plain dicts/arrays, v7 and v7.3 alike.

    h5py is imported lazily and only for v7.3 files, so an environment without it keeps
    working on the v7 sessions (TinyTim/Lucky/Ned, plus BB and Spot).
    """
    try:
        import h5py
    except ImportError:
        h5py = None

    if h5py is not None and h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            # `#refs#`/`#subsystem#` hold the targets of object references. They are reached
            # through the references that point into them, never as top-level variables.
            return {k: _h5_to_native(f[k], f) for k in f.keys() if not k.startswith("#")}

    try:
        m = loadmat(path, simplify_cells=True)
        return {k: v for k, v in m.items() if not k.startswith("__")}
    except TypeError:
        m = loadmat(path, struct_as_record=False, squeeze_me=True)
        return {k: _struct_to_native(v) for k, v in m.items() if not k.startswith("__")}
    except NotImplementedError as exc:  # v7.3 file, but h5py is not installed here
        raise RuntimeError(
            f"{path.name} is a MATLAB v7.3 (HDF5) file and needs h5py to read. "
            "Install it in this environment and rerun."
        ) from exc


def read_mat(path: str | Path) -> dict[str, Any]:
    """Read an already-resolved `.mat` into plain dicts, v7 or v7.3.

    For callers that already hold a path (a resolved `raw_path`, a discovered file) and so
    do not need `(rat, session)` resolution. Prefer `load_session_mat` when you have the
    identifiers, since that resolves the layout too.
    """
    return _loadmat_native(Path(path))
