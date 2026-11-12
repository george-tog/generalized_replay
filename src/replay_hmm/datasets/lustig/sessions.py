"""Where a Lustig session's raw `.mat` lives, in either on-disk layout, and which sessions exist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from replay_hmm.datasets.lustig.constants import constants
from replay_hmm.io.matlab import _loadmat_native

def session_mat_path(rat: str, session: str, data_root: str | Path | None = None) -> Path:
    """The raw session `.mat` for `(rat, session)`, in whichever layout holds it.

    Both layouts under `data/lee_lab` resolve through here: `<rat>/<rat>_<session>.mat`
    (TinyTim/Lucky/Ned) and the per-session directory
    `<rat>/<session>/rec01_BehavElectrDataLFP.mat` (the other rats). A consumer that builds
    either path by hand goes wrong on the other one, which is why callers should use this
    rather than an f-string.
    """
    root = Path(data_root) if data_root is not None else constants().DATA_ROOT
    main_file, _session_path = _resolve_animal_session_main_file(str(rat), str(session), root)
    return main_file


# The main file inside a per-session directory. Constant rather than derived from the
# session id: Huey/rec02 also holds a file named rec01_..., because the shipped folders
# were renamed into <rat>/<recNN>/ without touching the files inside them.
SESSION_DIR_MAIN_FILE = "rec01_BehavElectrDataLFP.mat"


def discover_rat_sessions(rat: str, data_root: str | Path | None = None) -> list[str]:
    """Sorted session ids available on disk for one rat, across both layouts.

    A `<rat>_` prefix-strip of file stems alone finds the `<rat>/<rat>_<session>.mat` files but
    none of the per-session directories, so use this instead.

    A subdirectory counts as a session only when it actually holds `SESSION_DIR_MAIN_FILE`,
    which keeps a non-session sibling such as `TinyTim/map_creation_code/` out of the list.
    """
    root = Path(data_root) if data_root is not None else constants().DATA_ROOT
    rat_dir = Path(root) / str(rat)
    if not rat_dir.is_dir():
        return []
    found: set[str] = set()
    prefix = f"{rat}_"
    for path in rat_dir.glob(f"{prefix}*.mat"):
        session = path.stem[len(prefix):]
        if path.stem.startswith(prefix) and session:
            found.add(session)
    for child in rat_dir.iterdir():
        if child.is_dir() and (child / SESSION_DIR_MAIN_FILE).exists():
            found.add(child.name)
    return sorted(found)


def discover_sessions_for_rat(data_root: Path, rat: str) -> list[str]:
    """Session ids on disk for one rat, in either layout; see `discover_rat_sessions`."""
    return discover_rat_sessions(rat, data_root=data_root)


def load_session_mat(rat: str, session: str, data_root: str | Path | None = None) -> dict[str, Any]:
    """Raw session `.mat` as plain dicts, for `(rat, session)`: any layout, any MAT version.

    Use this rather than a bare `loadmat(data_root / rat / f"{rat}_{session}.mat")`: the path
    resolves for either layout (`session_mat_path`), and the file is read through
    `_loadmat_native`, so MATLAB v7.3 sessions work (`scipy.io.loadmat` raises
    `NotImplementedError` on them).

    Structs come back as dicts, so fields read as `mat["Clu"]["shank"]`, without the `[0][0]`
    that scipy's struct-array records require.
    """
    return _loadmat_native(session_mat_path(rat, session, data_root=data_root))


def _resolve_animal_session_main_file(
    animal: str,
    session_id: str,
    root: Path,
) -> tuple[Path, Path]:
    """Resolve a session file for mixed directory/file naming conventions.

    Supports both:
    - <root>/<animal>/<session_id>/rec01_BehavElectrDataLFP.mat
    - <root>/<animal>/<animal>_<session_id>.mat   (lee_lab format)
    - <root>/<animal>/<animal><session_id>*.mat   (e.g., *wMaps4G.mat)
    """
    animal_root = root / animal
    if not animal_root.exists():
        raise FileNotFoundError(f"Animal folder not found: {animal_root}")

    session_dir = animal_root / session_id
    if session_dir.is_dir():
        preferred_in_dir = [
            session_dir / "rec01_BehavElectrDataLFP.mat",
            session_dir / f"{animal}_{session_id}.mat",
            session_dir / f"{animal}{session_id}.mat",
            session_dir / f"{session_id}.mat",
        ]
        for p in preferred_in_dir:
            if p.exists():
                return p, session_dir
        any_in_dir = sorted(session_dir.glob("*.mat"))
        if any_in_dir:
            return any_in_dir[0], session_dir

    preferred_in_animal_root = [
        animal_root / f"{animal}_{session_id}.mat",
        animal_root / f"{animal}{session_id}.mat",
        animal_root / f"{session_id}.mat",
        animal_root / f"{session_id}_BehavElectrDataLFP.mat",
        animal_root / f"{animal}_{session_id}_BehavElectrDataLFP.mat",
        animal_root / f"{animal}{session_id}wMaps4G.mat",
    ]
    for p in preferred_in_animal_root:
        if p.exists():
            return p, p.parent

    glob_patterns = [
        f"{animal}*{session_id}*wMaps4G*.mat",
        f"*{session_id}*wMaps4G*.mat",
        f"{animal}*{session_id}*.mat",
        f"*{session_id}*.mat",
    ]
    for pat in glob_patterns:
        hits = sorted(animal_root.glob(pat))
        if hits:
            return hits[0], hits[0].parent

    raise FileNotFoundError(
        f"Could not resolve session for animal={animal!r}, session={session_id!r} under {animal_root}. "
        "Tried both folder-style and file-style naming patterns."
    )
