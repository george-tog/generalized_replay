"""Canonical arm/slot labels for the Lustig (Lee lab) M-maze.

Every figure, CSV column, and dict key names a physical arm through this module, so the arm
labels cannot drift apart between consumers. The convention, checked against raw 2-D tracking
and the arm polylines:

    stored linear slot order = [L_in, L_out, C_in, C_out, R_in, R_out]   <- EVEN index = INBOUND
    linearize() armID        = 1:Left, 2:Center, 3:Right
    mazeSect on-arm          = 4:Left, 5:Right, 6:Center
    mazeSect reward port     = 7:Left, 8:Right, 9:Center   (median armDist 1.00, i.e. at the well)

Why the stored order is in-first: `maps.build.build_map_linear` builds slots as
`slot = (arm-1)*2 + d_idx` with `dir_masks = [outbound, inbound]`, giving a pre-remap
[L_out, L_in, C_out, C_in, R_out, R_in]; `LINEAR_SLOT_REINDEX = [1, 0, 3, 2, 5, 4]` then swaps
within each arm pair at save time. That parity is load-bearing: both slots of an arm put bin 0
at the hub, so inbound running lowers the bin index, and the jump code negates even-indexed
slots so positive means motion along each slot's own running direction (forward). Believing the
pre-remap order inverts the sign of every reported asymmetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from replay_hmm import paths
from replay_hmm.io.matlab import _as_1d


PROJECT_DIR = paths.REPO_ROOT


@dataclass(frozen=True)
class MapSpec:
    """One decode candidate: its index in the concatenated stack and what it physically is."""

    order: int
    label: str
    family: str
    arm: str | None
    template_direction: str | None
    legacy_label: str | None = None


# Orders 0-5 are the directional linear slots in STORED order, 6 is the wheel, 7-9 are the 2-D
# per-arm candidates used by the traditional decoder. `label`/`arm` are physical, so directional
# and 2-D candidates for one arm can be compared directly. `legacy_label` is the L<->R-transposed
# name that earlier result files carry, so rows stay joinable against them.
MAP_SPECS: tuple[MapSpec, ...] = (
    MapSpec(0, "L_in", "directional", "Left", "Inbound", "R_in"),
    MapSpec(1, "L_out", "directional", "Left", "Outbound", "R_out"),
    MapSpec(2, "C_in", "directional", "Center", "Inbound", "C_in"),
    MapSpec(3, "C_out", "directional", "Center", "Outbound", "C_out"),
    MapSpec(4, "R_in", "directional", "Right", "Inbound", "L_in"),
    MapSpec(5, "R_out", "directional", "Right", "Outbound", "L_out"),
    MapSpec(6, "Wheel", "wheel", None, None, "Wheel"),
    MapSpec(7, "L_2D", "2d", "Left", None, "L_2D"),
    MapSpec(8, "C_2D", "2d", "Center", None, "C_2D"),
    MapSpec(9, "R_2D", "2d", "Right", None, "R_2D"),
)
MAP_SPEC_BY_ORDER: dict[int, MapSpec] = {spec.order: spec for spec in MAP_SPECS}

N_LINEAR_SLOTS = 6
WHEEL_SLOT = 6

_DIRECTIONAL: tuple[MapSpec, ...] = tuple(
    MAP_SPEC_BY_ORDER[order] for order in range(N_LINEAR_SLOTS)
)

# --- Derived views. Every one of these is a projection of MAP_SPECS, never a restatement. ---

PHYSICAL_MAP_LABELS: tuple[str, ...] = tuple(spec.label for spec in _DIRECTIONAL)
LEGACY_MAP_LABELS: tuple[str, ...] = tuple(spec.legacy_label for spec in _DIRECTIONAL)
MAP_ARMS: tuple[str, ...] = tuple(spec.arm for spec in _DIRECTIONAL)
MAP_DIRECTIONS: tuple[str, ...] = tuple(spec.template_direction for spec in _DIRECTIONAL)

# Slot index -> single-letter arm, and slot index -> label, as dicts for lookup-by-index code.
PHYS_ARM: dict[int, str] = {spec.order: spec.arm[0] for spec in _DIRECTIONAL}
PHYS_MAP_LABEL: dict[int, str] = {spec.order: spec.label for spec in _DIRECTIONAL}

ARM_NAMES: tuple[str, ...] = ("Left", "Center", "Right")
ARM_ABBREV: tuple[str, ...] = ("L", "C", "R")

# linearize() assigns armID by nearest of the three named polylines, in [Left, Center, Right] order.
ARM_ID_TO_NAME: dict[int, str] = {1: "Left", 2: "Center", 3: "Right"}
ARM_NAME_TO_ID: dict[str, int] = {name: arm_id for arm_id, name in ARM_ID_TO_NAME.items()}

ARM_TO_SLOTS: dict[str, tuple[int, ...]] = {
    arm: tuple(spec.order for spec in _DIRECTIONAL if spec.arm == arm) for arm in ARM_NAMES
}
ARM_ID_TO_SLOTS: dict[int, tuple[int, ...]] = {
    arm_id: ARM_TO_SLOTS[name] for arm_id, name in ARM_ID_TO_NAME.items()
}
SLOT_BY_LABEL: dict[str, int] = {spec.label: spec.order for spec in MAP_SPECS}

# The 7-slot form, wheel LAST, matching the concatenation order of the decode state vector.
SEGMENT_ORDER: tuple[str, ...] = PHYSICAL_MAP_LABELS + ("Wheel",)
LINEAR_SEGMENTS: tuple[str, ...] = PHYSICAL_MAP_LABELS

# Prose form for figure tick labels ("left in", "left out", ...).
MAP_LABELS_PROSE: tuple[str, ...] = tuple(
    f"{spec.arm.lower()} {'in' if spec.template_direction == 'Inbound' else 'out'}"
    for spec in _DIRECTIONAL
) + ("wheel",)

# --- Track.mazeSect codebook. Verified against the arm polylines in all 19 sessions. ---

ONARM_ARM: dict[int, str] = {4: "L", 5: "R", 6: "C"}
REWARD_ARM: dict[int, str] = {7: "L", 8: "R", 9: "C"}
ARM_FROM_CODE: dict[int, str] = {**ONARM_ARM, **REWARD_ARM}
ARM_END_SECTION_TO_NAME: dict[int, str] = {
    code: dict(zip(ARM_ABBREV, ARM_NAMES))[abbrev] for code, abbrev in REWARD_ARM.items()
}


def remap_dir_choice(dir_choice: np.ndarray) -> np.ndarray:
    """Remap Laps.dirChoice as `na_remapDirChoice` does: 1->1, 2->3, 3->2."""
    old = _as_1d(dir_choice, dtype=int)
    new = old.copy()
    new[old == 2] = 3
    new[old == 3] = 2
    return new


# Laps.dirChoice AFTER remap_dir_choice above (1->1, 2->3, 3->2).
DIRCHOICE_ARM: dict[int, str] = {1: "L", 2: "C", 3: "R"}

WHEEL_SECTIONS: tuple[int, ...] = (2, 3)
HUB_SECTIONS: tuple[int, ...] = (0, 1)
ARM_END_SECTIONS: tuple[int, ...] = tuple(sorted(REWARD_ARM))

# Slot label -> the mazeSect codes whose samples belong to it. Both the on-arm and the reward-port
# code for that arm, since a slot spans the whole arm including the well.
SEGMENT_CODES: dict[str, tuple[int, ...]] = {
    spec.label: tuple(
        sorted(code for code, abbrev in ARM_FROM_CODE.items() if abbrev == spec.arm[0])
    )
    for spec in _DIRECTIONAL
}
SEGMENT_CODES["Wheel"] = WHEEL_SECTIONS

# --- The reindex the map archive must have been built under for any of the above to hold. ---

EXPECTED_SLOT_REINDEX: tuple[int, ...] = (1, 0, 3, 2, 5, 4)


def slots_for_arm(arm: int | str) -> tuple[int, ...]:
    """Stored slot indices for an arm, given either its armID (1/2/3) or its name."""
    if isinstance(arm, str):
        name = arm if arm in ARM_TO_SLOTS else {a[0]: a for a in ARM_NAMES}.get(arm.upper())
        if name is None:
            raise KeyError(f"unknown arm {arm!r}; expected one of {ARM_NAMES} or L/C/R")
        return ARM_TO_SLOTS[name]
    return ARM_ID_TO_SLOTS[int(arm)]


def arm_only_arg(value) -> str:
    """argparse type for one physical maze arm, normalised to "L"/"C"/"R".

    Validated against `slots_for_arm` rather than a literal `choices=[...]`, so the
    accepted spellings and the slots they resolve to cannot drift apart. Raises ValueError,
    which argparse reports as an "invalid value" error.

    "2" is deliberately rejected even though `slots_for_arm` takes the integer armID 2: as a
    CLI string it is ambiguous between armID 2 (Center) and stored slot 2 (C_in).
    """
    text = str(value).strip()
    for spelling in (text, text.upper(), text.title()):
        if not spelling:
            continue
        try:
            slots_for_arm(spelling)
        except (KeyError, ValueError, TypeError):
            continue
        return spelling[0].upper()
    raise ValueError(
        f"expected one of L/C/R or {ARM_NAMES}, got {value!r}"
    )


def maps_path(rat: str, session: str, bin_width_px: float, results_root: Path | None = None) -> Path:
    """Conventional location of a session's map archive."""
    root = Path(results_root) if results_root is not None else PROJECT_DIR / "results"
    return root / rat / session / f"lustig_maps_abs_bw{int(bin_width_px)}.npz"


def verify_slot_order(maps: dict[str, Any], rat: str, session: str) -> None:
    """Fail loudly unless the saved maps use the slot order these labels name.

    The physical labels only hold if the map builder applied LINEAR_SLOT_REINDEX. An archive built
    under a different reindex would silently invert every arm label, every local/remote call, and
    the sign of every jump asymmetry, so check the stamp the map builder leaves behind rather than
    trusting it. Maps built by the MATLAB pipeline (na_buildMapLinear) apply no reindex and
    carry no stamp, so they are rejected.
    """
    linear = maps["linear"]
    if "slot_reindex_applied" not in linear:
        raise KeyError(
            f"{rat}/{session}: map archive has no linear['slot_reindex_applied'], so the "
            "physical slot order cannot be confirmed; regenerate it with "
            "`replay-hmm maps build --dataset lustig`."
        )
    reindex = tuple(int(v) for v in np.asarray(linear["slot_reindex_applied"]).ravel())
    if reindex != EXPECTED_SLOT_REINDEX:
        raise ValueError(
            f"{rat}/{session}: map archive slot_reindex_applied={reindex} != "
            f"{EXPECTED_SLOT_REINDEX}; the physical arm labels would be wrong."
        )


def verify_slot_order_for_session(
    rat: str,
    session: str,
    bin_width_px: float,
    maps_root: Path | None = None,
) -> None:
    """Load a session's map archive and assert it uses the physical slot order we label against.

    A missing archive is an error, not a warning: this guard is the only thing standing between
    a re-ordered map archive and silently L/R-swapped arm labels, so it must not be skippable.
    `maps_root` defaults to the repo's results/ but is passed explicitly by callers that were
    given a --maps-root, which is NOT the fits root.
    """
    path = maps_path(rat, session, bin_width_px, results_root=maps_root)
    if not path.exists():
        raise FileNotFoundError(
            f"[{rat}/{session}] map archive {path} not found, so the physical slot order "
            "cannot be confirmed; build it with `replay-hmm maps build --dataset lustig`."
        )
    with np.load(path, allow_pickle=True) as maps_npz:
        verify_slot_order(maps_npz["Maps"][0], rat, session)


def _self_check() -> None:
    """Guard the derived views against MAP_SPECS drift, at import so no caller can miss it."""
    assert PHYSICAL_MAP_LABELS == ("L_in", "L_out", "C_in", "C_out", "R_in", "R_out")
    assert LEGACY_MAP_LABELS == ("R_in", "R_out", "C_in", "C_out", "L_in", "L_out")
    assert MAP_ARMS == ("Left", "Left", "Center", "Center", "Right", "Right")
    # Even slot = inbound. The jump sign convention depends on this parity.
    assert MAP_DIRECTIONS == ("Inbound", "Outbound") * 3
    assert all(
        (order % 2 == 0) == (MAP_SPEC_BY_ORDER[order].template_direction == "Inbound")
        for order in range(N_LINEAR_SLOTS)
    )
    # A slot's mazeSect codes must name the same arm the slot does.
    for spec in _DIRECTIONAL:
        assert all(ARM_FROM_CODE[code] == spec.arm[0] for code in SEGMENT_CODES[spec.label])
    assert SEGMENT_ORDER[-1] == "Wheel" and len(SEGMENT_ORDER) == 7

    # The reindex we assert against must be the one the map builder actually applies.
    from replay_hmm.datasets.lustig.constants import LINEAR_SLOT_REINDEX

    applied = tuple(int(v) for v in np.asarray(LINEAR_SLOT_REINDEX).ravel())
    assert applied == EXPECTED_SLOT_REINDEX, (
        f"datasets.lustig.constants.LINEAR_SLOT_REINDEX={applied} != EXPECTED_SLOT_REINDEX="
        f"{EXPECTED_SLOT_REINDEX}; every label in this module would be wrong."
    )


_self_check()
