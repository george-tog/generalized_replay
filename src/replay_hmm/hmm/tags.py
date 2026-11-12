"""Run-tag grammar for the HMM fits: the suffixes each fit CLI appends to its directory name.

A run tag is the contract between a fit and everything that reads it back (the strip plotters
select every session with one --run-tag), so each suffix here is spelled once and every CLI that
writes or rebuilds a tag calls it. Each suffix names a MODE, not a resolved number, and is empty
at the shipped default, so adding an axis never renames an existing directory.
"""
from __future__ import annotations

import hashlib

from replay_hmm.hmm import constants
from replay_hmm.util import tags


def sigma_cap_tag_suffix(sigma_cap) -> str:
    """Run-tag suffix for a non-default sigma cap (in bins); empty string at the default."""
    if float(sigma_cap) == float(constants.SIGMA_CAP):
        return ""
    return f"__sigmacap{tags.fmt_float_tag(float(sigma_cap))}"


def hard_zero_tag_suffix(hard_zero: bool) -> str:
    """Run-tag suffix for the hard-zero arm (eps = alpha = 0 exactly); empty otherwise."""
    return "__hardzero" if hard_zero else ""


# "C" spells __centerarm rather than __armC. An explicit table rather than an f-string, so
# these existing directory names cannot change by accident.
ARM_ONLY_TAG_SUFFIXES: dict[str, str] = {"C": "__centerarm", "L": "__armL", "R": "__armR"}


def arm_only_tag_suffix(arm_only) -> str:
    """Run-tag suffix for a one-arm mb_traj event set; empty when unrestricted."""
    if arm_only is None:
        return ""
    key = str(arm_only).strip().upper()
    if key not in ARM_ONLY_TAG_SUFFIXES:
        raise ValueError(
            f"arm_only must be one of {sorted(ARM_ONLY_TAG_SUFFIXES)}, got {arm_only!r}"
        )
    return ARM_ONLY_TAG_SUFFIXES[key]


def sigma_cap_tag_suffix_cm(sigma_cap_cm, default_cm: float) -> str:
    """Run-tag suffix for a non-default sigma cap (in cm); empty string at the default.

    Neither dataset's provenance-collision guard inspects sigma, so this suffix is the only
    thing keeping fits with different caps out of one directory.
    """
    if float(sigma_cap_cm) == float(default_cm):
        return ""
    return f"__sigmacap{tags.fmt_float_tag(float(sigma_cap_cm))}cm"


def momentum_grid_tag_suffix(a_grid, sigma_v_grid, momentum_eps) -> str:
    """Run-tag suffix naming the momentum sweep; empty string at the shipped defaults.

    Two runs with different grids select different points and score differently, so without
    a suffix they would land in one directory and be plotted together as though they were the
    same fit. That is exactly the collision `sigma_cap_tag_suffix_cm` exists to prevent, and
    the provenance guard does not inspect the grid either.

    The suffix is a HASH rather than the values: a full grid is too many numbers to go in
    a path, while one --run-tag has to select every session for a strip figure. The grid
    itself is written to the provenance dict, which is where it should be read from.
    """
    eps = constants.EPS_INIT if momentum_eps is None else float(momentum_eps)
    # "Default" is the MODE (both axes auto), not a comparison against constants: the
    # sigma_v axis is derived from each session's own dt and geometry, so there is no fixed
    # tuple to compare to. Same reasoning as `sigma_cap_tag_suffix_cm`.
    if a_grid is None and sigma_v_grid is None and eps == float(constants.EPS_INIT):
        return ""
    lam = None if a_grid is None else tuple(float(v) for v in a_grid)
    sig = None if sigma_v_grid is None else tuple(float(v) for v in sigma_v_grid)
    payload = repr((lam, sig, eps)).encode()
    return "__momgrid" + hashlib.blake2b(payload, digest_size=4).hexdigest()


def decode_run_tag(
    decode_tbin_s: float,
    decode_tbin_step_s: float,
    all_pbe_hmm: bool = False,
) -> str:
    """The `dtbin_X__step_Y[__allpbe]` run tag of saved HMM fit directories, without the
    `__bw…__cfg<hash>` part that `DecodeConfig.tag()` adds to Lustig tags."""
    tag = (
        f"dtbin_{tags.fmt_float_tag(float(decode_tbin_s))}"
        f"__step_{tags.fmt_float_tag(float(decode_tbin_step_s))}"
    )
    if all_pbe_hmm:
        tag += "__allpbe"
    return tag
