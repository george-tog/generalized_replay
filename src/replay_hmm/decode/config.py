"""The Lustig decode configuration: `DecodeConfig`, its run tag, and the flags built from it.

`DecodeConfig.tag()` hashes field values. Its fields, types and
defaults must not change, or existing run tags stop matching. No
`from __future__ import annotations` in this module: `add_decode_config_args` relies on
`dataclasses.fields(...).type` being the real class.

Every decode function takes an explicit config.
"""
import hashlib
import json
import warnings
from dataclasses import dataclass, fields

from replay_hmm.util.argtypes import bool_arg, opt_float_arg
from replay_hmm.util.tags import float_to_tag

PBE_FILTER_MODES = ("nonrun_nonwheel", "arm_end_reward_speed", "speed")
PBE_BASELINE_SCOPES = ("session", "lap")
PBE_TO_DECODE_BIN_RULES = ("coverage_half", "union")

# Fields omitted from DecodeConfig.tag()'s hash payload WHILE THEY HOLD THEIR REFERENCE VALUE
# (TAG_HASH_OMIT_REFERENCE). Each was added after saved fits existed; omitting it at that value
# keeps every existing `cfg<hash>` tag unchanged, while a different value still enters the
# hash, so tag() stays collision-free.
#
# `run_speed_thresh_m_s` is consumed only by the running-spikes fitting arm
# (`fit --dataset lustig --running`). `bin_width_cm` at None means the literal `bin_width_px`,
# which every such tag already encodes; a non-None value also changes the tag prefix.
#
# Only add a field here when no saved run could have set it to anything but its reference
# value. A field that has already been swept must stay in the payload.
TAG_HASH_OMIT_IF_DEFAULT: tuple[str, ...] = ("run_speed_thresh_m_s", "bin_width_cm")
# The values at which those two fields drop out of the hash. FROZEN, deliberately not read
# from the live dataclass defaults, so a default can change without any `cfg...` tag on disk
# moving.
TAG_HASH_OMIT_REFERENCE: dict[str, object] = {"run_speed_thresh_m_s": 0.05, "bin_width_cm": None}
# What a saved `decode_config` block means when it lacks a field: the value that field had when
# the block was written. FROZEN for the same reason as above: saved fits without
# `bin_width_cm` ran at None, whatever the live default is. A field not listed here fills from
# the live default (and warns).
PROVENANCE_MISSING_DEFAULTS: dict[str, object] = {"bin_width_cm": None}


@dataclass(frozen=True)
class DecodeConfig:
    """Every parameter that changes which spikes, bins, units, events or jumps survive.

    Every function that consumes a config takes it explicitly.
    """

    # --- decode grid -------------------------------------------------------
    decode_tbin_s: float = 0.010
    decode_tbin_step_s: float = 0.010
    bin_width_px: int = 8
    # A PHYSICAL bin-size target in cm. None keeps `bin_width_px` literal. Set it and the
    # pixel width is resolved PER SESSION by `resolve_bin_width_px`, because a pixel is not
    # one distance across this
    # dataset: the camera moved between recording groups, so 8 px spans ~3.5 cm for
    # TinyTim/Lucky/Ned/BB and ~2.8 cm for Spot/Huey/Stinky. Matching bins physically
    # is what makes anything measured in BINS -- a jump histogram before rescaling, a kernel
    # sigma -- comparable across the two groups.
    bin_width_cm: float | None = 3.5

    # --- behavioural gating ------------------------------------------------
    speed_thresh_m_s: float = 0.05
    # Movement-prior downsample stride. Consumed by `datasets.lustig.movement.
    # empirical_movement_P` (as `behavior_dt_s`), not by the bundle; carried here so one object
    # describes a whole run.
    dt_s: float = 0.100
    # None = use the raw speed trace. A float applies a zero-phase Butterworth low-pass at
    # that cutoff (e.g. 0.1 Hz).
    speed_smooth_cutoff_hz: float | None = None
    pbe_filter_mode: str = "nonrun_nonwheel"
    # Speed FLOOR for the running-spikes fitting arm (`fit --dataset lustig --running`), read by
    # `run_eligible_mask`. Deliberately separate from speed_thresh_m_s, which is the immobility
    # CEILING for PBE eligibility: one number cannot serve as both without forcing the two
    # gates to partition every sample, NaN-speed ones included. 0.05 m/s = 50 mm/s.
    # Omitted from tag()'s hash at this value -- see TAG_HASH_OMIT_IF_DEFAULT.
    run_speed_thresh_m_s: float = 0.05

    # --- PBE detection -----------------------------------------------------
    pbe_smooth_sigma_s: float = 0.020
    pbe_min_duration_s: float = 0.100
    pbe_merge_gap_s: float = 0.005
    pbe_spike_density_stdev: float = 3.0
    pbe_end_stdev: float = 0.0
    pbe_baseline_scope: str = "session"
    pbe_to_decode_bin_rule: str = "coverage_half"
    # Refuse to bridge a PBE gap that contains running time. False merges unconditionally,
    # which paints running samples into a span and lets the decode-timebase non-run AND split
    # the event below its own minimum duration.
    pbe_merge_gated: bool = True
    # Enforce pbe_min_duration_s on the FINAL decode-timebase runs, not just in the LFP base.
    # Catches window-width edge truncation: `dec_starts` stops at `end_20k - bin_w + 1` and
    # `local_ends` clips at the lap boundary, so a PBE near the end of a lap loses decode bins.
    # Independent of pbe_merge_gated, but both must be False together to reproduce the event
    # set of fits made with the ungated merge, because this filter also removes the runs the
    # ungated merge split.
    enforce_min_duration_on_decode_runs: bool = True

    # --- trajectory calling ------------------------------------------------
    trajectory_mass_threshold: float = 0.6
    trajectory_min_duration_s: float = 0.050 # 50 ms
    trajectory_merge_gap_s: float = 0.005 # 5 ms
    min_traj_prob_factor: float = 5.0

    # --- decode quality gates ----------------------------------------------
    # 0 disables. >0 zeroes the whole posterior row when fewer than this many
    # selected units fired in the window.
    min_active_cells: int = 0
    # Require every *selected* unit to have a finite rate for a state to be decodable.
    require_all_units_finite: bool = False

    # --- unit and map selection -------------------------------------------
    include_wheel_map: bool = True
    directional_unit_mode: str = "all"
    hpc_only: bool = True
    pyramidal_only: bool = False

    # --- jumps -------------------------------------------------------------
    hover_bins: int = 0

    def __post_init__(self):
        if self.pbe_filter_mode not in PBE_FILTER_MODES:
            raise ValueError(f"pbe_filter_mode={self.pbe_filter_mode!r} not in {PBE_FILTER_MODES}")
        if self.pbe_baseline_scope not in PBE_BASELINE_SCOPES:
            raise ValueError(f"pbe_baseline_scope={self.pbe_baseline_scope!r} not in {PBE_BASELINE_SCOPES}")
        if self.pbe_to_decode_bin_rule not in PBE_TO_DECODE_BIN_RULES:
            raise ValueError(
                f"pbe_to_decode_bin_rule={self.pbe_to_decode_bin_rule!r} not in {PBE_TO_DECODE_BIN_RULES}"
            )
        if not 0.0 < self.trajectory_mass_threshold <= 1.0:
            # 0 would let an all-zero (whitened-out) posterior row pass segment
            # assignment as segment 0.
            raise ValueError(
                f"trajectory_mass_threshold must be in (0, 1]; got {self.trajectory_mass_threshold}"
            )
        if self.decode_tbin_s <= 0 or self.decode_tbin_step_s <= 0:
            raise ValueError("decode_tbin_s and decode_tbin_step_s must both be > 0")

    @property
    def windowslideratio(self) -> int:
        """Decode window width in units of the stride."""
        return max(1, int(round(self.decode_tbin_s / self.decode_tbin_step_s)))

    def provenance(self) -> dict:
        """Flat dict for embedding in saved summaries."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def tag(self) -> str:
        """Short, filename-safe, collision-free identifier for this config.

        A readable prefix plus a hash of every field, except those in
        TAG_HASH_OMIT_IF_DEFAULT that still hold their reference value (see that constant for
        why). The hash is what makes it safe: omitting a field from the prefix cannot make two
        different configs share a name. That property survives the omission, because a field
        only drops out of the payload when it holds the one value every other config also holds.
        """
        prov = {
            name: value
            for name, value in self.provenance().items()
            if not (name in TAG_HASH_OMIT_IF_DEFAULT and value == TAG_HASH_OMIT_REFERENCE[name])
        }
        payload = json.dumps(prov, sort_keys=True, separators=(",", ":"), default=str)
        h = hashlib.sha1(payload.encode()).hexdigest()[:10]
        # A cm-target run gets its own namespace: its per-session pixel widths differ, so
        # `bw8` would be a lie for half the sessions and would collide with literal-8 runs.
        width_tag = (
            f"__bw{int(self.bin_width_px)}" if self.bin_width_cm is None
            else f"__bwcm{float_to_tag(float(self.bin_width_cm), replace_negative=True)}"
        )
        prefix = (
            f"dtbin_{float_to_tag(self.decode_tbin_s, replace_negative=True)}"
            f"__step_{float_to_tag(self.decode_tbin_step_s, replace_negative=True)}"
            f"{width_tag}"
        )
        return f"{prefix}__cfg{h}"


_CFG_CHOICES = {
    "pbe_filter_mode": PBE_FILTER_MODES,
    "pbe_baseline_scope": PBE_BASELINE_SCOPES,
    "pbe_to_decode_bin_rule": PBE_TO_DECODE_BIN_RULES,
}


class _Unset:
    """Sentinel for "this flag was not passed", distinct from a passed value of None."""

    def __repr__(self):
        return "<unset>"


UNSET = _Unset()


def _is_optional(annotation) -> bool:
    """`float | None` (a real union type here; the module has no future annotations)."""
    return type(None) in getattr(annotation, "__args__", ())


def add_decode_config_args(
    parser,
    *,
    prefix: str = "",
    exclude: tuple[str, ...] = (),
    defaults_unset: bool = False,
    nargs: str | None = None,
    aliases: dict[str, tuple[str, ...]] | None = None,
    defaults: "DecodeConfig | None" = None,
):
    """Add one argparse argument per DecodeConfig field.

    `defaults` supplies the flag defaults (and the converters derived from them); None means
    `DecodeConfig()`. A command whose own defaults must not follow DecodeConfig's (the MB jump,
    pinned at the values Fig 4B was made with) passes its own instance.

    Derived from `dataclasses.fields` rather than hand-written, so a field added to the
    config cannot be silently unreachable from the CLI.

    `defaults_unset=True` defaults every field to the UNSET sentinel rather than its
    dataclass value, so a caller can tell "not passed" from "passed". Job submitters want
    that: forward only what the user explicitly set and let the downstream CLI apply its own
    defaults to the rest, instead of pinning every field on every command line.

    The sentinel matters rather than just using None: `speed_smooth_cutoff_hz` is legitimately
    None, so with None as the "unset" marker an explicit `--speed-smooth-cutoff-hz none` is
    indistinguishable from omitting it.

    `nargs` (e.g. "+") makes every field accept a list, for sweeping callers like the Lustig
    MB jump; their defaults are wrapped in a list to match.

    `aliases` maps a field name to extra option strings accepted alongside the canonical flag.

    Returns the list of dest names added, for `decode_config_from_args`.
    """
    aliases = aliases or {}
    added = []
    for f in fields(DecodeConfig):
        if f.name in exclude:
            continue
        flag = "--" + (prefix + f.name).replace("_", "-")
        default = getattr(defaults if defaults is not None else DecodeConfig(), f.name)
        kwargs: dict = {
            "default": UNSET if defaults_unset else default,
            "dest": prefix + f.name,
        }
        # Derive the converter from the DEFAULT VALUE, not the annotation: defaults are
        # concrete and unambiguous. bool must precede int: bool subclasses int.
        if f.name in _CFG_CHOICES:
            kwargs["type"] = str
            kwargs["choices"] = list(_CFG_CHOICES[f.name])
        elif isinstance(default, bool):
            kwargs["type"] = bool_arg
            kwargs["metavar"] = "true|false"
        elif default is None or _is_optional(f.type):  # float | None
            kwargs["type"] = opt_float_arg
            kwargs["metavar"] = "FLOAT|none"
        elif isinstance(default, int):
            kwargs["type"] = int
        elif isinstance(default, float):
            kwargs["type"] = float
        else:
            kwargs["type"] = str
        if nargs is not None:
            kwargs["nargs"] = nargs
            if not isinstance(kwargs["default"], (list, _Unset)):
                kwargs["default"] = [kwargs["default"]]
        flags = [flag, *aliases.get(f.name, ())]
        parser.add_argument(*flags, help=f"DecodeConfig.{f.name} (default {default!r})", **kwargs)
        added.append(prefix + f.name)
    return added


def decode_config_from_args(args, *, prefix: str = "") -> DecodeConfig:
    """Build a DecodeConfig from a parsed namespace produced by `add_decode_config_args`."""
    kwargs = {}
    for f in fields(DecodeConfig):
        dest = prefix + f.name
        if hasattr(args, dest):
            kwargs[f.name] = getattr(args, dest)
    return DecodeConfig(**kwargs)


def decode_config_flags_from_args(args, *, prefix: str = "") -> list[str]:
    """CLI tokens for the DecodeConfig fields explicitly set on `args`.

    Intended for job submitters built on `add_decode_config_args(..., defaults_unset=True)`:
    forward exactly what the user asked for and let the downstream CLI apply its own defaults
    to everything else. Only fields still holding the UNSET sentinel are omitted -- an
    explicitly passed None is forwarded as the literal "none", which `opt_float_arg` parses
    back to None.
    """
    out: list[str] = []
    for f in fields(DecodeConfig):
        dest = prefix + f.name
        if not hasattr(args, dest):
            continue
        value = getattr(args, dest)
        if isinstance(value, _Unset):
            continue
        out.extend(["--" + dest.replace("_", "-"), "none" if value is None else str(value)])
    return out


def decode_config_from_provenance(prov: dict) -> DecodeConfig:
    """Rebuild a DecodeConfig from a saved `provenance()` dict.

    Unknown keys are dropped (with a warning) so a block written by a newer version still
    loads. A key the block lacks takes the value in PROVENANCE_MISSING_DEFAULTS, the value the
    field had when the block was written; a key absent from both fills from the current
    default, with a warning.
    """
    names = {f.name for f in fields(DecodeConfig)}
    prov = dict(prov)
    dropped = sorted(k for k in prov if k not in names)
    if dropped:
        warnings.warn(
            f"decode_config provenance has keys this DecodeConfig does not know and ignores: "
            f"{dropped}",
            RuntimeWarning,
            stacklevel=2,
        )
    kwargs = {k: v for k, v in prov.items() if k in names}
    for name, value in PROVENANCE_MISSING_DEFAULTS.items():
        kwargs.setdefault(name, value)
    unfilled = sorted(names - set(kwargs))
    if unfilled:
        warnings.warn(
            f"decode_config provenance lacks {unfilled}; filled from the current defaults",
            RuntimeWarning,
            stacklevel=2,
        )
    return DecodeConfig(**kwargs)
