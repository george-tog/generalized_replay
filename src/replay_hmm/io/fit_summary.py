"""Readers of a saved fit summary JSON.

Applying a summary's parameters to a rebuilt model is `replay_hmm.hmm.restore`.
`fit_event_source` reads which event set a fit was trained on, for every dataset, each caller
naming its own `EventSourcePolicy`; `resolve_traditional_events_csv` finds a Lustig traditional
fit's events CSV (the arm restriction is
`replay_hmm.datasets.lustig.event_sets.fit_mb_traj_arm`).
"""
from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
import json
from pathlib import Path


def load_saved_fit_summary(filepath_summary: Path) -> tuple[dict, dict[str, dict]]:
    """Load prior full-fit summary and index rows by model name."""
    if not filepath_summary.exists():
        raise FileNotFoundError(f"Missing saved full-fit summary: {filepath_summary}")

    payload = json.loads(filepath_summary.read_text())
    rows = payload.get("models", None)
    if not isinstance(rows, list):
        raise ValueError(f"Saved summary has invalid `models` payload: {filepath_summary}")

    rows_by_model: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or "model" not in row:
            continue
        rows_by_model[str(row["model"])] = row

    if len(rows_by_model) == 0:
        raise ValueError(f"No model rows found in saved summary: {filepath_summary}")

    return payload, rows_by_model


def momentum_kwargs_from_summary(summary: dict, *, fallback_bin_width_cm=None) -> dict:
    """The four `momentum` arguments `build_two_map_hmm_dict` needs, taken from a fit summary.

    Same rule as the eps and sigma caps a rebuilding caller already reads from the summary:
    the fit's own values, not this module's defaults. `bin_width_cm` and the decode step are
    STRUCTURAL for this kernel -- they are what make sigma_m mean the same thing across decode
    grids -- so rebuilding at a different width would restore (lambda_m, sigma_m) into a
    kernel those numbers do not describe.

    The grid axes are provenance rather than correctness: the selected point is restored by
    `hmm.restore.apply_transition_params_from_summary_row`, which rebuilds from
    lambda_m and sigma_m directly. They are passed anyway so a rebuilt model reports the axes
    the fit actually swept instead of re-deriving the current defaults from dt and geometry.

    `fallback_bin_width_cm` covers summaries with no `momentum_bin_width_cm`. Callers pass
    whatever their own geometry key holds (Buzsaki records metres, Pfeiffer centimetres),
    and a rebuild that does not request
    `momentum` never reads the result anyway.
    """
    bin_width_cm = summary.get("momentum_bin_width_cm")
    if bin_width_cm is None:
        bin_width_cm = fallback_bin_width_cm
    a_grid = summary.get("momentum_a_grid")
    sigma_v_grid = summary.get("momentum_sigma_v_grid")
    eps = summary.get("momentum_eps")
    return {
        "bin_width_cm": None if bin_width_cm is None else float(bin_width_cm),
        "momentum_a_grid": None if a_grid is None else tuple(float(v) for v in a_grid),
        "momentum_sigma_v_grid": (
            None if sigma_v_grid is None else tuple(float(v) for v in sigma_v_grid)
        ),
        "momentum_eps": None if eps is None else float(eps),
    }


TRADITIONAL_SOURCE = "traditional"
# The event sets a fit can be trained on, as the fit CLIs record them in `event_source`.
EVENT_SOURCES = (TRADITIONAL_SOURCE, "all_pbe", "mb_traj")


@dataclass(frozen=True)
class EventSourcePolicy:
    """How one reader of a fit's `event_source` treats a summary that lacks it, or holds a
    value the reader does not accept. The readers of `fit_event_source` differ on both, so each
    passes one of the named policies below.

    `missing` is what a summary without the key resolves to:
      "use_all_pbes_or_flag"  the legacy `use_all_pbes` boolean when the key is present, else
                              the caller's `all_pbe_jumps` flag;
      "use_all_pbes"          the legacy boolean's truthiness (an absent key reads False);
      "mb_traj"               "mb_traj";
      "raise"                 ValueError(`missing_message`);
      "keep"                  the value itself, which then meets `accepted` like any other.
    `missing_if_falsy` counts every falsy value as missing, as a reader written with `or` does;
    otherwise only None is. `rejected` lists values refused (with `unknown_exc`) before
    `accepted` is consulted, each with its own message. `accepted` (None: anything) is tested
    on the value as read, in the reader's own container: a frozenset refuses an unhashable value
    with TypeError, and a tuple compares by equality.
    `unknown_message` is formatted with `label`, `source` (the value as read) and `source_str`
    (its str). `as_str=False` returns the value as read instead of its str.
    """
    missing: str
    missing_if_falsy: bool = False
    missing_message: str = ""
    rejected: tuple[tuple[str, str], ...] = ()
    accepted: Collection | None = None
    unknown_exc: type[BaseException] = ValueError
    unknown_message: str = ""
    as_str: bool = True


# The Lustig jump distribution, per-event evidence and kernel PPC. `event_source` is
# authoritative. Fits without it recorded `use_all_pbes` instead, which is read next, before
# the CLI flag: consulting the flag first would compare the flag against itself, so an
# __allpbe fit scored over trajectory segments would agree with any flag it was given. Only a
# fit carrying neither key falls back to the flag, and no fit trained on traditional events
# lacks both.
LUSTIG_FIT = EventSourcePolicy(
    missing="use_all_pbes_or_flag",
    # A real arm (`fit --dataset lustig --running`), deliberately refused here rather than
    # unrecognised: jumps are harvested from replay-type events, and there is no sense in
    # which a running bout has a "jump" to compare against the null. It must never be
    # treated as mb_traj.
    rejected=((
        "running",
        "this fit was trained on running bouts (event_source='running'); the jump "
        "distribution is defined over replay events, so there is no matching jump event "
        "set. Point --run-tag at a trajectory, all-PBE or traditional fit instead.",
    ),),
    accepted=EVENT_SOURCES,
    unknown_message=(
        "unrecognised event_source {source!r} in the saved fit summary; expected one of "
        f"{list(EVENT_SOURCES)}."
    ),
)
# The two-map kernel PPC's scope check and the two-map viewer: fits without the key recorded
# only `use_all_pbes`; any recorded value is returned as is.
TWO_MAP_PPC = EventSourcePolicy(missing="use_all_pbes")
# A saved Buzsaki fit's event set, as its per-event kernel evidence (`score-events`) reads it.
# Taken from the fit rather than from a run's flags so the scored events cannot silently disagree
# with the fitted ones -- which is also what makes the sum-to-stored-loglik check meaningful.
# Fits without the key recorded only `use_all_pbes`, and its absence means the fit CLI's
# default (False), i.e. the trajectory collector ran. `accepted` is the fit's EVENT_SET_LABELS'
# keys.
BUZSAKI_FIT = EventSourcePolicy(
    missing="use_all_pbes",
    accepted=frozenset(EVENT_SOURCES),
    unknown_message=(
        "[{label}] unrecognised event_source {source!r} in the fit summary; "
        f"expected one of {sorted(EVENT_SOURCES)}."
    ),
)
# A saved Pfeiffer fit's event set, as its jump distribution and kernel evidence read it: a fit
# without the key is refused. `accepted` is the fit's EVENT_SET_LABELS' keys.
PFEIFFER_FIT = EventSourcePolicy(
    missing="raise",
    missing_message=(
        "This fit records no 'event_source'. It predates the provenance contract and "
        "cannot be scored safely -- refit it."
    ),
    accepted=frozenset(EVENT_SOURCES),
    unknown_message="Unrecognised event_source {source!r} in fit summary.",
)
# `datasets.lustig.event_sets.fit_mb_traj_arm`: an arm restriction applies to mb_traj only, and
# an empty or absent key is mb_traj.
ARM_GATE = EventSourcePolicy(missing="mb_traj", missing_if_falsy=True)
# Cross-session transfer. use_all_pbes is the legacy boolean that cannot name the traditional
# or running arms; dispatching on it alone would score a running-trained fit on MB-trajectory
# events. The transfer rebuilds only the mb_traj and all_pbe event sets.
CROSS_SESSION = EventSourcePolicy(
    missing="use_all_pbes",
    missing_if_falsy=True,
    accepted=("mb_traj", "all_pbe"),
    unknown_exc=NotImplementedError,
    unknown_message=(
        "[{label}] this fit was trained on event_source={source_str!r}; "
        "cross-session transfer only knows how to rebuild the mb_traj and all_pbe event "
        "sets, so the rebuilt events would not be the ones the donor parameters were "
        "estimated against."
    ),
)
# The misspecification gallery's template: "every event of the source fit", all mbtraj
# segments or all PBE windows. A summary without the key is refused like any other value.
MISSPEC_TEMPLATE = EventSourcePolicy(
    missing="keep",
    accepted=("mb_traj", "all_pbe"),
    unknown_exc=SystemExit,
    unknown_message="{label} is neither an unrestricted mbtraj nor an all-PBE fit.",
    as_str=False,
)


# Synthetic recovery's template on a two-map fit: the fitted event set's lengths, trajectory
# segments (mb_traj) or whole candidate-event windows (all_pbe). Traditional-replay fits need
# their events table and are refused, as is a summary without the key.
RECOVERY_TEMPLATE = EventSourcePolicy(
    missing="keep",
    accepted=("mb_traj", "all_pbe"),
    unknown_exc=SystemExit,
    unknown_message=(
        "[{label}] this fit was trained on event_source={source_str!r}; synthetic recovery "
        "rebuilds only the mb_traj and all_pbe event sets."
    ),
)


def fit_event_source(
    summary_saved: dict,
    *,
    policy: EventSourcePolicy,
    all_pbe_jumps: bool | None = None,
    label: str = "",
):
    """Which event set a saved fit was trained on, from the fit's own summary.

    `policy` is the caller's (one of the named policies above); `all_pbe_jumps` is the CLI flag
    the "use_all_pbes_or_flag" policy falls back to, and `label` names the session in the
    messages that carry one.
    """
    if policy.missing == "use_all_pbes_or_flag" and all_pbe_jumps is None:
        raise TypeError("this event-source policy falls back to the CLI flag: pass all_pbe_jumps")
    source = summary_saved.get("event_source")
    if (not source) if policy.missing_if_falsy else source is None:
        if policy.missing == "use_all_pbes_or_flag":
            if "use_all_pbes" in summary_saved:
                return "all_pbe" if summary_saved["use_all_pbes"] else "mb_traj"
            return "all_pbe" if all_pbe_jumps else "mb_traj"
        if policy.missing == "use_all_pbes":
            return "all_pbe" if summary_saved.get("use_all_pbes") else "mb_traj"
        if policy.missing == "mb_traj":
            return "mb_traj"
        if policy.missing == "raise":
            raise ValueError(policy.missing_message)
        if policy.missing != "keep":
            raise ValueError(f"unknown event-source policy for a missing key: {policy.missing!r}")
    for value, message in policy.rejected:
        if source == value:
            raise policy.unknown_exc(message)
    if policy.accepted is not None and source not in policy.accepted:
        raise policy.unknown_exc(
            policy.unknown_message.format(label=label, source=source, source_str=str(source))
        )
    return str(source) if policy.as_str else source


def resolve_traditional_events_csv(
    summary_saved: dict,
    override: Path | None,
    *,
    rat: str,
    session: str,
) -> Path:
    """Path to the traditional-decode CSV this fit's event set comes from.

    The recorded path is absolute and was written wherever the fit ran (e.g. on the cluster),
    so the override is the ordinary case locally, not an escape hatch. When it is used, the
    CSV's run-tag directory must match the one the fit recorded: substituting a *different*
    traditional decode would change the event set while leaving every tag and filename
    identical.
    """
    recorded = summary_saved.get("traditional_events_csv")
    recorded_tag = summary_saved.get("traditional_run_tag")
    if override is not None:
        override = Path(override)
        if not override.exists():
            raise FileNotFoundError(f"--traditional-events-csv not found: {override}")
        if recorded_tag and override.parent.name != str(recorded_tag):
            raise ValueError(
                f"[{rat}/{session}] --traditional-events-csv is under "
                f"{override.parent.name!r} but this fit was trained on traditional run tag "
                f"{recorded_tag!r}. Pass the CSV from that run, or the scored event set will "
                "not be the fitted one."
            )
        return override
    if recorded and Path(recorded).exists():
        return Path(recorded)
    raise FileNotFoundError(
        f"[{rat}/{session}] this fit was trained on traditional events from {recorded!r}, "
        "which does not exist here (fits made on the cluster record a cluster path). Pass "
        "--traditional-events-csv pointing at that traditional decode's "
        "traditional_decode_events.csv."
    )
