"""Per-event kernel evidence: every event's log-likelihood under each scored kernel.

One event's evidence is its log-likelihood under each of the scored kernels
(SCORED_KERNELS -- the comparison ballot plus the off-ballot arms), the argmax
over the BALLOT alone (`best_kernel`), and `coherent_margin_per_bin` -- how far the best
coherent kernel beats the best incoherent one, per bin. All three are properties of the
*event*, not of whichever kernel is being decoded for a figure, which is why they live in one
filter- and decoder-independent cache that every consumer reads.

This module is dataset-agnostic: it scores a list of event handles
`{trial, event_id, s, e, spike_traj}` against a rebuilt `hmm_dict` and writes the cache. Each
dataset's `datasets.<ds>.kernel_share` rebuilds its fits' event sets and calls it; the cache
layout, columns and readers are `replay_hmm.io.evidence`.
"""
from __future__ import annotations

from pathlib import Path

from replay_hmm.hmm.kernel_sets import (
    COHERENT_KERNELS,
    COMPARISON_KERNELS,
    DIRECTION_KERNELS,
    INCOHERENT_KERNELS,
    OFF_BALLOT_KERNELS,
    PARAM_DONORS,
    SCORED_KERNELS,
    ll_col,
)
from replay_hmm.hmm.scoring import check_stored_loglik
from replay_hmm.io import names
from replay_hmm.io.evidence import (
    BEST_DIRECTION_COL,
    DIRECTION_EVIDENCE_COLUMNS,
    EVIDENCE_COLUMNS,
    write_evidence,
)


# --------------------------------------------------------------------------- #
# The calculation
# --------------------------------------------------------------------------- #
def score_events_per_kernel(
    events: list[dict], hmm_dict: dict, rows_by_model: dict, *, rat: str, session: str,
    kernels: tuple[str, ...] = SCORED_KERNELS,
    param_donors: dict[str, str] | None = None,
    restore=None,
) -> dict:
    """Per-event log-likelihood under each kernel in `kernels`.

    Independent of which kernels were requested for decoding: the filter is a property of
    the event, so it must be computed over the whole comparison set or not at all. A missing
    kernel raises -- a filter silently evaluated over four kernels is worse than a failure,
    because nothing downstream would reveal it.

    `param_donors` maps a kernel with no fit of its own to the fitted kernel whose transition
    parameters it borrows (PARAM_DONORS). Empty for every ballot -- every kernel on every
    ballot is fitted -- and an empty mapping changes nothing. A kernel listed here reads its
    parameters from the donor's row and is exempt from the stored-loglik identity check below,
    since that stored value belongs to the donor.

    Correctness check for free: ssm sums log-likelihood over independent sequences, so when
    the event set here is the one the fit was trained on (equal total bin count) the
    per-event values must sum to the loglik the fit stored. Mirrors the self-transfer check
    in analyses.cross_session. A borrowed kernel is exempt -- the stored loglik belongs
    to the donor, computed under the donor's kernel -- but the bin-count check still applies,
    because the donor row describes the same event set.

    Leaves every hmm in `hmm_dict` carrying the last-scored kernel's transition params, so a
    caller that decodes with one of them afterwards must re-apply its own row first.

    `restore` is the module that applies a summary row (default `replay_hmm.hmm.restore`).
    """
    import numpy as np

    if restore is None:
        from replay_hmm.hmm import restore
    # Applies the house rcParams.
    from replay_hmm.plots.style import COLOR_PALETTE  # noqa: F401

    param_donors = dict(param_donors or {})
    param_row_for = {k: param_donors.get(k, k) for k in kernels}

    missing = [
        k for k in kernels if param_row_for[k] not in rows_by_model or k not in hmm_dict
    ]
    # An OFF-BALLOT kernel is scored for the record and nothing else: it cannot win an event
    # and it does not enter the coherence margin, so its absence costs a column, not a
    # correctness guarantee. Dropping it keeps a fit without segment_uniform scorable. The
    # ballot kernels stay all-or-nothing for the reason in the docstring -- a filter quietly
    # evaluated over a short ballot is worse than a failure.
    missing_optional = [k for k in missing if k in OFF_BALLOT_KERNELS]
    missing_required = [k for k in missing if k not in OFF_BALLOT_KERNELS]
    if missing_required:
        # The remedy differs by ballot, so name the one actually in play.
        remedy = (
            "Refit this session with the full kernel set."
            if set(kernels) >= set(COMPARISON_KERNELS)
            else "Refit this session with the kernels this ballot needs "
                 f"(donors: {param_donors or 'none'})."
        )
        raise ValueError(
            f"[{rat}/{session}] scoring needs all of "
            f"{[k for k in kernels if k not in OFF_BALLOT_KERNELS]} "
            f"but these are absent from the saved fit summary: {missing_required}. {remedy}"
        )
    if missing_optional:
        print(f"  [{rat}/{session}] not in this fit, so left unscored (off the argmax "
              f"ballot, nothing downstream depends on them): {missing_optional}")
        kernels = tuple(k for k in kernels if k not in missing_optional)

    total_bins = int(sum(ev["e"] - ev["s"] for ev in events))
    ll_by_kernel: dict[str, "np.ndarray"] = {}
    for kernel in kernels:
        donor = param_row_for[kernel]
        hmm = hmm_dict[kernel]
        restore.apply_transition_params_from_summary_row(
            hmm=hmm, summary_row=rows_by_model[donor]
        )
        ll_by_kernel[kernel] = np.array(
            [float(hmm.log_likelihood([ev["spike_traj"]])) for ev in events], dtype=float
        )

        recorded_bins = int(rows_by_model[donor].get("n_bins", -1))
        if recorded_bins != total_bins:
            # The event set is taken from the fit, so a differing bin count means the
            # reconstruction is wrong, not that the check is inapplicable.
            raise ValueError(
                f"[{rat}/{session}/{kernel}] scored {total_bins} bins but the fit recorded "
                f"{recorded_bins}. The event set being scored is not the one this fit was "
                "trained on; check it against the fit's event_source."
            )
        if donor != kernel:
            # `rows_by_model[donor]["loglik"]` is the donor kernel's total under the donor's
            # OWN transition structure. This kernel borrows only the scalars, so there is no
            # identity to check -- asserting one would fail on every correct run.
            continue
        # Same two tiers the self-transfer check uses, and for the same reason: a genuine
        # event-set or parameter mismatch is orders of magnitude off, while restoring
        # scalars through JSON perturbs a parameter pinned at its cap in the last digits,
        # which is negligible beside the margins the filter compares.
        check_stored_loglik(
            f"{rat}/{session}/{kernel}",
            stored=float(rows_by_model[kernel]["loglik"]),
            got=float(ll_by_kernel[kernel].sum()),
            n_bins=total_bins,
            mode="per_bin",
            exc=ValueError,
        )
    return ll_by_kernel


def per_event_kernel_evidence(
    events: list[dict], hmm_dict: dict, rows_by_model: dict, *, rat: str, session: str,
    restore=None,
):
    """One row per event: every scored kernel's log-likelihood, the argmax, and the margin.

    argmax over total LL and over per-bin LL agree (same n_bins per event), so the winner is
    unambiguous. Row order matches `events`, so a caller can index the frame by the same
    event index it iterates with.

    The log-likelihood columns cover SCORED_KERNELS; `best_kernel` and the coherence margin
    cover COMPARISON_KERNELS. An off-ballot kernel therefore has its evidence on disk but can
    never win an event or shift the filter -- see OFF_BALLOT_KERNELS for why
    it is off the ballot in the first place.
    """
    import numpy as np
    import pandas as pd

    ll_by_kernel = score_events_per_kernel(
        events, hmm_dict, rows_by_model, rat=rat, session=session, restore=restore,
    )
    frame, n_bins_int = event_key_frame(events, rat=rat, session=session)
    n_bins_arr = n_bins_int.astype(float)
    best_kernel_arr = np.array(COMPARISON_KERNELS)[
        np.vstack([ll_by_kernel[k] for k in COMPARISON_KERNELS]).argmax(axis=0)
    ]
    coherent_best = np.vstack([ll_by_kernel[k] for k in COHERENT_KERNELS]).max(axis=0)
    incoherent_best = np.vstack([ll_by_kernel[k] for k in INCOHERENT_KERNELS]).max(axis=0)
    margin_arr = (coherent_best - incoherent_best) / np.maximum(1.0, n_bins_arr)
    won_by = pd.Series(best_kernel_arr).value_counts()
    print(f"  [{rat}/{session}] best kernel per event: "
          + ", ".join(f"{k}={int(won_by.get(k, 0))}" for k in COMPARISON_KERNELS))

    # Fixed schema regardless of which off-ballot arms this fit happens to carry: an unfitted
    # one leaves NaN rather than dropping a column, so every cache has the same header and a
    # reader can tell "not fitted" from "fitted and lost".
    for kernel in SCORED_KERNELS:
        lls = ll_by_kernel.get(kernel)
        if lls is None:
            frame[ll_col(kernel, "total")] = np.nan
            frame[ll_col(kernel, "per_bin")] = np.nan
            continue
        frame[ll_col(kernel, "total")] = lls
        frame[ll_col(kernel, "per_bin")] = lls / np.maximum(1, n_bins_int)
    frame["best_kernel"] = best_kernel_arr
    frame["coherent_margin_per_bin"] = margin_arr
    return frame[list(EVIDENCE_COLUMNS)]


def event_key_frame(events: list[dict], *, rat: str, session: str):
    """The EVENT_KEY columns plus n_bins, in `events` order.

    Shared by both evidence builders so the two caches are keyed identically -- they are meant
    to be joined on EVENT_KEY, and a key assembled twice is a key that can drift.
    """
    import numpy as np
    import pandas as pd

    n_bins_int = np.array([ev["e"] - ev["s"] for ev in events], dtype=int)
    frame = pd.DataFrame({
        "rat": rat, "session": session,
        "trial": [ev["trial"] for ev in events],
        "event_id": [ev["event_id"] for ev in events],
        "s": [ev["s"] for ev in events],
        "e": [ev["e"] for ev in events],
        "n_bins": n_bins_int,
    })
    return frame, n_bins_int


def per_event_direction_evidence(
    events: list[dict], hmm_dict: dict, rows_by_model: dict, *, rat: str, session: str,
):
    """One row per event: the three DIRECTION kernel log-likelihoods and their argmax.

    The ballot is TCM-SR / SR / reverse SR (DIRECTION_KERNELS), each scored at
    its own fitted parameters. No coherence margin: this
    ballot holds no incoherent kernel, and a consumer that wants the filter joins the
    full-ballot cache on EVENT_KEY instead of having a second copy of the margin here.

    Row order matches `events`, as with per_event_kernel_evidence.
    """
    import numpy as np
    import pandas as pd

    ll_by_kernel = score_events_per_kernel(
        events, hmm_dict, rows_by_model, rat=rat, session=session,
        kernels=DIRECTION_KERNELS, param_donors=PARAM_DONORS,
    )
    frame, n_bins_int = event_key_frame(events, rat=rat, session=session)
    best_arr = np.array(DIRECTION_KERNELS)[
        np.vstack([ll_by_kernel[k] for k in DIRECTION_KERNELS]).argmax(axis=0)
    ]
    won_by = pd.Series(best_arr).value_counts()
    print(f"  [{rat}/{session}] best direction kernel per event: "
          + ", ".join(f"{k}={int(won_by.get(k, 0))}" for k in DIRECTION_KERNELS))

    for kernel in DIRECTION_KERNELS:
        frame[ll_col(kernel, "total")] = ll_by_kernel[kernel]
        frame[ll_col(kernel, "per_bin")] = (
            ll_by_kernel[kernel] / np.maximum(1, n_bins_int)
        )
    frame[BEST_DIRECTION_COL] = best_arr
    return frame[list(DIRECTION_EVIDENCE_COLUMNS)]


# --------------------------------------------------------------------------- #
# Session selection and the cache write, shared by every dataset's scorer
# --------------------------------------------------------------------------- #
def filter_pairs(discovered, *, rats=None, sessions=None) -> list[tuple[str, str]]:
    """(rat, session) pairs from `discovered` kept by the --rats / --sessions filters.

    Raises SystemExit when nothing was discovered or nothing matched, naming which.
    """
    if not discovered:
        raise SystemExit("No saved fit summaries for this run tag.")
    rat_set = set(rats) if rats else None
    session_set = set(sessions) if sessions else None
    if rat_set is None and session_set is None:
        return list(discovered)
    pairs = [
        (rat, session)
        for rat, session in discovered
        if (rat_set is None or rat in rat_set) and (session_set is None or session in session_set)
    ]
    if not pairs:
        raise SystemExit("No saved fit summaries matched the requested rat/session filters.")
    return pairs


def ballot_metadata(direction: bool = False) -> dict:
    """The kernel-set fields every cache's metadata records."""
    return {
        "ballot": "direction" if direction else "comparison",
        "kernels": list(DIRECTION_KERNELS if direction else SCORED_KERNELS),
        # Which of `kernels` the argmax actually ran over. Equal to `kernels` on the direction
        # ballot; on the comparison one it is the strict subset that excludes the off-ballot
        # arms, so a reader can tell "scored" from "could have won" without re-deriving it.
        "ballot_kernels": list(DIRECTION_KERNELS if direction else COMPARISON_KERNELS),
        "off_ballot_kernels": [] if direction else list(OFF_BALLOT_KERNELS),
        # Which kernels borrowed whose fitted scalars. Empty for the comparison ballot, where
        # every kernel has a fit of its own.
        "param_donors": dict(PARAM_DONORS) if direction else {},
        "coherent_kernels": list(COHERENT_KERNELS),
        "incoherent_kernels": list(INCOHERENT_KERNELS),
    }


def write_cache(frames: list, out_dir: Path, metadata: dict, *, direction: bool = False) -> Path:
    """Concatenate per-session evidence frames and write the cache plus its metadata.

    Adds `n_events` and `won_by` (wins per ballot kernel) to `metadata`. Raises SystemExit when
    no session produced events, so an empty cache is never written.
    """
    import pandas as pd

    if not frames:
        raise SystemExit(f"No events scored for {Path(out_dir).name}; nothing written.")
    best_col = BEST_DIRECTION_COL if direction else "best_kernel"
    ballot = DIRECTION_KERNELS if direction else COMPARISON_KERNELS
    evidence = pd.concat(frames, ignore_index=True)
    won_by = evidence[best_col].value_counts()
    metadata = {
        **metadata,
        "n_events": int(len(evidence)),
        "won_by": {k: int(won_by.get(k, 0)) for k in ballot},
    }
    out_path = write_evidence(
        evidence, out_dir, metadata,
        name=names.DIRECTION_EVIDENCE_NAME if direction else names.EVIDENCE_NAME,
        meta_name=names.DIRECTION_EVIDENCE_META_NAME if direction else names.EVIDENCE_META_NAME,
    )
    print(f"\nwrote {len(evidence)} events -> {out_path}")
    print(f"  best {'direction ' if direction else ''}kernel per event: "
          + ", ".join(f"{k}={int(won_by.get(k, 0))}" for k in ballot))
    return out_path


# --------------------------------------------------------------------------- #
# Two-map datasets (Buzsaki, Pfeiffer): one fitted session -> its evidence frame
# --------------------------------------------------------------------------- #
def two_map_session_evidence(
    rat: str,
    session: str,
    *,
    fit_dir: Path,
    inputs_name: str,
    policy,
    rebuild_events,
    traditional_events_csv: Path | None = None,
):
    """Rebuild one two-map fit's event set from its own summary and score every event.

    The event source, detection setting, decode grid and traditional table are the fit's, read
    from its summary, so the scored events are the fitted ones; `score_events_per_kernel`'s
    bin-count and stored-loglik checks then hold exactly. `policy` is the dataset's
    `io.fit_summary` event-source policy. `rebuild_events(summary, inputs, event_source,
    traditional_csv)` returns the dataset's event handles `{event_id, s, e, spike_traj}` in
    fit order; `trial` is 0 for every two-map event. `traditional_events_csv` overrides the
    path a __tradreplay fit recorded (checked against its traditional run tag).

    Returns (evidence frame, provenance dict), or (None, None) when the session has no fit
    summary under `fit_dir`.
    """
    import numpy as np

    from replay_hmm.hmm.rebuild_two_map import rebuild_session_hmms
    from replay_hmm.io import fit_summary

    label = f"{rat}/{session}"
    summary_path = Path(fit_dir) / names.FIT_SUMMARY_NAME
    if not summary_path.exists():
        print(f"Skipping {label}: missing {summary_path}")
        return None, None
    summary, rows_by_model = fit_summary.load_saved_fit_summary(summary_path)
    event_source = fit_summary.fit_event_source(summary, policy=policy, label=label)
    traditional_csv = None
    if event_source == fit_summary.TRADITIONAL_SOURCE:
        traditional_csv = fit_summary.resolve_traditional_events_csv(
            summary, traditional_events_csv, rat=rat, session=session
        )
    elif traditional_events_csv is not None:
        raise ValueError(
            f"[{label}] --traditional-events-csv was given but this fit records "
            f"event_source={event_source!r}, so its event set does not come from a "
            "traditional decode. Drop the flag, or point --run-tag at a __tradreplay fit."
        )
    with np.load(Path(fit_dir) / inputs_name, allow_pickle=False) as z:
        inputs = {k: z[k] for k in z.files}

    kernels = tuple(k for k in SCORED_KERNELS if k in rows_by_model)
    hmm_dict = rebuild_session_hmms(summary, inputs, kernels)
    events = [
        {**h, "trial": 0}
        for h in rebuild_events(summary, inputs, event_source, traditional_csv)
    ]
    print(f"  [{label}] events for scoring: {len(events)} ({event_source})")
    provenance = {
        "decode_config": {
            "decode_tbin_s": float(summary["decode_tbin_s"]),
            "decode_tbin_step_s": float(summary["decode_tbin_step_s"]),
            "hmm_bin_width_cm": summary.get("hmm_bin_width_cm"),
            "hmm_bin_width_m": summary.get("hmm_bin_width_m"),
            "event_detection_source": summary.get("event_detection_source"),
            "pbe_provenance": summary.get("pbe_provenance"),
            "traj_provenance": summary.get("traj_provenance"),
            "event_source_provenance": summary.get("event_source_provenance"),
        },
        "event_source": event_source,
        "traditional_events_csv": None if traditional_csv is None else str(traditional_csv),
        "traditional_run_tag": summary.get("traditional_run_tag"),
        "traditional_statuses": summary.get("traditional_statuses"),
    }
    if not events:
        return None, provenance
    frame = per_event_kernel_evidence(events, hmm_dict, rows_by_model, rat=rat, session=session)
    return frame, provenance


def two_map_parser(description: str, default_results_root: Path):
    """The CLI shared by the two-map scorers."""
    import argparse

    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--run-tag", required=True,
                        help="Fit run directory to score, e.g. "
                             "dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens. The event set, "
                             "detection setting and decode grid are read from each fit.")
    parser.add_argument("--rats", nargs="+", default=None)
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--results-root", type=Path, default=default_results_root)
    parser.add_argument("--traditional-events-csv", type=Path, default=None,
                        help="traditional_decode_events.csv for fits whose event_source is "
                             "'traditional' (the __tradreplay tags), when the path the fit "
                             "recorded does not exist here. Must come from the same "
                             "traditional run tag the fit recorded.")
    parser.add_argument("--force", action="store_true",
                        help="Rescore even if the evidence CSV already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the run plan without loading sessions.")
    return parser


def two_map_main(args, *, dataset: str, discover, session_evidence) -> int:
    """Score every discovered (rat, session) of one run tag and write its cache.

    `discover(results_root, run_tag)` lists the fitted (rat, session) pairs;
    `session_evidence(rat, session, args)` returns `two_map_session_evidence`'s pair.
    """
    from replay_hmm.io.evidence import cache_dir

    pairs = filter_pairs(
        discover(Path(args.results_root), args.run_tag), rats=args.rats, sessions=args.sessions
    )
    out_dir = cache_dir(Path(args.results_root), args.run_tag)
    out_path = out_dir / names.EVIDENCE_NAME
    print(f"Event kernel evidence run plan ({dataset})")
    print(f"  fit_run_tag: {args.run_tag}")
    print(f"  kernels: {list(SCORED_KERNELS)}")
    print(f"  sessions ({len(pairs)}): {pairs}")
    print(f"  output: {out_path}")
    if args.dry_run:
        return 0
    if out_path.exists() and not args.force:
        print(f"  {out_path} exists; pass --force to rescore.")
        return 0

    frames, provenance = [], None
    for rat, session in pairs:
        frame, prov = session_evidence(rat, session, args)
        if provenance is None and prov is not None:
            provenance = prov
        if frame is not None and len(frame):
            frames.append(frame)
    metadata = {
        "dataset": dataset,
        "fit_run_tag": args.run_tag,
        "decode_tag": args.run_tag,
        **ballot_metadata(),
        "event_source": (provenance or {}).get("event_source"),
        "traditional_events_csv": (provenance or {}).get("traditional_events_csv"),
        "traditional_run_tag": (provenance or {}).get("traditional_run_tag"),
        "traditional_statuses": (provenance or {}).get("traditional_statuses"),
        "decode_config": (provenance or {}).get("decode_config"),
        "rat_session_pairs": [list(p) for p in pairs],
    }
    write_cache(frames, out_dir, metadata)
    return 0
