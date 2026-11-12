"""The Lustig event sets: the mb_traj map-level gates and per-segment spike-count collector, and
the decode-mask windows (`collect_event_sequences_for_fit`) the --all-pbes and --running arms
fit on. `fit_mb_traj_arm` reads a saved fit's one-arm restriction back; `collect_fit_event_set`
rebuilds a saved mb_traj or all-PBE fit's event set."""
from __future__ import annotations

import numpy as np

from replay_hmm.datasets.lustig import labels as lml
from replay_hmm.datasets.lustig.bundle import SessionBundle
from replay_hmm.events import runs
from replay_hmm.io import fit_summary

# MB trajectory segments -> event arrays for HMM scoring
def mb_traj_arm_slots(arm_only) -> tuple[int, ...] | None:
    """Stored map/slot indices a segment must be decoded on, for one physical maze arm.

    None in, None out: "unrestricted" and "all six slots happen to be allowed" must stay
    distinguishable, because the wheel exclusion is a SEPARATE gate and a caller handed
    (0, 1, 2, 3, 4, 5) could not tell which of the two it was being told.

    Single-sourced on `lml.slots_for_arm`; nothing here restates the arm<->slot table.
    """
    if arm_only is None:
        return None
    if isinstance(arm_only, bool):
        # bool is a subclass of int, so True would reach ARM_ID_TO_SLOTS[1] and silently
        # resolve to the LEFT arm (e.g. if this were wired to a `store_true` flag).
        raise TypeError(
            f"arm_only must name an arm (L/C/R, a name, or an armID), got {arm_only!r}; a "
            "boolean cannot select one."
        )
    return tuple(int(s) for s in lml.slots_for_arm(arm_only))


def mb_traj_arm_letter(arm_only) -> str | None:
    """Canonical "L"/"C"/"R" for any spelling `mb_traj_arm_slots` accepts; None in, None out.

    Derived from the RESOLVED slots via `lml.PHYS_ARM` rather than by string-munging the input,
    so it accepts exactly what the collector accepts -- including the integer armID 2, which
    `arm_only_arg` rejects on purpose because the CLI string "2" is ambiguous between armID 2
    (Center) and stored slot 2 (C_in). That distinction belongs to the CLI, not here.
    """
    slots = mb_traj_arm_slots(arm_only)
    if slots is None:
        return None
    return lml.PHYS_ARM[slots[0]]


def mb_traj_map_included(m, *, wheel_map, include_wheel, arm_slots) -> bool:
    """Whether map index `m`'s trajectory segments belong in the mb_traj event set.

    Shared, through `iter_mb_traj_spans`, by `collect_mb_traj_event_datas` (what the fitter
    trains on) and `datasets.lustig.kernel_share.collect_events_with_metadata` (what the scorer
    re-scores). The two must select the same events bin for bin;
    `score_events_per_kernel`'s bin-count identity checks that they do.

    An arm restriction drops the wheel whatever `include_wheel` says: `mb_traj_arm_slots`
    never returns the wheel slot.
    """
    if (not include_wheel) and wheel_map is not None and m == wheel_map:
        return False
    if arm_slots is not None and int(m) not in arm_slots:
        return False
    return True


def iter_mb_traj_spans(
    segments_by_event,
    *,
    wheel_map,
    include_wheel,
    arm_slots,
    min_len=2,
    dedupe_across_maps=True,
    sort=False,
    int_event_key=False,
):
    """Yield ``(event_id, map_idx, s, e)`` for each mb_traj span of one lap, deduped.

    The mb_traj span loop, shared by the fitter's collector
    (`collect_mb_traj_event_datas`), the scorer's (`datasets.lustig.kernel_share`), the MB jump
    pipeline (`mb_jump._compute_trial_metrics`), the traditional-window extractor
    (`traditional_jumps._jumps_from_traj_segments`) and the event viewer. What each does with
    a span stays at the call site.

    `segments_by_event` is one lap's ``event_id -> map_idx -> [(s, e), ...]``. Maps go through
    `mb_traj_map_included` before the dedupe, so an excluded map never claims a span that an
    included map also holds. `s` and `e` are cast to int; spans with ``e - s < min_len`` are
    skipped.
    The dedupe key is ``(event_id, s, e)``, or ``(event_id, map_idx, s, e)`` with
    `dedupe_across_maps=False`; at any mass threshold > 0.5 only one map can qualify, so it is
    a no-op in practice. `int_event_key=True` puts ``int(event_id)`` in the key (the
    traditional extractor's); the yielded `event_id` is always the key as given.
    `sort=True` walks events and maps in sorted order (the jump pipelines'), otherwise in
    dict order (the collectors').
    """
    seen = set()
    events = sorted(segments_by_event.items()) if sort else segments_by_event.items()
    for ev_id, ev_maps in events:
        maps = sorted(ev_maps.items()) if sort else ev_maps.items()
        for m, segs in maps:
            if not mb_traj_map_included(
                m, wheel_map=wheel_map, include_wheel=include_wheel, arm_slots=arm_slots
            ):
                continue
            for s, e in segs:
                s, e = int(s), int(e)
                if e - s < min_len:
                    continue
                ev_key = int(ev_id) if int_event_key else ev_id
                key = (ev_key, s, e) if dedupe_across_maps else (ev_key, m, s, e)
                if key in seen:
                    continue
                seen.add(key)
                yield ev_id, m, s, e


def collect_mb_traj_event_datas(
    session_obj, *, min_len=2, include_wheel=False, dedupe_across_maps=True, arm_only=None
):
    """
    Collect decoded MB-trajectory event spike-count arrays from one session.
    Iterates over all trials and decoded PBE events in ``session_obj``, extracts
    each trajectory segment ``[s:e)`` from the per-trial spike-count matrix, and
    returns it as an event array of shape ``(T_event, n_units)`` suitable for HMM
    fitting/scoring.

    Parameters
    ----------
    session_obj : SessionBundle
        Session container with at least:
        - ``session_obj.trial_ids`` : list[int]
        - ``session_obj.trial_rate_trajs`` : dict[int, np.ndarray]
          Maps ``trial_id -> rate_traj`` with
          ``rate_traj.shape == (n_units, T_trial)``.
        - ``session_obj.traj_segments_by_pbe`` : dict[int, dict[int, dict[int, list[tuple[int, int]]]]]
          Maps
          ``trial_id -> event_id -> map_id -> list[(s, e)]``,
          where each segment uses decode-bin indices with
          ``0 <= s < e <= T_trial``.
        - ``session_obj.lin_map_bounds`` : np.ndarray of shape ``(n_maps + 1,)``
        - ``session_obj.include_wheel`` : bool
    min_len : int, default=2
        Minimum allowed event length in decode bins. Segments with
        ``(e - s) < min_len`` are skipped.
    include_wheel : bool, default=False
        Whether to include segments from the wheel map. If False and the session
        includes a wheel map, segments from that final map are excluded.
    dedupe_across_maps : bool, default=True
        Controls deduplication of segments within each trial.
        - If True, deduplicate by ``(event_id, s, e)``, so identical time ranges
          appearing under multiple maps are kept only once.
        - If False, deduplicate by ``(event_id, map_id, s, e)``, so the same
          ``(s, e)`` interval may appear multiple times if assigned to different maps.
    arm_only : {None, "L", "C", "R"} or armID or arm name, default=None
        Restrict the event set to segments decoded on ONE physical maze arm, i.e. to that
        arm's two directional slots (``"C"`` -> ``C_in``/``C_out`` = stored slots 2, 3).
        None keeps every non-wheel map. The restriction is on each segment's DECODED map --
        no behaviour or ``mazeSect`` is read -- because ``build_session_bundle`` already cut
        each segment where one map block held >= ``cfg.trajectory_mass_threshold`` of the
        posterior. At any threshold > 0.5 that membership is exclusive, so the three arms
        PARTITION the unrestricted event set.
    Returns
    -------
    datas : list[np.ndarray]
        List of event arrays. Each element has shape ``(T_event, n_units)``, where:
        - ``T_event = e - s`` for that extracted segment
        - ``n_units`` is the number of decoded units for the session
        Each array is ``dtype=np.int32`` and is formed from:
        ``rate_traj[:, s:e].T``.
    
    Notes
    -----
    For each trial:
    - ``rate_traj`` has shape ``(n_units, T_trial)``
    - slicing ``rate_traj[:, s:e]`` gives shape ``(n_units, T_event)``
    - transposing gives ``(T_event, n_units)``, which is the returned event format
    The returned list may contain variable-length events, so ``T_event`` can differ
    across entries.
    """    
    datas = []
    n_maps = len(session_obj.lin_map_bounds) - 1
    wheel_map = n_maps - 1 if session_obj.include_wheel else None
    # Resolved once, outside the trial loop, so a bad arm spelling raises once here rather
    # than per trial from inside the hot path.
    arm_slots = mb_traj_arm_slots(arm_only)

    for tr in session_obj.trial_ids:
        rate_traj = np.asarray(session_obj.trial_rate_trajs[tr])  # (n_units, T)
        by_event = session_obj.traj_segments_by_pbe[tr]           # ev -> map -> [(s,e)]

        for _ev_id, _m, s, e in iter_mb_traj_spans(
            by_event,
            wheel_map=wheel_map,
            include_wheel=include_wheel,
            arm_slots=arm_slots,
            min_len=min_len,
            dedupe_across_maps=dedupe_across_maps,
        ):
            datas.append(rate_traj[:, s:e].T.astype(np.int32))  # (T_event, n_units) spike counts

    return datas


WINDOW_SOURCES = ("pbe_nonrun", "pbe", "run")


def collect_event_sequences_for_fit(
    session_obj: SessionBundle,
    *,
    window_source: str = "pbe_nonrun",
) -> list[np.ndarray]:
    """
    Extract event-aligned spike-count sequences for transition fitting.

    Each element should be shaped (T_event, n_units), matching `ssm.HMM` expected input.
    Event windows should match the same decode-bin logic used for trajectory calling.

    `window_source` selects the decode-timebase mask, one contiguous run of which becomes
    one sequence:

    - ``"pbe_nonrun"``: PBE AND eligible/immobile (`trial_pbe_nonrun_masks`) -- the replay arm.
    - ``"pbe"``: PBE windows regardless of speed (`trial_pbe_masks`).
    - ``"run"``: running on the track, wheel excluded (`trial_run_masks`) -- one sequence per
      maximal running bout within a lap, for the `--running` fitting arm.

    Returns:
        List of per-event count matrices used for model fitting/grid search.
    """
    if window_source not in WINDOW_SOURCES:
        raise ValueError(f"window_source={window_source!r} not in {WINDOW_SOURCES}")
    mask_name = {
        "pbe_nonrun": "trial_pbe_nonrun_masks",
        "pbe": "trial_pbe_masks",
        "run": "trial_run_masks",
    }[window_source]
    mask_by_trial = getattr(session_obj, mask_name, None)

    if mask_by_trial is None:
        raise ValueError(
            f"SessionBundle is missing `{mask_name}`. "
            "Rebuild bundles with decode-bin PBE masks retained."
        )

    event_datas: list[np.ndarray] = []
    for tr in session_obj.trial_ids:
        rate_traj = np.asarray(session_obj.trial_rate_trajs[tr])  # (n_units, T_trial)
        trial_mask = np.asarray(mask_by_trial[tr], dtype=bool)    # (T_trial,)

        if trial_mask.shape[0] != rate_traj.shape[1]:
            raise ValueError(
                f"Mask length mismatch in trial {tr}: "
                f"{trial_mask.shape[0]} vs {rate_traj.shape[1]}"
            )

        for s, e in runs.runs_from_mask(trial_mask, as_int=True, cast_bool=True):
            if e - s < 2:
                continue
            # int32 to match collect_mb_traj_event_datas. rate_traj is uint16, and any
            # signed arithmetic inside ssm on the observation array would wrap.
            event_datas.append(rate_traj[:, s:e].T.astype(np.int32))  # (T_event, n_units)

    return event_datas


# The event sets `collect_fit_event_set` rebuilds. Traditional events need the decode table
# they were accepted from (`traditional_io`).
FIT_EVENT_SET_SOURCES = ("mb_traj", "all_pbe")


def collect_fit_event_set(session_obj, event_source: str, *, arm_only=None) -> list[np.ndarray]:
    """A saved fit's event set, rebuilt from its session bundle: whole PBE windows as int32
    (`event_source="all_pbe"`; `arm_only` does not apply) or the MB-called trajectory segments
    of every arm or of `arm_only`'s (`"mb_traj"`), with the fitter's min_len=2, no wheel and
    deduplication across maps. The jump distribution and cross-session transfer collect their
    events with it."""
    if event_source not in FIT_EVENT_SET_SOURCES:
        raise ValueError(
            f"event_source must be one of {FIT_EVENT_SET_SOURCES}; got {event_source!r}."
        )
    if event_source == "all_pbe":
        return [
            d.astype(np.int32, copy=False)
            for d in collect_event_sequences_for_fit(
                session_obj=session_obj, window_source="pbe_nonrun"
            )
        ]
    return collect_mb_traj_event_datas(
        session_obj=session_obj,
        min_len=2,
        include_wheel=False,
        dedupe_across_maps=True,
        arm_only=arm_only,
    )


# The three spellings a run tag and a summary may carry. Not a restatement of
# datasets.lustig.labels: `mb_traj_arm_slots` is what resolves an arm to slots, and this tuple only
# names which letters a saved summary is allowed to hold.
MB_TRAJ_ARMS: tuple[str, ...] = ("L", "C", "R")


def fit_mb_traj_arm(summary_saved: dict) -> str | None:
    """Which single maze arm a saved mb_traj fit was restricted to, from the fit's own summary.

    Same contract as `fit_event_source`: the FIT is authoritative and there is deliberately no
    CLI flag to override it. An arm restriction is a property of the event set a fit was
    trained on, and unlike the traditional arm's CSV path, nothing external is needed to
    reproduce it.

    A summary with no key resolves to None (unrestricted). That default is safe rather than a
    guess: a restricted event set is a strict SUBSET, so a restricted fit read as unrestricted
    fails `score_events_per_kernel`'s bin-count identity loudly instead of being scored quietly.

    The arm is its own summary key rather than a fourth `event_source` value because it is
    orthogonal to which kind of event set was fitted.
    """
    arm = summary_saved.get("mb_traj_arm_only")
    if arm is None:
        return None
    source = fit_summary.fit_event_source(summary_saved, policy=fit_summary.ARM_GATE)
    if source != "mb_traj":
        raise ValueError(
            f"saved summary records mb_traj_arm_only={arm!r} with event_source={source!r}; "
            "an arm restriction applies only to MB-called trajectory segments."
        )
    key = str(arm).strip().upper()
    if key not in MB_TRAJ_ARMS:
        raise ValueError(
            f"unrecognised mb_traj_arm_only {arm!r} in the saved fit summary; expected one of "
            f"{list(MB_TRAJ_ARMS)} or null."
        )
    recorded = summary_saved.get("mb_traj_arm_slots")
    if recorded is not None:
        expected = list(mb_traj_arm_slots(key))
        if [int(s) for s in recorded] != expected:
            # The fit named an arm AND the slots it actually used. If the two no longer agree,
            # the arm<->slot table changed under the fit; refusing is the only safe answer,
            # because scoring would silently substitute a different arm's segments.
            raise ValueError(
                f"saved summary records mb_traj_arm_only={key!r} with "
                f"mb_traj_arm_slots={list(recorded)}, but this build's label table gives "
                f"{expected}; the arm<->slot table changed since the fit was written."
            )
    return key
