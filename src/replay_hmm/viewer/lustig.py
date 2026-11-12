"""Interactive per-event browser for Lustig HMM decodes.

Time-aligned panels for one PBE at a time:

    z-PFR       z-scored population spike density, the trace PBEs were called from
    posterior   HMM smoothed posterior (forward-backward) over the full 7-map stack, one
                row per kernel, with the Viterbi path overlaid (and, with
                --highlight-mb-traj, the called trajectory segments shaded)
    raster      spike raster of the decode units, sorted by field peak on the
                event's dominant map

Beside them: an info block, a zoom square per kernel on its dominant map and, with a single
kernel only, this event's Viterbi jumps over a faint session distribution.

Controls: left/right arrows, n/p or the Previous/Next buttons step events, the slider seeks,
`s` or the Save button saves the current figure.

The decode grid is chosen per invocation (`--decode-tbin-s` / `--decode-tbin-step-s`),
so comparing 0.003/0.003 against 0.010/0.010 means running this twice.

Everything is rebuilt from the saved fit's own `decode_config`, never from the CLI
flags: the flags select WHICH fit to load, the fit decides HOW the bundle is built.

`--jump-scope traditional` browses the traditional-replay events instead of the bundle's PBEs:
the windows a __tradreplay fit trained on, on this session's lap grid, each scored whole. On a
__tradreplay fit the records are checked against the fit (event count, bin count, summed
log-likelihood per kernel); the windows are the classifier detector's, so no MB trajectory
segments are drawn on them.

Run from the repo root:

    replay-hmm viewer --dataset lustig --rat Huey --session rec01 --run-tag <tag> ...
    replay-hmm viewer --dataset lustig --rat TinyTim --session rec01 \\
        --run-tag dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67__tradreplay --jump-scope traditional
    replay-hmm viewer --dataset lustig --list-run-tags
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from functools import cached_property
from pathlib import Path

import numpy as np

from replay_hmm.viewer.engine import (
    JUMP_SCOPE_BY_EVENT_SOURCE,
    WHOLE_WINDOW_SCOPES,
    EventRecord,
    ViewerSource,
    DensityTrace,
    add_viewer_args,
    fit_jump_scope,
    resolve_traditional_table,
    validate_kernel_args,
    warn_if_scope_mismatch,
    warn_if_sigma_capped,
    run_viewer,
)
from replay_hmm.viewer.input_cache import ViewerInputs, lfp_span, load_or_build_inputs
from replay_hmm import paths
from replay_hmm.datasets.lustig import kernels as lustig_kernels
from replay_hmm.datasets.lustig import movement as lustig_movement
from replay_hmm.hmm import restore
from replay_hmm.io import fit_summary
from replay_hmm.io.fit_tree import print_run_tags, resolve_fit_dir
from replay_hmm.datasets.lustig import labels as lml
from replay_hmm.datasets.lustig import calibration
from replay_hmm.datasets.lustig import constants as lustig_constants
from replay_hmm.datasets.lustig import event_sets
from replay_hmm.datasets.lustig import traditional_io
from replay_hmm.decode import grid as decode_grid
from replay_hmm.events import runs as events_runs
from replay_hmm.decode import config
from replay_hmm.io.jump_store import load_saved_jump_distributions
from replay_hmm.jumps.signed import signed_jumps_from_state_sequence
from replay_hmm.io import names

DEFAULT_RESULTS_ROOT = paths.LUSTIG_FITS_DIR
DEFAULT_FIGURES_ROOT = paths.FIGURES_ROOT

# Physical map order, wheel last -- see datasets.lustig.labels for the evidence behind these names.
MAP_LABELS_PHYSICAL = lml.MAP_LABELS_PROSE


# --------------------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------------------

def event_traj_segments(session_obj, trial: int, event_idx: int) -> list[tuple[int, int, int]]:
    """Called trajectory segments for one event as `[(map, s, e), ...]`.

    Reproduces `event_sets.collect_mb_traj_event_datas(min_len=2, include_wheel=False,
    dedupe_across_maps=True)` restricted to a single event, and in the same order -- so
    concatenating this over every event yields exactly the event list the jump pipeline
    harvests, and the overlay drawn from it is the trace that generated the jumps.
    """
    n_maps = len(session_obj.lin_map_bounds) - 1
    wheel_map = n_maps - 1 if session_obj.include_wheel else None
    out: list[tuple[int, int, int]] = []
    # The fitter's shared mb_traj span loop over this one event, so this copy cannot drift
    # from the fitter's; with one event its (event, s, e) dedupe key is (s, e). The viewer
    # shows every arm whatever a fit was restricted to, hence arm_slots=None.
    for _ev, m, s, e in event_sets.iter_mb_traj_spans(
        {event_idx: session_obj.traj_segments_by_pbe[trial][event_idx]},
        wheel_map=wheel_map,
        include_wheel=False,
        arm_slots=None,
    ):
        out.append((int(m), s, e))
    return out


# --------------------------------------------------------------------------------------
# jumps
# --------------------------------------------------------------------------------------

def event_spike_arrays(session_obj, rec: EventRecord, scope: str) -> list[np.ndarray]:
    """The spike-count arrays this event contributes, per `--jump-scope`.

    `full-pbe` gives one array for the whole PBE-and-nonrun window, matching
    `event_sets.collect_event_sequences_for_fit`; `traditional` the whole traditional event
    window, matching `traditional_io.collect_traditional_event_datas`. `traj` gives one array
    per called trajectory segment, matching `event_sets.collect_mb_traj_event_datas` -- the
    pipeline's default, where Viterbi is run over each segment independently rather than over
    the whole event.
    """
    rate_traj = session_obj.trial_rate_trajs[rec.trial]
    if scope in WHOLE_WINDOW_SCOPES:
        return [rate_traj[:, rec.s_bin:rec.e_bin].T.astype(np.int32)]
    return [
        rate_traj[:, s:e].T.astype(np.int32)
        for _m, s, e in event_traj_segments(session_obj, rec.trial, rec.event_idx)
    ]


def load_backdrop(
    *, results_root: Path, run_tag: str, kernel: str, rat: str, session: str, scope: str,
) -> np.ndarray:
    """Session jump distribution from a previous `replay-hmm jumps --dataset lustig` run.

    The run's own directory, or `<run_tag>__tradreplay_jumps` (`--tradreplay-jumps`); the one
    whose recorded scope matches `scope` wins.
    """
    base = Path(results_root) / names.JUMP_DIST_DIRNAME
    found = [
        base / tag for tag in (run_tag, run_tag + "__tradreplay_jumps")
        if (base / tag / names.JUMP_DIST_NAME).exists()
    ]

    def _saved_scope(d: Path):
        meta_path = d / names.JUMP_META_NAME
        return (JUMP_SCOPE_BY_EVENT_SOURCE.get(json.loads(meta_path.read_text()).get("jump_event_source"))
                if meta_path.exists() else None)

    matched = [d for d in found if _saved_scope(d) == scope]
    results_dir = (matched or found or [base / run_tag])[0]
    filepath = results_dir / names.JUMP_DIST_NAME
    if not filepath.exists():
        raise SystemExit(
            f"--session-backdrop saved needs {filepath}, which does not exist.\n"
            f"  Produce it with:  replay-hmm jumps --dataset lustig --rats {rat} "
            f"--sessions {session} --run-tag {run_tag}\n"
            "  or pass --session-backdrop compute (decodes every event at startup) "
            "or --session-backdrop none."
        )

    # The saved harvest has its own scope, recorded in the metadata. The run tag does not
    # tell you: a tag ending __allpbe can hold a trajectory-scope harvest, since the fit axis
    # and the jump axis are separate flags.
    #
    # `jump_event_source` is read in preference to the `all_pbe_jumps` boolean because it
    # distinguishes all three event sets; the boolean alone cannot tell a traditional harvest
    # from a trajectory one. A traditional harvest matches --jump-scope traditional, whose
    # records are the same table's events on the same grid.
    meta_path = results_dir / names.JUMP_META_NAME
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        saved_scope = JUMP_SCOPE_BY_EVENT_SOURCE.get(meta.get("jump_event_source"))
        if saved_scope is None:
            saved_scope = "full-pbe" if meta.get("all_pbe_jumps") else "traj"
        if saved_scope != scope:
            warnings.warn(
                f"saved backdrop in {results_dir.name} was harvested at scope "
                f"{saved_scope!r} (jump_event_source="
                f"{meta.get('jump_event_source')!r}) but --jump-scope is {scope!r}. The "
                "backdrop and the per-event histogram are not directly comparable; use "
                "--session-backdrop compute for a matched one.",
                RuntimeWarning,
                stacklevel=2,
            )

    _by_rat, by_session = load_saved_jump_distributions(
        results_dir=results_dir,
        kernels=[kernel],
        session_pairs=[(rat, session)],
    )
    jumps = by_session.get((kernel, rat, session))
    if jumps is None:
        raise SystemExit(
            f"{filepath} has no entry for {kernel}__{rat}__{session}. "
            "Pass --session-backdrop compute or none."
        )
    return np.asarray(jumps, dtype=int)


# --------------------------------------------------------------------------------------
# dataset hook
# --------------------------------------------------------------------------------------

class LustigSource(ViewerSource):
    """`ViewerSource` over one Lustig session: lap-local records over `trial_rate_trajs`."""

    map_labels = MAP_LABELS_PHYSICAL

    def __init__(self, inputs: ViewerInputs, *, data_root: Path | None = None,
                 traditional: bool = False) -> None:
        s = inputs.session_obj
        self.inputs = inputs
        # Records are traditional events: their windows are the classifier detector's, with
        # no entry in the bundle's traj_segments_by_pbe.
        self.traditional = bool(traditional)
        self.session_obj = s
        self.cfg = s.cfg
        self.data_root = data_root
        self.name = f"{s.rat}/{s.session}"
        self.name_spaced = f"{s.rat} / {s.session}"
        self.decode_tbin_s = self.cfg.decode_tbin_s
        self.decode_tbin_step_s = self.cfg.decode_tbin_step_s
        self.bounds = s.lin_map_bounds
        self.lin_map_concat = s.lin_map_concat
        # The six arm slots; the wheel can never be the dominant map.
        self.arm_maps = tuple(range(min(6, len(s.lin_map_bounds) - 1)))
        self.max_map_bins = int(max(s.C_bins, s.L_bins, s.R_bins))

    @cached_property
    def cm_per_bin(self) -> float:
        return calibration.bin_cm_from_bundle(self.session_obj, data_root=self.data_root)

    @cached_property
    def bin_width_px(self) -> int:
        # Resolved, not cfg.bin_width_px: under a bin_width_cm target the two differ (10 px vs
        # 8 for Spot/Huey/Stinky at 3.5 cm), and a bundle rebuilt from the cache leaves the
        # field at its 0 default.
        s = self.session_obj
        return int(
            getattr(s, "bin_width_px_resolved", 0)
            or calibration.resolve_bin_width_px(self.cfg, s.rat, s.session, self.data_root)
        )

    def n_bins_total(self, rec: EventRecord) -> int:
        return self.session_obj.trial_rate_trajs[rec.trial].shape[1]

    def counts(self, rec: EventRecord, lo: int, hi: int) -> np.ndarray:
        return self.session_obj.trial_rate_trajs[rec.trial][:, lo:hi].T.astype(np.int32)

    def bin_centers_s(self, rec: EventRecord, lo: int, hi: int) -> np.ndarray:
        _s, _e, centers = self.inputs.grids[rec.trial]
        return centers[lo:hi] / decode_grid.SPK_SAMP_RATE

    def traj_segments(self, rec: EventRecord) -> list[tuple[int, int, int]]:
        if self.traditional:
            return []
        return event_traj_segments(self.session_obj, rec.trial, rec.event_idx)

    def scope_arrays(self, rec: EventRecord, scope: str) -> list[np.ndarray]:
        return event_spike_arrays(self.session_obj, rec, scope)

    def signed_jumps(self, z: np.ndarray, bounds) -> np.ndarray:
        return signed_jumps_from_state_sequence(z, bounds)

    def event_label(self, rec: EventRecord) -> str:
        if self.traditional:
            return f"lap {rec.trial} trad #{rec.event_idx}"
        return f"lap {rec.trial} #{rec.event_idx}"

    def event_file_tag(self, rec: EventRecord) -> str:
        if self.traditional:
            return f"lap{rec.trial}_trad{rec.event_idx}"
        return f"lap{rec.trial}_ev{rec.event_idx}"

    def density(self, rec: EventRecord, t_lo_ms: float, t_hi_ms: float) -> DensityTrace:
        ts = int(self.inputs.arrays.trial_starts[rec.trial - 1])
        z_lap = self.inputs.z_by_trial[rec.trial]
        elig = self.inputs.eligible_by_trial[rec.trial]

        j_lo, j_hi = lfp_span(
            z_lap.size, ts, rec.t0_s + t_lo_ms / 1000.0, rec.t0_s + t_hi_ms / 1000.0,
        )
        j = np.arange(j_lo, j_hi)
        t_ms = ((ts + j) / decode_grid.LFP_SAMP_RATE - rec.t0_s) * 1000.0
        # PBE-ineligible time: no event may start here, whatever the density does.
        spans = []
        if j.size:
            ineligible = ~elig[j_lo:min(j_hi, elig.size)]
            spans = [(t_ms[a], t_ms[min(b, t_ms.size - 1)]) for a, b in events_runs.runs_from_mask(ineligible)]
        return DensityTrace(
            t_ms=t_ms, values=z_lap[j_lo:j_hi],
            thresholds=(self.cfg.pbe_spike_density_stdev, self.cfg.pbe_end_stdev),
            ineligible_ms=spans,
        )

    def raster_ms(self, row: int, rec: EventRecord, t_lo_ms: float, t_hi_ms: float) -> np.ndarray:
        spikes = self.inputs.spk_dict20k[int(row)]
        lo_20k = (rec.t0_s + t_lo_ms / 1000.0) * decode_grid.SPK_SAMP_RATE
        hi_20k = (rec.t0_s + t_hi_ms / 1000.0) * decode_grid.SPK_SAMP_RATE
        a = np.searchsorted(spikes, lo_20k, side="left")
        b = np.searchsorted(spikes, hi_20k, side="right")
        return (spikes[a:b] / decode_grid.SPK_SAMP_RATE - rec.t0_s) * 1000.0

    def position_lines(self, rec: EventRecord, t_lo_ms: float, t_hi_ms: float):
        """The rat's actual position, in BOTH slots of the arm it is on.

        A physical location corresponds to two states -- the inbound and the outbound slot of
        that arm -- and during a PBE the rat is immobile by construction, so the travel
        direction flags that would pick one are unset for most samples, so both are drawn.

        On the wheel there is only one slot, and the trace's own bin (time since wheel-run
        onset) is used. The returned description says where the rat was, so an absent line
        reads as "unassigned".

        "On the wheel" is `mazeSect in WHEEL_SECTIONS`, a POSITION test, not
        `trace.wheel_mask`, which comes from `get_wheel_mask(method="epochIndices")` and
        tracks wheel ROTATION: the wheel keeps turning after the rat dismounts.
        """
        trace = self.inputs.state_trace
        if trace is None:
            return [], "not loaded"
        bounds = np.asarray(self.bounds, dtype=int)

        lo = max(0, int(np.floor((rec.t0_s + t_lo_ms / 1000.0) * decode_grid.LFP_SAMP_RATE)))
        hi = min(trace.n, int(np.ceil((rec.t0_s + t_hi_ms / 1000.0) * decode_grid.LFP_SAMP_RATE)) + 1)
        if hi <= lo:
            return [], "n/a"
        idx = np.arange(lo, hi)
        t_ms = (idx / decode_grid.LFP_SAMP_RATE - rec.t0_s) * 1000.0
        arm = trace.arm_id[lo:hi]
        bin_local = np.floor(
            trace.arm_dist_px[lo:hi] / float(self.bin_width_px)
        ).astype(int)
        on_wheel = np.isin(self.inputs.arrays.maze_section[lo:hi], lustig_constants.WHEEL_SECTIONS)
        state = trace.state_idx[lo:hi]
        # The wheel axis is time since wheel-run onset, which only exists inside a bout the
        # trace actually binned, so the wheel line needs both tests to agree.
        wheel_binned = on_wheel & (trace.seg_id[lo:hi] == 6)

        lines = []
        for arm_id in (1, 2, 3):
            for slot in lustig_movement.slots_for_arm(arm_id) or ():
                n_bins = int(bounds[slot + 1] - bounds[slot])
                lines.append((t_ms, np.where(
                    (arm == arm_id) & ~on_wheel,
                    bounds[slot] + np.clip(bin_local, 0, n_bins - 1),
                    np.nan,
                ).astype(float)))
        # Wheel: one slot, and the bin is time since wheel-run onset, already in the trace.
        lines.append((t_ms, np.where(wheel_binned & (state >= 0), state, np.nan).astype(float)))

        # Describe the dominant location so a missing line is interpretable.
        if on_wheel.mean() > 0.5:
            return lines, "wheel"
        assigned = arm[~on_wheel]
        assigned = assigned[assigned > 0]
        if assigned.size == 0:
            return lines, "unassigned"
        top = int(np.bincount(assigned).argmax())
        frac = float((assigned == top).mean())
        return lines, f"{ {1: 'left', 2: 'center', 3: 'right'}[top] } arm ({frac:.0%})"


def saved_event_source(summary: dict) -> str | None:
    """The fit's training set as recorded: `event_source`, else the legacy `use_all_pbes`.

    None when the summary records neither, and then no scope warning is raised.
    """
    source = summary.get("event_source")
    if source is None and summary.get("use_all_pbes") is not None:
        source = "all_pbe" if summary["use_all_pbes"] else "mb_traj"
    return source


def check_traditional_records(summary: dict, rows_by_model: dict, hmm_dict: dict, inputs,
                              kernels, *, rat: str, session: str) -> None:
    """The traditional records are the __tradreplay fit's own event set.

    The event count against the fit's `traditional_event_stats`, then
    `kernel_share.score_events_per_kernel`: the total bin count exactly, and each kernel's summed
    log-likelihood against the one it stored. Leaves the HMMs carrying the last-scored row,
    so the caller applies each kernel's row afterwards.
    """
    from replay_hmm.analyses import kernel_share

    stats = summary.get("traditional_event_stats") or {}
    n_saved = stats.get("n_used")
    if n_saved is not None and int(n_saved) != len(inputs.records):
        raise SystemExit(
            f"[{rat}/{session}] rebuilt {len(inputs.records)} traditional events but the fit "
            f"used {n_saved}: the table or the bundle is not the fit's."
        )
    s = inputs.session_obj
    events = [
        {"s": 0, "e": rec.n_bins,
         "spike_traj": s.trial_rate_trajs[rec.trial][:, rec.s_bin:rec.e_bin].T.astype(np.int32)}
        for rec in inputs.records
    ]
    kernel_share.score_events_per_kernel(
        events, hmm_dict, rows_by_model, rat=rat, session=session, kernels=tuple(kernels),
    )
    print(f"  {len(events)} traditional events: bin count and stored log-likelihoods match the fit")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--rat")
    parser.add_argument("--session")
    add_viewer_args(parser)
    parser.add_argument(
        "--run-tag", default=None,
        help="Run directory name; supersedes the tag derived from the decode grid. "
             "Required for config-hashed tags (dtbin_..__bwcm3p5__cfg<hash>).",
    )
    parser.add_argument(
        "--refresh-cache", action="store_true",
        help="Ignore any existing session-input cache and rebuild it from the raw .mat.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Neither read nor write the session-input cache.",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES_ROOT)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--maps-root", type=Path, default=None)
    parser.add_argument("--list-run-tags", action="store_true")
    # The decode grid, and every other event-defining parameter, for tag derivation only.
    config.add_decode_config_args(parser)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_run_tags:
        return print_run_tags(args.results_root)
    if not args.rat or not args.session:
        raise SystemExit("--rat and --session are required (or pass --list-run-tags).")

    cfg_cli = config.decode_config_from_args(args)
    run_dir = resolve_fit_dir(
        results_root=args.results_root,
        rat=args.rat,
        session=args.session,
        run_tag=args.run_tag,
        decode_tbin_s=cfg_cli.decode_tbin_s,
        decode_tbin_step_s=cfg_cli.decode_tbin_step_s,
    )
    run_tag = run_dir.name

    summary, rows_by_model = fit_summary.load_saved_fit_summary(
        filepath_summary=run_dir / names.FIT_SUMMARY_NAME,
    )
    compare_kernels = validate_kernel_args(args, rows_by_model, run_dir / names.FIT_SUMMARY_NAME)

    cfg = restore.extract_saved_run_settings(saved_summary=summary, fit_dir=run_dir).cfg
    # The CLI grid picks the fit; the fit's own config builds the bundle. Only flags the user
    # actually typed are checked -- an explicit --run-tag names the grid on its own, and the
    # unpassed defaults (`DecodeConfig`'s 0.010/0.010) would otherwise reject every fit on
    # another grid. The flags as typed: `argv` as the dispatcher passes it, else the command line.
    typed = sys.argv if argv is None else argv
    for field in ("decode_tbin_s", "decode_tbin_step_s"):
        flag = "--" + field.replace("_", "-")
        if not any(a == flag or a.startswith(flag + "=") for a in typed):
            continue
        if not np.isclose(getattr(cfg, field), getattr(cfg_cli, field)):
            raise SystemExit(
                f"{flag}={getattr(cfg_cli, field)} disagrees with the fit at {run_tag} "
                f"({field}={getattr(cfg, field)}). Drop the flag, or pass a --run-tag "
                "whose fit uses that grid."
            )
    fit_scope = fit_jump_scope(saved_event_source(summary))
    if fit_scope is not None:
        warn_if_scope_mismatch(fit_scope, args.jump_scope, run_tag)
    traditional = None
    if args.jump_scope == "traditional":
        csv_path, statuses = resolve_traditional_table(
            summary, args, fit_scope=fit_scope, local_dir=paths.LUSTIG_TRADITIONAL_DECODE_DIR,
            rat=args.rat, session=args.session,
        )
        events = traditional_io.load_traditional_events(
            csv_path, statuses=statuses, rats=[args.rat], sessions=[args.session],
        )
        traditional = (events, csv_path, statuses)
        print(f"  traditional events: {csv_path.parent.name}  statuses={list(statuses)}  "
              f"{len(events)} events")

    print(f"[{args.rat}/{args.session}] fit {run_tag}")
    print(f"  kernel={args.kernel}  scope={args.jump_scope}  "
          f"grid={cfg.decode_tbin_s:g}/{cfg.decode_tbin_step_s:g}")

    # The position trace is always in the inputs (and the cache); --no-position only
    # suppresses the line, in run_viewer.
    inputs = load_or_build_inputs(args=args, cfg=cfg, run_dir=run_dir, run_tag=run_tag,
                                  traditional=traditional)
    session_obj = inputs.session_obj

    # The HMM is rebuilt every launch: it is free once movement_P is in hand (build_hmm_dict
    # reads only lin_map_concat / lin_map_bounds / decode_tbin_s off the bundle), and the
    # fitted parameters come from the summary JSON, not from the cache.
    kernels = (args.kernel, *compare_kernels)
    fit_arm = restore.arm_kwargs_from_summary(summary)
    fit_sigma_cap = fit_arm["sigma_cap"]
    hmm_dict = lustig_kernels.build_hmm_dict(
        session_obj=session_obj,
        config=lustig_kernels.HMMDecodeConfig(kernels=kernels),
        movement_P_empirical=inputs.movement_P,
        # From the fit: rebuilding a larger-cap fit at the module default would make
        # apply_transition_params_from_summary_row raise on its stored sigma, and an eps=0 or
        # hard-zero fit would silently come back with the default kernels.
        **fit_arm,
    )
    if traditional is not None and fit_scope == "traditional":
        check_traditional_records(summary, rows_by_model, hmm_dict, inputs, kernels,
                                  rat=args.rat, session=args.session)
    for name in kernels:
        restore.apply_transition_params_from_summary_row(
            hmm=hmm_dict[name], summary_row=rows_by_model[name],
        )
        warn_if_sigma_capped(name, rows_by_model[name], sigma_cap=fit_sigma_cap)
    hmms = [hmm_dict[name] for name in kernels]

    if not inputs.records:
        what = "traditional events" if traditional is not None else "PBE-and-nonrun events"
        raise SystemExit(f"No {what} for {args.rat}/{args.session} at {run_tag}.")
    save_dir = args.save_dir or (
        Path(args.figures_root) / args.rat / args.session / run_tag / "events"
    )
    return run_viewer(
        source=LustigSource(
            inputs, data_root=None if args.data_root is None else Path(args.data_root),
            traditional=traditional is not None,
        ),
        records=inputs.records, hmms=hmms, kernels=kernels, args=args, run_tag=run_tag,
        save_dir=save_dir,
        load_saved_backdrop=lambda: load_backdrop(
            results_root=args.results_root, run_tag=run_tag, kernel=args.kernel,
            rat=args.rat, session=args.session, scope=args.jump_scope,
        ),
    )
