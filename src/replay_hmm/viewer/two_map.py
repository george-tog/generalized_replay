"""Two-map datasets (Pfeiffer, Buzsaki) for the per-event HMM browser.

The browser itself is `replay_hmm.viewer.engine` (imported as `ev`), with the same panels,
fixed-ceiling heatmaps, zoom squares, multi-kernel ΔLL ranking and controls, so the figures read
exactly like Lustig's. This module turns a saved two-map MB-trajectory or all-PBE fit into an
`ev.ViewerSource`; `replay_hmm.viewer.pfeiffer` and `replay_hmm.viewer.buzsaki` supply what is
genuinely theirs -- how a session's events are re-detected and decoded, its spikes, position and
detector trace, the jump sign function, the fit tree and the backdrop keys.
`replay-hmm viewer --dataset pfeiffer|buzsaki` runs it.

What differs from Lustig, and why:
  * Events are session-level, in absolute seconds, each decoded on its own window grid -- not
    lap-local runs over a lap-long rate trajectory. A record's bins are its own padded window.
  * Padding. Neither dataset's decoder takes a pad, so the pad windows are counted here: the
    fit's own window starts, extended by whole steps either side. The interior is therefore the
    fit's count matrix by construction (not a re-decode of a widened event, whose float window
    starts could move a boundary spike), and each event's interior is re-counted once with this
    module's counter and asserted equal, which pins the counter to the dataset's own rule.
  * State space [outbound | inbound], bounds [0, n, 2n], no wheel. Both maps index the same
    position bins, so the animal's position is drawn in both -- during a PBE it is immobile by
    construction and its direction is undefined, Lustig's reason for drawing both slots.
  * Jumps are signed by `jumps.signed.two_map_signed_jumps` (each dataset module's jump sign
    function), which negates the inbound map: + is forward, not "outbound".
  * Fits are rebuilt as the PPC rebuilds them (`rebuild_two_map.build_two_map_hmms`, the fitted
    rows, then `rebuild_two_map.restore_below_floor`), and guarded: the decoded-event and fitted-sequence counts against
    the summary, and every kernel's summed log-likelihood against its stored `loglik`
    (`kernel_share.score_events_per_kernel`). A viewer showing events the fit never saw fails here.

Scope: MB-trajectory (`traj`), all-PBE (`full-pbe`) and traditional-replay (`traditional`,
__tradreplay) fits. A traditional event is one of the PBEs this detection already finds (the
classifier table names it by `session_pbe_index`, aligned by start time), and the fit scores its
whole window on the fit's grid, so `--jump-scope traditional` is the decoded records filtered to
the table's events. For a __tradreplay fit the guards run on that filtered set, which is the
fit's own; on any other fit the scope needs --traditional-events-csv and warns, like any other
scope the fit was not trained on.

No input cache: a cold start is one session file plus one detection and decode, quick
enough without one (Lustig's raw .mat load is what its cache avoids).
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from replay_hmm.viewer import engine as ev
from replay_hmm.events import runs
from replay_hmm.io import fit_summary
from replay_hmm.io import names
from replay_hmm.hmm import rebuild_two_map

# --jump-scope by `replay-hmm jumps`'s recorded `jump_event_source`. Pfeiffer writes the fit's own
# event-source names, Buzsaki Lustig's.
JUMP_SCOPE_BY_EVENT_SOURCE = {
    **ev.JUMP_SCOPE_BY_EVENT_SOURCE,
    "mb_traj": "traj",
    "all_pbe": "full-pbe",
}
FIT_SCOPE_BY_EVENT_SOURCE = {"mb_traj": "traj", "all_pbe": "full-pbe", "traditional": "traditional"}

TWO_MAP_LABELS = ("outbound", "inbound")


@dataclass
class TwoMapEvent:
    """One decoded event, exactly as the fit CLI decoded it."""
    index: int                  # session_pbe_index
    starts_s: np.ndarray        # the fit's decode-window starts, absolute seconds
    counts: np.ndarray          # (n_windows, n_units) int32, the fit's spike counts
    segments: list              # [(map, s, e)] event-local, in the fit collector's order


@dataclass
class TwoMapSession:
    """What a dataset adapter hands `TwoMapSource`: its events plus the raw panels' inputs."""
    events: list                # [TwoMapEvent], decoded events in the fit's order
    spike_times: dict           # cell id -> sorted spike times (s)
    cell_ids: np.ndarray        # row order of the fit's lin_map_concat
    density_grids: list         # [(centers_s, density, eligible)], the detector's 1 ms trace(s)
    baseline_mean: float
    baseline_sd: float
    z_thresholds: tuple         # (peak, boundary), in baseline SDs
    pos_ts: np.ndarray          # s
    pos_cm: np.ndarray          # 1-D position, cm from the track's low end; may be filled
    bin_edges_cm: np.ndarray
    # Where pos_cm is a fill rather than an observation, and the fill's name for the info
    # block. None: every finite sample was observed.
    pos_observed: np.ndarray | None = None
    pos_fill: str | None = None


@dataclass
class TwoMapDataset:
    """What differs between Pfeiffer and Buzsaki outside the shared assembly."""
    name: str
    inputs_name: str
    default_results_root: Path
    default_figures_root: Path
    add_session_args: Callable          # (parser) -> None
    session_key: Callable               # (args) -> (rat, session); rat == session for Buzsaki
    session_rel: Callable               # (rat, session) -> Path under the results/figures roots
    display_names: Callable             # (rat, session) -> (name, name_spaced)
    # (rat, session, summary, inputs, label, *, traditional_rows=None) -> TwoMapSession; given
    # the table's rows, checks they name this detection's events (index and start time).
    load_session: Callable
    signed_jumps: Callable              # (z, bounds) -> signed jumps, + = forward
    backdrop_key: Callable              # (kernel, rat, session) -> key in jump_distributions.npz
    description: str
    # Where the traditional decode writes its tables (<dir>/<traditional run tag>/), and
    # (csv, statuses, rat, session, summary) -> this session's rows, in session_pbe_index order.
    traditional_dir: Path | None = None
    load_traditional_rows: Callable | None = None


def count_windows(spike_times: dict, cell_ids, starts_s: np.ndarray, tbin_s: float) -> np.ndarray:
    """(n_windows, n_units) int32 spike counts over [start, start + tbin_s).

    Both datasets' rule: `searchsorted(..., "left")` at each end, windows ending at start +
    tbin_s (`datasets.pfeiffer.events.build_spike_traj_for_event`, `decode.two_map.decode_pbe`).
    `TwoMapSource` asserts it against each event the viewer shows, so a change to either rule
    fails loudly.
    """
    starts_s = np.asarray(starts_s, dtype=float)
    ends_s = starts_s + float(tbin_s)
    out = np.zeros((starts_s.size, len(cell_ids)), dtype=np.int32)
    for i, cid in enumerate(cell_ids):
        spikes = spike_times.get(int(cid), np.empty(0))
        out[:, i] = (np.searchsorted(spikes, ends_s, side="left")
                     - np.searchsorted(spikes, starts_s, side="left"))
    return out


class TwoMapSource(ev.ViewerSource):
    """`ev.ViewerSource` over one two-map session. A record's bins index its padded window."""

    map_labels = TWO_MAP_LABELS
    arm_maps = (0, 1)
    jump_axis_labels = ("- reverse", "+ forward")

    def __init__(
        self, *, name: str, name_spaced: str, session: TwoMapSession, inputs: dict,
        summary: dict, pad_ms: float, signed_jumps: Callable,
    ) -> None:
        self.name, self.name_spaced = name, name_spaced
        self.session = session
        self.decode_tbin_s = float(summary["decode_tbin_s"])
        self.decode_tbin_step_s = float(summary["decode_tbin_step_s"])
        self.bounds = np.asarray(inputs["lin_map_bounds"], dtype=int)
        self.lin_map_concat = np.asarray(inputs["lin_map_concat"], dtype=float)
        self.cm_per_bin = rebuild_two_map.bin_width_cm(summary)
        self.max_map_bins = int(self.bounds[1] - self.bounds[0])
        self._signed_jumps = signed_jumps
        # The engine pads by exactly this many bins (EventViewer.pad_bins), so its padded slice
        # is this whole window.
        self.pad_bins = int(np.ceil((float(pad_ms) / 1000.0) / self.decode_tbin_step_s))
        self.events = {e.index: e for e in session.events}
        self._padded: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # -- records -------------------------------------------------------------------------

    def records(self) -> list:
        """One `ev.EventRecord` per decoded event, in the fit's order (= time order)."""
        out = []
        for e in self.session.events:
            n = int(e.counts.shape[0])
            t0 = float(e.starts_s[0])
            t1 = float(e.starts_s[-1]) + self.decode_tbin_s
            out.append(ev.EventRecord(
                trial=0, event_idx=int(e.index), s_bin=self.pad_bins,
                e_bin=self.pad_bins + n, t0_s=t0, t1_s=t1, peak_z=self._peak_z(t0, t1),
            ))
        return out

    def _peak_z(self, t0_s: float, t1_s: float) -> float:
        z = self._z_between(t0_s, t1_s)[1]
        return float(np.nanmax(z)) if z.size and np.isfinite(z).any() else float("nan")

    def event_label(self, rec) -> str:
        return f"PBE #{rec.event_idx}"

    def event_file_tag(self, rec) -> str:
        return f"pbe{rec.event_idx}"

    # -- counts --------------------------------------------------------------------------

    def _window(self, rec) -> tuple[np.ndarray, np.ndarray]:
        """(starts_s, counts) over the padded window, built on first use."""
        if rec.event_idx in self._padded:
            return self._padded[rec.event_idx]
        e = self.events[rec.event_idx]
        s = self.session
        recount = count_windows(s.spike_times, s.cell_ids, e.starts_s, self.decode_tbin_s)
        if not np.array_equal(recount, e.counts):
            raise AssertionError(
                f"{self.name} PBE #{e.index}: re-counting the fit's own windows does not reproduce "
                "the fit's spike counts, so count_windows no longer matches the dataset's rule."
            )
        k, step = self.pad_bins, self.decode_tbin_step_s
        left = e.starts_s[0] - step * np.arange(k, 0, -1)
        right = e.starts_s[-1] + step * np.arange(1, k + 1)
        starts = np.concatenate([left, e.starts_s, right])
        counts = np.concatenate([
            count_windows(s.spike_times, s.cell_ids, left, self.decode_tbin_s),
            e.counts,
            count_windows(s.spike_times, s.cell_ids, right, self.decode_tbin_s),
        ]).astype(np.int32)
        self._padded[rec.event_idx] = (starts, counts)
        return starts, counts

    def n_bins_total(self, rec) -> int:
        return rec.e_bin + self.pad_bins

    def counts(self, rec, lo: int, hi: int) -> np.ndarray:
        return self._window(rec)[1][lo:hi]

    def bin_centers_s(self, rec, lo: int, hi: int) -> np.ndarray:
        return self._window(rec)[0][lo:hi] + 0.5 * self.decode_tbin_s

    def traj_segments(self, rec) -> list:
        k = self.pad_bins
        return [(m, s + k, e + k) for m, s, e in self.events[rec.event_idx].segments]

    def scope_arrays(self, rec, scope: str) -> list:
        # Straight from the fit's counts, so scoring never needs the padded window.
        counts, k = self.events[rec.event_idx].counts, self.pad_bins
        return [counts[s - k:e - k] for s, e in self.scope_spans(rec, scope)]

    def signed_jumps(self, z, bounds) -> np.ndarray:
        return self._signed_jumps(z, bounds)

    # -- panels --------------------------------------------------------------------------

    def _z_between(self, lo_s: float, hi_s: float):
        """(centers, z, eligible) of the detector trace over [lo_s, hi_s], from the grid holding
        lo_s; z is in baseline SDs, or the raw trace when the source records no baseline."""
        s = self.session
        for centers, density, eligible in s.density_grids:
            if centers.size and centers[0] <= lo_s <= centers[-1]:
                a = int(np.searchsorted(centers, lo_s, side="left"))
                b = int(np.searchsorted(centers, hi_s, side="right"))
                vals = np.asarray(density[a:b], dtype=float)
                if np.isfinite(s.baseline_sd) and s.baseline_sd > 0:
                    vals = (vals - s.baseline_mean) / s.baseline_sd
                return centers[a:b], vals, np.asarray(eligible[a:b], dtype=bool)
        return np.empty(0), np.empty(0), np.empty(0, dtype=bool)

    def density(self, rec, t_lo_ms: float, t_hi_ms: float):
        centers, vals, eligible = self._z_between(
            rec.t0_s + t_lo_ms / 1000.0, rec.t0_s + t_hi_ms / 1000.0,
        )
        if not centers.size:
            return None
        s = self.session
        z_scored = np.isfinite(s.baseline_sd) and s.baseline_sd > 0
        t_ms = (centers - rec.t0_s) * 1000.0
        spans = [(t_ms[a], t_ms[min(b, t_ms.size - 1)]) for a, b in runs.runs_from_mask(~eligible)]
        return ev.DensityTrace(
            t_ms=t_ms, values=vals,
            thresholds=tuple(s.z_thresholds) if z_scored else (),
            ineligible_ms=spans,
            ylabel="z spike density" if z_scored else "spike density",
        )

    def raster_ms(self, row: int, rec, t_lo_ms: float, t_hi_ms: float) -> np.ndarray:
        spikes = self.session.spike_times.get(int(self.session.cell_ids[int(row)]), np.empty(0))
        a = np.searchsorted(spikes, rec.t0_s + t_lo_ms / 1000.0, side="left")
        b = np.searchsorted(spikes, rec.t0_s + t_hi_ms / 1000.0, side="right")
        return (spikes[a:b] - rec.t0_s) * 1000.0

    def position_lines(self, rec, t_lo_ms: float, t_hi_ms: float):
        """The animal's position bin, drawn in both maps; 1 ms resolution, NaN where untracked."""
        s = self.session
        t_ms = np.arange(np.ceil(t_lo_ms), np.floor(t_hi_ms) + 1.0)
        if not t_ms.size:
            return [], "n/a"
        pos = np.interp(rec.t0_s + t_ms / 1000.0, s.pos_ts, s.pos_cm, left=np.nan, right=np.nan)
        n = self.max_map_bins
        b = np.searchsorted(s.bin_edges_cm, pos, side="right") - 1
        ok = np.isfinite(pos) & (b >= 0) & (b < n)
        lines = [
            (t_ms, np.where(ok, self.bounds[m] + np.clip(b, 0, n - 1), np.nan).astype(float))
            for m in (0, 1)
        ]
        if not ok.any():
            return lines, "untracked"
        desc = f"{np.nanmedian(pos[ok]):.0f} cm"
        if s.pos_observed is not None:
            # A sample counts as observed only when both neighbouring position samples were.
            seen = np.interp(rec.t0_s + t_ms / 1000.0, s.pos_ts,
                             s.pos_observed.astype(float), left=0.0, right=0.0) >= 1.0
            if not seen[ok].all():
                desc += f" ({s.pos_fill} fill)"
        return lines, desc


# --------------------------------------------------------------------------------------
# fit loading and guards
# --------------------------------------------------------------------------------------

def resolve_fit_dir(results_root: Path, rel: Path, run_tag: str) -> Path:
    """The run directory, or SystemExit naming the run tags that do hold a full fit."""
    session_dir = Path(results_root) / rel
    run_dir = session_dir / run_tag
    if (run_dir / names.FIT_SUMMARY_NAME).exists() and (run_dir / names.TRANSITION_MATS_NAME).exists():
        return run_dir
    usable = sorted(
        p.parent.name for p in session_dir.glob(f"*/{names.FIT_SUMMARY_NAME}")
        if (p.parent / names.TRANSITION_MATS_NAME).exists()
    )
    lines = [f"No usable fit at {run_dir}."]
    if usable:
        lines.append("  run tags here with a full-data fit:")
        lines += [f"    {t}" for t in usable]
    else:
        lines.append(f"  {session_dir} holds no full-data fit.")
    raise SystemExit("\n".join(lines))


def fit_scope(summary: dict, label: str) -> str:
    """`traj`, `full-pbe` or `traditional` for the fit's event set."""
    source = fit_summary.fit_event_source(summary, policy=fit_summary.TWO_MAP_PPC)
    if source not in FIT_SCOPE_BY_EVENT_SOURCE:
        raise SystemExit(f"[{label}] unrecognised event_source {source!r}.")
    return FIT_SCOPE_BY_EVENT_SOURCE[source]


def traditional_records(records: list, rows, label: str) -> list:
    """The records the traditional table names, in the records' (the fit's) order.

    Every named event must have been decoded: both fits raise on an accepted event they cannot
    decode, so one missing here means this rebuild is not the fit's.
    """
    wanted = {int(i) for i in rows["session_pbe_index"].tolist()}
    missing = sorted(wanted - {int(r.event_idx) for r in records})
    if missing:
        raise ValueError(
            f"[{label}] the traditional table names {len(missing)} event(s) with no decoded "
            f"record, e.g. session_pbe_index {missing[:5]}: detection or decoding is not "
            "reproducing the traditional decode's event set."
        )
    return [r for r in records if int(r.event_idx) in wanted]


def fitted_sequences(source: TwoMapSource, records: list, scope: str) -> list[np.ndarray]:
    """The fit's training sequences in its own order: the collectors keep length >= 2."""
    return [a for rec in records for a in source.scope_arrays(rec, scope) if a.shape[0] >= 2]


def check_event_set(summary: dict, n_decoded: int, sequences: list, label: str) -> None:
    """Decoded-event and fitted-sequence counts against the summary, as the PPC rebuild does."""
    for key, got in (("n_decoded_pbes", n_decoded), ("n_cv_events", len(sequences))):
        saved = summary.get(key)
        if saved is None:
            raise ValueError(f"[{label}] the fit records no {key}; its event set cannot be checked.")
        if int(got) != int(saved):
            raise ValueError(
                f"[{label}] rebuilt {got} for {key} but the fit recorded {saved}: detection, "
                "decoding or trajectory calling is not reproducing the fit."
            )


def build_hmms(summary: dict, rows_by_model: dict, inputs: dict, kernels: tuple,
               sequences: list, *, rat: str, session: str) -> dict:
    """Each kernel at its fitted parameters, guarded against the fit's stored log-likelihood.

    `kernel_share.score_events_per_kernel` restores each row through ssm's setters and checks the bin
    count (hard) and the summed log-likelihood (the shared tolerance tiers); the row is then
    re-applied with the below-floor scalars restored exactly, as the kernel PPC's rebuild does.
    """
    from replay_hmm.analyses import kernel_share

    hmm_dict = rebuild_two_map.build_two_map_hmms(summary, inputs, kernels)
    events = [{"s": 0, "e": int(a.shape[0]), "spike_traj": a} for a in sequences]
    kernel_share.score_events_per_kernel(
        events, hmm_dict, rows_by_model, rat=rat, session=session, kernels=tuple(kernels),
    )
    rebuild_two_map.restore_and_score(
        hmm_dict, rows_by_model, sequences, kernels, label=f"{rat}/{session}",
        note=rebuild_two_map.FLOOR_RESTORED_NOTE_VIEWER, flush=False,
        after_restore=lambda k: ev.warn_if_sigma_capped(
            k, rows_by_model[k], sigma_cap=summary.get("sigma_cap")
        ),
    )
    return hmm_dict


def load_backdrop(*, results_root: Path, run_tag: str, scope: str, key: str) -> np.ndarray:
    """One session's saved jumps for one kernel, from the `replay-hmm jumps` run matching `scope`.

    The two datasets' jump runs name a jump-set cell differently -- Pfeiffer appends
    __allpbe_jumps (__tradreplay_jumps) only when it differs from the fit's set, Buzsaki whenever
    the jumps are all-PBE (traditional) -- so every candidate directory is tried and the one
    whose metadata records this scope wins. A directory of the other scope is used only as a
    last resort, with Lustig's warning.
    """
    base = Path(results_root) / names.JUMP_DIST_DIRNAME
    found = []
    for tag in (run_tag, run_tag + "__allpbe_jumps", run_tag + "__tradreplay_jumps"):
        d = base / tag
        if not (d / names.JUMP_DIST_NAME).exists():
            continue
        meta_path = d / names.JUMP_META_NAME
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        found.append((d, JUMP_SCOPE_BY_EVENT_SOURCE.get(meta.get("jump_event_source")), meta))
    if not found:
        raise SystemExit(
            f"--session-backdrop saved needs {base / run_tag / names.JUMP_DIST_NAME}, which does not "
            "exist. Run `replay-hmm jumps --dataset buzsaki|pfeiffer` for this tag, or pass "
            "--session-backdrop compute or none."
        )
    matched = [f for f in found if f[1] == scope]
    d, saved_scope, meta = (matched or found)[0]
    if saved_scope != scope:
        warnings.warn(
            f"saved backdrop in {d.name} was harvested at scope {saved_scope!r} "
            f"(jump_event_source={meta.get('jump_event_source')!r}) but --jump-scope is "
            f"{scope!r}. The backdrop and the per-event histogram are not directly comparable; "
            "use --session-backdrop compute for a matched one.",
            RuntimeWarning, stacklevel=2,
        )
    with np.load(d / names.JUMP_DIST_NAME, allow_pickle=False) as z:
        if key not in z.files:
            raise SystemExit(f"{d / names.JUMP_DIST_NAME} has no entry {key!r}. "
                             "Pass --session-backdrop compute or none.")
        return np.asarray(z[key], dtype=int)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv, ds: TwoMapDataset) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=ds.description, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    ds.add_session_args(parser)
    parser.add_argument("--run-tag", required=True,
                        help="Fit directory name, e.g. "
                             "dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens (Pfeiffer) or "
                             "dtbin_0p01__step_0p01__hmmbin_0p02 (Buzsaki).")
    parser.add_argument("--results-root", type=Path, default=ds.default_results_root)
    parser.add_argument("--figures-root", type=Path, default=ds.default_figures_root)
    ev.add_viewer_args(parser)
    return parser.parse_args(argv)


def main(argv=None, *, dataset: TwoMapDataset) -> int:
    ds = dataset
    args = parse_args(argv, ds)
    rat, session = ds.session_key(args)
    name, name_spaced = ds.display_names(rat, session)
    rel = ds.session_rel(rat, session)
    run_dir = resolve_fit_dir(args.results_root, rel, args.run_tag)
    run_tag = run_dir.name

    summary, rows_by_model = fit_summary.load_saved_fit_summary(
        filepath_summary=run_dir / names.FIT_SUMMARY_NAME
    )
    compare_kernels = ev.validate_kernel_args(args, rows_by_model, run_dir / names.FIT_SUMMARY_NAME)
    kernels = (args.kernel, *compare_kernels)
    fscope = fit_scope(summary, name)
    ev.warn_if_scope_mismatch(fscope, args.jump_scope, run_tag)
    rows = None
    if "traditional" in (fscope, args.jump_scope):
        if ds.load_traditional_rows is None:
            raise SystemExit(f"[{name}] {ds.name} has no traditional events table.")
        csv_path, statuses = ev.resolve_traditional_table(
            summary, args, fit_scope=fscope, local_dir=ds.traditional_dir, rat=rat, session=session,
        )
        rows = ds.load_traditional_rows(csv_path, statuses, rat, session, summary)
        print(f"  traditional events: {csv_path.parent.name}  statuses={list(statuses)}  "
              f"{len(rows)} events")

    print(f"[{name}] fit {run_tag}")
    print(f"  kernel={args.kernel}  scope={args.jump_scope}  "
          f"grid={float(summary['decode_tbin_s']):g}/{float(summary['decode_tbin_step_s']):g}")

    inputs = rebuild_two_map.load_inputs(run_dir / ds.inputs_name, summary, name)
    print("  detecting and decoding events...")
    sess = ds.load_session(rat, session, summary, inputs, name, traditional_rows=rows)
    source = TwoMapSource(
        name=name, name_spaced=name_spaced, session=sess, inputs=inputs, summary=summary,
        pad_ms=args.pad_ms, signed_jumps=ds.signed_jumps,
    )
    records = source.records()
    trad = None if rows is None else traditional_records(records, rows, name)
    # The fitted set is fixed by the fit, not by --jump-scope: a __tradreplay fit decoded only
    # the table's events, so its guards run on those.
    fit_records = trad if fscope == "traditional" else records
    fit_set = fitted_sequences(source, fit_records, fscope)
    check_event_set(summary, len(fit_records), fit_set, name)
    hmm_dict = build_hmms(summary, rows_by_model, inputs, kernels, fit_set, rat=rat, session=session)
    print(f"  rebuilt {len(fit_records)} decoded events, {len(fit_set)} fitted sequences; "
          "counts and stored log-likelihoods match the fit")
    if args.jump_scope == "traditional":
        records = trad

    if not records:
        raise SystemExit(f"No decoded events for {name} at {run_tag} (scope {args.jump_scope}).")
    save_dir = args.save_dir or (Path(args.figures_root) / rel / run_tag / "events")
    return ev.run_viewer(
        source=source, records=records, hmms=[hmm_dict[k] for k in kernels], kernels=kernels,
        args=args, run_tag=run_tag, save_dir=save_dir,
        load_saved_backdrop=lambda: load_backdrop(
            results_root=args.results_root, run_tag=run_tag, scope=args.jump_scope,
            key=ds.backdrop_key(args.kernel, rat, session),
        ),
    )

