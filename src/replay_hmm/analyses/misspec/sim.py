"""Generators, decoders and synthetic events of the misspecification gallery and sweep.

The state space is the Lustig center arm's two directional maps, C_in + C_out, of one source
fit's own session bundle (`load_center_arm_inputs`); spikes are Poisson draws from that
session's center-arm place fields, the exact observation model every decoder uses. TCM-SR and
continuous generators take the source fit's parameters, momentum the Pfeiffer & Foster 2015
fits at the same bin width (`pfeiffer_momentum_point`). Every kernel is built with
eps = alpha = 0 exactly (`analyses.misspec.exact_zero`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from replay_hmm import paths
from replay_hmm.hmm import kernel_sets as hks
from replay_hmm.hmm import momentum as m2
from replay_hmm.hmm import two_map_kernels
from replay_hmm.datasets.lustig import movement as lustig_movement
from replay_hmm.hmm import restore
from replay_hmm.io import fit_summary
from replay_hmm.hmm.sampling import sample_state_path_from_P
from replay_hmm.analyses.recovery import run as rec
from replay_hmm.analyses.recovery import template as rec_template
from replay_hmm.datasets.lustig import bundle
from replay_hmm.datasets.lustig import calibration
from replay_hmm.datasets.lustig import event_sets
from replay_hmm.analyses.misspec.exact_zero import make_exact_zero

GENERATOR_KERNELS = ("continuous_EM", "momentum", "sticky_TCMSR")
# Each generator's RNG stream index, fixed by name so the display order can change without
# changing a simulated event or invalidating saved fits.
GENERATOR_RNG_STREAMS = {"sticky_TCMSR": 0, "continuous_EM": 1, "momentum": 2}
assert set(GENERATOR_RNG_STREAMS) == set(GENERATOR_KERNELS), GENERATOR_RNG_STREAMS
# Generators parameterised by this session's own fit; momentum comes from Pfeiffer & Foster.
FIT_ROW_GENERATORS = ("sticky_TCMSR", "continuous_EM")
PFEIFFER_FITS_ROOT = paths.PFEIFFER_FITS_DIR
# One preset per decode bin width: the source fit (event set, TCM-SR and continuous parameters),
# the Pfeiffer & Foster tag momentum is taken from, and the continuous generator's sigma in bins
# (None keeps the source fit's own sigma_arm).
BIN_CONFIGS = {
    "10ms": {
        "run_tag": "dtbin_0p01__step_0p01__bwcm3p5__cfgddf2e05f67",
        "pfeiffer_tag": "dtbin_0p01__step_0p01__hmmbin_4__evt_spikedens",
        # Fixed at 2 bins rather than taken from the fit: the fitted sigma_arm sits at the
        # decoders' 3-bin cap.
        "continuous_sigma_bins": 2.0,
    },
    "3ms": {
        # rec04 has no 3 ms mbtraj fit, so the 3 ms source fit is the all-PBE one.
        "run_tag": "dtbin_0p003__step_0p003__bwcm3p5__cfgf58e7e21b2__allpbe",
        "pfeiffer_tag": "dtbin_0p003__step_0p003__hmmbin_4__evt_spikedens__allpbe",
        "continuous_sigma_bins": None,
    },
}
# Every kernel fitted and decoded: the scored kernels minus stationary, movement_empirical,
# sticky_SR_rev and sticky_TCMSR_sym. All of them enter the aggregate figures (error heatmap,
# LL-vs-error scatter); the galleries show `GALLERY_DECODERS` only.
DECODER_KERNELS = (
    "sticky_TCMSR", "sticky_SR", "continuous_EM", "momentum", "segment_uniform", "fragmented",
)
assert set(DECODER_KERNELS) <= set(hks.SCORED_KERNELS), DECODER_KERNELS
CENTER_ARM = "C"


# --------------------------------------------------------------------------------------
# Inputs, kernels, simulation
# --------------------------------------------------------------------------------------

def load_center_arm_inputs(
    rat: str, session: str, run_tag: str, fits_root: Path, required_rows=FIT_ROW_GENERATORS
) -> dict:
    """The fit's own session bundle, restricted to the center arm's two maps.

    `required_rows` are the kernels whose fitted row the caller reads parameters from."""
    summary_path = Path(fits_root) / rat / session / run_tag / "full_fit_summary_full_em_mbtraj.json"
    summary, rows_by_model = fit_summary.load_saved_fit_summary(summary_path)
    # The template is "every event of the source fit": all mbtraj segments, or all PBE windows.
    # An arm-restricted or traditional-replay fit describes a different event set.
    event_source = fit_summary.fit_event_source(
        summary, policy=fit_summary.MISSPEC_TEMPLATE, label=str(summary_path)
    )
    if summary.get("mb_traj_arm_only"):
        raise SystemExit(
            fit_summary.MISSPEC_TEMPLATE.unknown_message.format(label=str(summary_path))
        )
    missing = [k for k in required_rows if k not in rows_by_model]
    if missing:
        raise SystemExit(f"{summary_path} has no fitted row for generator kernel(s) {missing}.")

    cfg = restore.extract_saved_run_settings(summary, fit_dir=summary_path.parent).cfg
    # Absolute roots; the bundle's and the movement prior's None defaults are the same `paths`
    # roots.
    session_obj = bundle.build_session_bundle(
        rat, session, cfg=cfg, data_root=paths.LUSTIG_DATA_DIR, maps_root=paths.RESULTS_ROOT,
    )
    if event_source == "all_pbe":
        # The collector the Lustig fit CLI fits all-PBE runs on.
        event_lengths = [
            int(d.shape[0])
            for d in event_sets.collect_event_sequences_for_fit(
                session_obj=session_obj, window_source="pbe_nonrun"
            )
        ]
    else:
        event_lengths, _ = rec_template._extract_template_event_lengths(session_obj)
    rec._assert_template_matches_fit(
        event_lengths=event_lengths,
        rows_by_model=rows_by_model,
        rat=rat,
        session=session,
        filepath_summary=summary_path,
    )
    # The arguments `datasets.lustig.session.prepare_session_and_hmm_dict` passes, including
    # the RESOLVED bin width.
    movement_out = lustig_movement.empirical_movement_P(
        session_obj=session_obj,
        speed_thresh_m_s=float(cfg.speed_thresh_m_s),
        behavior_dt_s=float(cfg.dt_s),
        bin_width_px=int(getattr(session_obj, "bin_width_px_resolved", 0) or cfg.bin_width_px),
        data_root=paths.LUSTIG_DATA_DIR,
    )
    movement_P = movement_out[0] if isinstance(movement_out, tuple) else movement_out

    bounds = np.asarray(session_obj.lin_map_bounds, dtype=int)
    slots = event_sets.mb_traj_arm_slots(CENTER_ARM)
    if len(slots) != 2 or slots[1] != slots[0] + 1:
        raise ValueError(f"Expected two adjacent center-arm slots, got {slots}.")
    lo, mid, hi = bounds[slots[0]], bounds[slots[1]], bounds[slots[1] + 1]
    n_c = int(mid - lo)
    if int(hi - mid) != n_c:
        raise ValueError(f"C_in and C_out differ in size: {n_c} vs {int(hi - mid)} bins.")
    movement_P_c, zeroed_states = center_arm_movement_P(movement_P, lo, hi)

    return {
        "summary": summary,
        "rows_by_model": rows_by_model,
        "summary_path": summary_path,
        "event_lengths": [int(t) for t in event_lengths],
        "place_fields_hz": np.asarray(session_obj.lin_map_concat[:, lo:hi], dtype=float),
        "movement_P": movement_P_c,
        "zeroed_states": zeroed_states,
        "n_c": n_c,
        "center_slots": [int(s) for s in slots],
        "state_lo": int(lo),
        "bin_width_cm": float(calibration.bin_cm_from_bundle(session_obj)),
        "decode_tbin_s": float(session_obj.decode_tbin_s),
    }


def center_arm_movement_P(movement_P: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, list[int]]:
    """Center-arm block of the empirical movement P, with uniform-filled rows zeroed.

    `empirical_movement_P` fills a row with no behavioural transitions with a uniform 1/K. Such
    a row keeps a jump channel open in the empirical kernels even at eps = alpha = 0:
    movement_empirical can jump anywhere from that state, and the SR family's multi-step
    lookahead spreads the fill to every state upstream of it. Zeroed, the state has no local
    move, so with alpha = 0 it can only stay (kappa). Returns the block and the zeroed states,
    indexed within the block.
    """
    filled = np.all(movement_P[lo:hi] == 1.0 / movement_P.shape[1], axis=1)
    block = np.array(movement_P[lo:hi, lo:hi], dtype=float)
    block[filled] = 0.0
    return block, np.flatnonzero(filled).tolist()


def build_center_arm_kernels(inputs: dict, kernels) -> dict:
    """Kernels from the two-map fits' builder (`build_two_map_hmm_dict`) on the center arm's
    C_in + C_out state space, with eps = alpha = 0 exactly."""
    n_c = inputs["n_c"]
    hmm_dict = two_map_kernels.build_two_map_hmm_dict(
        lin_map_concat_hmm=inputs["place_fields_hz"],
        lin_map_bounds_hmm=np.array([0, n_c, 2 * n_c]),
        movement_P=inputs["movement_P"],
        kernels=tuple(kernels),
        decode_tbin_s=inputs["decode_tbin_s"],
        sigma_cap=restore.sigma_cap_from_summary(inputs["summary"]),
        freeze_eps_zero=True,
        bin_width_cm=inputs["bin_width_cm"],
    )
    return make_exact_zero(hmm_dict)


def pfeiffer_momentum_point(tag: str, root: Path = PFEIFFER_FITS_ROOT):
    """Momentum generator parameters from the Pfeiffer & Foster 2015 fits under `tag`.

    At 10 ms momentum beats continuous_EM on every session of that dataset, which Lustig's fits
    do not show (at 3 ms, on 6 of 9). The point is the per-session median of the fitted
    (a, sigma_v), each read at that fit's own decode bin. It is returned as
    (lambda_m, sigma_m), the model's physical parameters (1/s, cm/s^1.5), so it describes the
    same process at the misspec run's bin width and decode step. Also returns the per-session rows.
    """
    sessions = []
    for path in sorted(Path(root).glob(f"*/*/{tag}/full_fit_summary_full_em_mbtraj.json")):
        summary, rows = fit_summary.load_saved_fit_summary(path)
        if "momentum" not in rows:
            continue
        row = rows["momentum"]
        sessions.append({
            "session": f"{path.parts[-4]}/{path.parts[-3]}",
            "a": float(row["momentum_a"]),
            "sigma_v_cm_s": float(row["momentum_sigma_v_cm_s"]),
            "dt_s": float(summary["decode_tbin_s"]),
            "loglik_minus_continuous_EM": float(row["loglik"]) - float(rows["continuous_EM"]["loglik"]),
        })
    if not sessions:
        raise SystemExit(f"No Pfeiffer momentum fits under {root}/*/*/{tag}.")
    dts = {s["dt_s"] for s in sessions}
    if len(dts) != 1:
        raise ValueError(f"Pfeiffer fits under {tag} disagree on the decode bin: {sorted(dts)}.")
    point = m2.MomentumGridPoint.from_a_sigma_v(
        float(np.median([s["a"] for s in sessions])),
        float(np.median([s["sigma_v_cm_s"] for s in sessions])),
        dts.pop(),
    )
    return point, sessions


def pure_gaussian_kappa(transitions, n_c: int) -> float:
    """The kappa that adds no stickiness to the Gaussian kernel: its own self-transition
    probability, read off the middle row of the block the class builds.

    The class zeroes the Gaussian's diagonal and puts kappa there instead, so with this kappa
    every row away from the arm ends is exactly the plain Gaussian walk. Rows within ~2 sigma
    of an end have a larger self-probability under truncation and get slightly less here.
    """
    G = transitions._gaussian_block(n_c, transitions.sigma_arm)
    return float(G[n_c // 2, n_c // 2])


def build_generators(inputs: dict, momentum_point, continuous_sigma_bins: float | None) -> dict:
    """Generator kernels with eps = alpha = 0 exactly.

    TCM-SR and continuous take the source fit's full-data rows, applied without their `eps`;
    continuous sigma is then set to `continuous_sigma_bins` unless that is None, and continuous
    kappa to the Gaussian's own self-transition probability (`pure_gaussian_kappa`). Momentum
    takes `momentum_point` (see `pfeiffer_momentum_point`).
    """
    generators = build_center_arm_kernels(inputs, GENERATOR_KERNELS)
    for name in FIT_ROW_GENERATORS:
        row = {k: v for k, v in inputs["rows_by_model"][name].items() if k != "eps"}
        restore.apply_transition_params_from_summary_row(generators[name], row)
    cont = generators["continuous_EM"].transitions
    if continuous_sigma_bins is not None:
        cont.sigma_arm = float(continuous_sigma_bins)
    cont.kappa = pure_gaussian_kappa(cont, inputs["n_c"])
    generators["momentum"].set_grid_point(momentum_point, eps=0.0)
    return make_exact_zero(generators)


def generator_params(gen) -> dict:
    """The parameters a generator actually samples with, read back off the object."""
    t = gen.transitions
    if hasattr(gen, "set_grid_point"):
        return {
            "momentum_a": float(t.a),
            "momentum_sigma_v_cm_s": float(t.sigma_v_cm_s),
            "jump_sd_cm": float(t.jump_sd_cm),
            "lambda_m": float(t.lambda_m),
            "sigma_m": float(t.sigma_m),
            "eps": float(t.eps),
        }
    out = {"kappa": float(t.kappa), "alpha": float(t.alpha), "eps_effective": 0.0}
    if getattr(t, "use_tcmsr", False):
        out.update(gamma=float(t.gamma), omega=float(t.omega))
    if hasattr(t, "sigma_arm"):
        out["sigma_arm_bins"] = float(t.sigma_arm)
    return out


def emission_rates_per_bin(inputs: dict) -> np.ndarray:
    """(n_units, K) Poisson means per bin -- the lambdas `build_two_map_hmm_dict` fixes."""
    rates_hz = np.nan_to_num(inputs["place_fields_hz"], nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(rates_hz * inputs["decode_tbin_s"], 1e-12, None)


def simulate_events(
    gen, event_lengths, rates_per_bin: np.ndarray, rng, init_probs=None
) -> tuple[list, list]:
    """One (true state path, int32 spike counts) pair per length; `init_probs` defaults to the
    generator's uniform start over all states."""
    P = None if hasattr(gen, "sample_state_path") else np.asarray(gen.transitions.transition_matrix)
    paths, datas = [], []
    for T in event_lengths:
        if P is None:
            z = gen.sample_state_path(int(T), rng, init_probs=init_probs)
        else:
            z = sample_state_path_from_P(P, int(T), rng, init_probs=init_probs)
        z = np.asarray(z, dtype=int)
        paths.append(z)
        datas.append(rng.poisson(rates_per_bin[:, z].T).astype(np.int32))
    return paths, datas
