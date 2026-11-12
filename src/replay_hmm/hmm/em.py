"""Full EM for one kernel: the learn flags it fits under, the per-bin stopping rule, the driver
and its convergence diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def configure_full_em(
    model_name: str,
    hmm,
    mstep_lr: float = 0.01,
    mstep_steps: int = 25,
    mstep_clip: float = 5.0,
):
    """
    Configure transition learning flags and M-step optimizer settings for full EM.
    """
    t = hmm.transitions

    if hasattr(t, "mstep_lr"):
        t.mstep_lr = float(mstep_lr)
    if hasattr(t, "mstep_steps"):
        t.mstep_steps = int(mstep_steps)
    if hasattr(t, "mstep_clip"):
        t.mstep_clip = float(mstep_clip)

    # Baseline/null models remain fixed by construction. `momentum` joins them here not
    # because it is a null -- it is fitted, at p = 2 -- but because ssm's EM is not its
    # optimizer: its transitions bag carries no learn_* flags at all, so every hasattr guard
    # below would skip anyway. Naming it is what keeps that intentional rather than incidental.
    if model_name in {"stationary", "fragmented", "momentum"}:
        if hasattr(t, "learn_eps"):
            t.learn_eps = False
        if hasattr(t, "learn_kappa"):
            t.learn_kappa = False
        if hasattr(t, "learn_gamma"):
            t.learn_gamma = False
        if hasattr(t, "learn_omega"):
            t.learn_omega = False
        return hmm

    # `freeze_eps` is set under freeze_eps_zero, by the Lustig `build_hmm_dict` and the two-map
    # `build_two_map_hmm_dict` (through `build.apply_freeze_eps_zero`). Without this guard the
    # unconditional re-enable would undo the freeze at fit time -- and do it silently, since eps
    # still reads correct until the first M-step moves it.
    if hasattr(t, "learn_eps"):
        t.learn_eps = not bool(getattr(t, "freeze_eps", False))
    if hasattr(t, "learn_kappa"):
        t.learn_kappa = True

    uses_gamma = bool(getattr(t, "use_sr", False) or getattr(t, "use_tcmsr", False))
    if hasattr(t, "learn_gamma"):
        t.learn_gamma = uses_gamma
    # `freeze_omega` is set by both builders for `sticky_TCMSR_sym`, whose omega is pinned at
    # 1. Same guard, and same failure mode, as `freeze_eps` above: without it this line would
    # re-enable learning and the symmetric kernel would silently drift back into being a second
    # copy of sticky_TCMSR, charged three BIC parameters while fitting four.
    if hasattr(t, "learn_omega"):
        t.learn_omega = bool(getattr(t, "use_tcmsr", False)) and not bool(
            getattr(t, "freeze_omega", False)
        )

    return hmm


def has_learnable_transition_params(hmm) -> bool:
    t = hmm.transitions
    learn_flags = (
        "learn_eps",
        "learn_kappa",
        "learn_gamma",
        "learn_sigma_wheel",
        "learn_sigma_arm",
        "learn_omega",
    )
    return any(bool(getattr(t, flag, False)) for flag in learn_flags)


# ─── EM convergence: tolerance scaling and diagnostics ────────────────────────
#
# ssm's `_fit_em` stops when `abs(lls[-1] - lls[-2]) < tolerance`, where lls are
# TOTAL log-probabilities summed over every bin of every event. That total grows
# with dataset size, so a fixed absolute tolerance means something different for
# a 1.4k-bin Lustig session than for a 34k-bin Buzsaki session. Expressing the
# tolerance per bin makes the stopping rule comparable across sessions, datasets
# and CV folds (folds hold 4/5 of the bins, so even within one session the
# absolute threshold is not fold-invariant).

# Log-prob change per bin below which an EM iteration counts as converged.
DEFAULT_EM_TOL_PER_BIN = 1e-5


# Every capped transition scalar is a logistic onto its bounded interval in ssm: eps, gamma
# and omega are `cap * logistic(u)`, sigma is `sigma_min + (cap - sigma_min) * logistic(u)`.
# The M-step gradient in u shrinks with logistic'(u), so a parameter driven towards a bound
# slows to a stop just inside it and never lands on it (fits held at the eps cap of 0.02 stop
# between 0.0198 and 0.01996). Such a fit has stopped at a bound, not at an interior optimum,
# and the likelihood is flat in that direction. The at-bound flags separate the two with a
# threshold relative to the width of each parameter's own interval.
#
# Expected by construction: `sticky_TCMSR_sym` reports omega_at_cap=True on every fit, because
# omega=1 at omega_cap=1 IS that kernel. It is pinned and excluded from the M-step, so there is
# nothing to diagnose; lowering the cap would NaN the M-step (see the kernel's block in
# build_hmm_dict).
_LOGISTIC_BOUND_TOL_FRAC = 1e-2

# Parameters whose ssm setter clips silently to the cap (see `restore`).
_SIGMA_PARAMS = frozenset({"sigma_arm", "sigma_wheel"})


_PARAM_BOUND_ATTRS = {
    "eps": ("eps_cap", None),
    "gamma": ("gamma_cap", None),
    "omega": ("omega_cap", None),
    "sigma_arm": ("sigma_cap", "sigma_min"),
    "sigma_wheel": ("sigma_cap", "sigma_min"),
}


def resolve_em_tolerance(
    *,
    em_tol: float | None,
    em_tol_per_bin: float | None,
    n_bins: int,
) -> tuple[float, str]:
    """Return (effective_tolerance, mode) for one call to `model.fit(method="em")`.

    `em_tol_per_bin` takes precedence when given: the tolerance handed to ssm is
    `em_tol_per_bin * n_bins`, where `n_bins` is the bin count of the data being
    fit (train bins for a CV fold, all bins for the full fit). Pass only `em_tol`
    for an absolute threshold.
    """
    n_bins = max(1, int(n_bins))
    if em_tol_per_bin is not None:
        return float(em_tol_per_bin) * n_bins, "per_bin"
    if em_tol is None:
        raise ValueError("Provide em_tol or em_tol_per_bin.")
    return float(em_tol), "absolute"


def transition_param_snapshot(t) -> dict:
    """Current value of every scalar transition parameter, plus at-bound flags."""
    snap = {}
    for name, (cap_attr, min_attr) in _PARAM_BOUND_ATTRS.items():
        if not hasattr(t, name):
            continue
        val = float(getattr(t, name))
        snap[name] = val
        cap = getattr(t, cap_attr, None)
        lo = None if min_attr is None else getattr(t, min_attr, None)
        if cap is None:
            if lo is not None:
                snap[f"{name}_at_min"] = bool(abs(val - float(lo)) <= 1e-9)
            continue
        span = float(cap) - (0.0 if lo is None else float(lo))
        tol = _LOGISTIC_BOUND_TOL_FRAC * span
        snap[f"{name}_at_cap"] = bool(val >= float(cap) - tol)
        if lo is not None:
            snap[f"{name}_at_min"] = bool(val <= float(lo) + tol)
    if hasattr(t, "kappa"):
        kappa = float(t.kappa)
        snap["kappa"] = kappa
        snap["kappa_at_cap"] = bool(kappa >= 1.0 - 1e-6)
    return snap


def transition_bound_flags(t) -> dict:
    """At-bound flags for every known parameter, False when the model lacks it.

    Always returning the full set keeps these columns real booleans across models. If
    absent parameters were simply omitted, a DataFrame built from several models would
    fill the gaps with NaN -- which is truthy, so `if row["sigma_arm_at_cap"]` would
    fire for kernels that have no sigma at all.
    """
    snap = transition_param_snapshot(t)
    flags = {}
    for name, (_, min_attr) in _PARAM_BOUND_ATTRS.items():
        flags[f"{name}_at_cap"] = bool(snap.get(f"{name}_at_cap", False))
        if min_attr is not None:
            flags[f"{name}_at_min"] = bool(snap.get(f"{name}_at_min", False))
    flags["kappa_at_cap"] = bool(snap.get("kappa_at_cap", False))
    return flags


def em_diagnostics(
    lls,
    *,
    num_iters: int,
    tolerance: float,
    n_bins: int,
    aligned: bool,
) -> dict:
    """Summarise an EM log-prob trace into convergence diagnostics.

    `aligned=False` marks a trace straight out of `model.fit`, whose last entry is
    computed from expectations taken BEFORE that iteration's M-step and therefore
    lags the returned parameters by one M-step.
    """
    lls = np.asarray(lls, dtype=float)
    n_bins = max(1, int(n_bins))
    n_em_iters = int(lls.size)
    d = np.diff(lls) if lls.size >= 2 else np.array([], dtype=float)

    # ssm appends one entry per iteration on top of the initial log-prob, so a
    # trace of num_iters+1 entries means the tolerance break never fired.
    hit_cap = bool(n_em_iters >= int(num_iters) + 1)
    final_delta = float(d[-1]) if d.size else float("nan")

    # Sign flips in the increments separate a decaying approach to an optimum from
    # a limit cycle -- the M-step is gradient ascent with a fixed step, so it can
    # orbit the optimum forever without the increments ever shrinking.
    sign_flips = int(np.sum(np.sign(d[1:]) * np.sign(d[:-1]) < 0)) if d.size >= 2 else 0
    tail = lls[-5:]

    return {
        "n_em_iters": n_em_iters,
        "em_hit_iter_cap": hit_cap,
        "em_tol_used": float(tolerance),
        "em_tol_per_bin_used": float(tolerance) / n_bins,
        "em_ll_aligned": bool(aligned),
        "em_final_delta": final_delta,
        "em_final_delta_per_bin": final_delta / n_bins if d.size else float("nan"),
        "em_max_decrease": float(min(0.0, d.min())) if d.size else 0.0,
        "em_monotone": bool(d.min() >= -1e-8) if d.size else True,
        "em_n_sign_flips": sign_flips,
        "em_frac_sign_flips": float(sign_flips / (d.size - 1)) if d.size >= 2 else 0.0,
        "em_tail_ll_range": float(tail.max() - tail.min()) if tail.size >= 2 else 0.0,
        "em_converged": bool((not hit_cap) and d.size > 0 and abs(final_delta) < tolerance),
    }


def em_fit_with_diagnostics(
    model,
    datas: list[np.ndarray],
    *,
    num_iters: int,
    tolerance: float,
    n_bins: int,
    trace_params: bool = False,
):
    """Run EM and return (lls, diagnostics, param_trace).

    With `trace_params=False` this is one `model.fit` call plus a summary of the
    returned trace. With `trace_params=True` the same iterations are driven one at
    a time so the parameter values can be recorded per iteration and the log-prob
    trace can be realigned to the returned parameters; that costs one extra
    forward pass per iteration (~2x the E-step work), so it is opt-in.
    """
    if not trace_params:
        lls = np.asarray(
            model.fit(
                datas=datas,
                method="em",
                initialize=False,
                num_iters=int(num_iters),
                tolerance=float(tolerance),
                verbose=0,
            ),
            dtype=float,
        )
        diag = em_diagnostics(
            lls, num_iters=num_iters, tolerance=tolerance, n_bins=n_bins, aligned=False
        )
        return lls, diag, None

    lls: list[float] = []
    ptrace = [transition_param_snapshot(model.transitions)]
    broke_on_tol = False
    for it in range(int(num_iters)):
        # tolerance=0 so the inner call never breaks; the break is decided here on
        # the realigned trace. chunk[0] is log_probability at this iteration's
        # starting parameters, i.e. the result of the previous M-step.
        chunk = np.asarray(
            model.fit(
                datas=datas,
                method="em",
                initialize=False,
                num_iters=1,
                tolerance=0.0,
                verbose=0,
            ),
            dtype=float,
        )
        lls.append(float(chunk[0]))
        ptrace.append(transition_param_snapshot(model.transitions))
        if len(lls) >= 2 and abs(lls[-1] - lls[-2]) < float(tolerance):
            broke_on_tol = True
            break
    lls.append(float(model.log_probability(datas=datas)))
    lls_arr = np.asarray(lls, dtype=float)
    diag = em_diagnostics(
        lls_arr, num_iters=num_iters, tolerance=tolerance, n_bins=n_bins, aligned=True
    )
    # The break above tested the last two *pre-append* entries; em_diagnostics then sees
    # one more entry (the true final log-prob), so its final delta covers a different
    # M-step and can exceed the tolerance even though the loop did stop on it. Report
    # what actually happened rather than re-deriving it from the appended trace.
    diag["em_converged"] = bool(broke_on_tol)
    return lls_arr, diag, ptrace


def em_convergence_report(cv_df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    """One row per model summarising whether its EM fits actually converged."""
    cols = {}
    if "em_hit_iter_cap" in cv_df.columns:
        g = cv_df.groupby("model")
        cols["folds"] = g["fold"].nunique()
        cols["folds_capped"] = g["em_hit_iter_cap"].sum()
        cols["folds_converged"] = g["em_converged"].sum()
        cols["cv_iters_mean"] = g["n_em_iters"].mean()
        cols["cv_worst_decrease"] = g["em_max_decrease"].min()
        cols["cv_frac_sign_flips"] = g["em_frac_sign_flips"].mean()
    report = pd.DataFrame(cols).reset_index() if cols else pd.DataFrame({"model": []})

    keep = [
        "model",
        "n_em_iters",
        "em_hit_iter_cap",
        "em_converged",
        "em_tol_used",
        "em_final_delta_per_bin",
        "em_post_mstep_gain",
        "em_max_decrease",
        "em_frac_sign_flips",
        "em_tail_ll_range",
    ]
    keep += [c for c in fit_df.columns if c.endswith("_at_cap") or c.endswith("_at_min")]
    full = fit_df[[c for c in keep if c in fit_df.columns]].copy()
    full = full.rename(columns={c: f"full_{c}" for c in full.columns if c != "model"})
    if report.empty:
        return full
    return report.merge(full, on="model", how="outer")


def print_em_convergence_report(cv_df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
    report = em_convergence_report(cv_df, fit_df)
    print("\n── EM convergence ─────────────────────────────────────────────")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(report.to_string(index=False))
    def _flags(col):
        return report[col].fillna(False).infer_objects(copy=False).astype(bool)

    if "full_em_hit_iter_cap" in report.columns and bool(_flags("full_em_hit_iter_cap").any()):
        names = ", ".join(report.loc[_flags("full_em_hit_iter_cap"), "model"].astype(str))
        print(f"  ! hit the iteration cap on the full fit: {names}")
        print("    -> raise --num-em-iters, or shrink the M-step (see --mstep-lr).")
    at_cap = [c for c in report.columns if c.endswith("_at_cap") or c.endswith("_at_min")]
    for c in at_cap:
        hits = report.loc[_flags(c), "model"].astype(str).tolist()
        if hits:
            print(f"  ! {c.replace('full_', '')}: {', '.join(hits)} (parameter frozen at its bound)")
    return report
