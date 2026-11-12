"""Synthetic-recovery generator specs: the parameter grids and random samples a run sweeps, the
fixed-generator replicates, the deterministic shard filter, and applying one spec's parameters
to a generator model.

The three spec factories build every spec with `_generator_spec`.
"""
from __future__ import annotations

import numpy as np

from replay_hmm.hmm import momentum as hmm_order2_momentum


def _filter_generator_specs_for_shard(
    generator_specs: list[dict],
    generator_shard_index: int,
    generator_num_shards: int,
) -> list[dict]:
    """Select deterministic generator specs while preserving original indices."""
    if int(generator_num_shards) <= 1:
        return list(generator_specs)

    return [
        spec
        for spec in generator_specs
        if int(spec["generator_index"]) % int(generator_num_shards) == int(generator_shard_index)
    ]


def _subsample(values, n: int):
    """`n` evenly-spaced entries of `values`, endpoints always kept.

    Used to coarsen momentum's refit grid for recovery. Keeping both endpoints matters: the
    `_at_min` / `_at_cap` edge flags are read off axis positions, so a subsample that dropped
    an end would move the bound the flags report without saying so.
    """
    values = list(values)
    n = int(n)
    if n < 1:
        raise ValueError(f"subsample size must be >= 1; got {n}.")
    if n >= len(values):
        return tuple(values)
    if n == 1:
        return (values[len(values) // 2],)
    idx = np.unique(np.linspace(0, len(values) - 1, n).round().astype(int))
    return tuple(values[i] for i in idx)


def _apply_grid_params_to_generator(hmm, grid_params: dict[str, float]) -> None:
    """Assign grid parameters to a copied generator HMM."""
    # `momentum` FIRST, and returning, because the generic loop below is not merely
    # insufficient for it -- it is silently wrong. `MomentumTransitionParams` really does
    # carry `a`, `sigma_v_cm_s`, `lambda_m` and `sigma_m` as plain floats, so every setattr
    # succeeds and `did_update` goes True; but the model's actual state is `hmm.kernel`,
    # whose per-segment tensors BAKE IN lambda_m, sigma_m and eps, and it has no
    # `_transition_matrix` to invalidate. The generator would report the new parameters as
    # ground truth and go on sampling from the old kernel, with nothing anywhere disagreeing.
    #
    # `set_grid_point` replaces both the kernel and the transitions bag rather than mutating,
    # which is what makes the rebuild total. Same handling, for the same reason, as
    # hmm.restore.apply_transition_params_from_summary_row.
    if hasattr(hmm, "set_grid_point"):
        a, sigma_v = grid_params.get("a"), grid_params.get("sigma_v_cm_s")
        if a is None or sigma_v is None:
            # An empty `params` is the fixed-generator path, where the caller has already
            # applied the fitted parameters; anything else is a spec built for the wrong
            # kernel and must not be silently ignored.
            if grid_params:
                raise ValueError(
                    "A momentum grid spec must carry both 'a' and 'sigma_v_cm_s'; got "
                    f"{sorted(grid_params)}. Without them the kernel would keep the grid "
                    "point it was constructed with while the row reported otherwise."
                )
            return
        hmm.set_grid_point(
            hmm_order2_momentum.MomentumGridPoint.from_a_sigma_v(
                float(a), float(sigma_v), hmm.dt_s
            )
        )
        return

    transition_obj = hmm.transitions
    did_update = False

    for param_name, raw_val in grid_params.items():
        if not hasattr(transition_obj, param_name):
            continue
        val = float(raw_val)
        if not np.isfinite(val):
            continue
        setattr(transition_obj, param_name, val)
        did_update = True

    if did_update and hasattr(transition_obj, "_transition_matrix"):
        transition_obj._transition_matrix = None


# The `grid_*` columns every generator spec carries, in their saved order, and the two that
# only the momentum and segment_uniform specs carry (after them). A column a spec does not set
# is NaN.
_GRID_COLUMNS = (
    "grid_gamma", "grid_kappa", "grid_eps", "grid_omega",
    "grid_sigma_arm", "grid_sigma_wheel", "grid_wheel_ratio",
)
_MOMENTUM_GRID_COLUMNS = ("grid_a", "grid_sigma_v")


def _generator_spec(
    generator_index: int,
    generator_kernel: str,
    params: dict,
    *,
    momentum_columns: bool,
    replicate: int | None = None,
    **grid: float,
) -> dict:
    """One generator spec, as the three spec factories write it: the index and kernel header
    (`grid_index` is the generator index), the `grid_*` columns in their saved order,
    `replicate` for a fixed-generator replicate, then `params`, the values
    `_apply_grid_params_to_generator` sets."""
    columns = _GRID_COLUMNS + (_MOMENTUM_GRID_COLUMNS if momentum_columns else ())
    unknown = sorted(set(grid) - set(columns))
    if unknown:
        raise TypeError(f"unknown grid columns for this spec: {unknown}")
    spec = {
        "generator_index": generator_index,
        "generator_model": generator_kernel,
        "generator_kernel": generator_kernel,
        "grid_index": generator_index,
    }
    for column in columns:
        spec[column] = grid.get(column, np.nan)
    if replicate is not None:
        spec["replicate"] = replicate
    spec["params"] = params
    return spec


def _build_generator_grid_specs(
    grid_generator_kernels: list[str],
    grid_size: int,
    grid_omega_size: int,
    grid_eps_max: float,
    grid_sigma_min: float,
    grid_sigma_max: float,
    momentum_a_span: tuple[float, float] | None = None,
    momentum_sigma_v_span: tuple[float, float] | None = None,
) -> list[dict]:
    """Construct one generator spec per point in the requested parameter grids.

    `momentum_a_span` / `momentum_sigma_v_span` bound momentum's two axes. They are passed in
    rather than derived here because the sigma_v window depends on the session's dt and
    geometry: `template._momentum_spans_and_narrow` takes them from the fit's full `a` grid
    and the momentum model's own resolved sigma_v axis, so a generator can never be drawn
    somewhere its refit could not reach.
    """
    gamma_values = np.linspace(0.0, 0.99, int(grid_size))
    kappa_values = np.linspace(0.0, 0.99, int(grid_size))
    eps_values = np.linspace(0.0, float(grid_eps_max), int(grid_size) // 2)
    omega_values = np.linspace(0.0, 0.99, int(grid_omega_size))
    sigma_values = np.linspace(float(grid_sigma_min), float(grid_sigma_max), int(grid_size))
    # Deliberately LINEAR over the span, while the refit grid is log-spaced -- so a generator
    # lands off the refit grid by construction. That is the intended asymmetry: momentum is
    # the one kernel whose refit can only land on grid points, and drawing generators onto
    # those points would test identifiability while hiding what the grid's resolution costs.
    momentum_a_values = (
        np.linspace(*momentum_a_span, int(grid_size)) if momentum_a_span else None
    )
    momentum_sigma_v_values = (
        np.linspace(*momentum_sigma_v_span, int(grid_size)) if momentum_sigma_v_span else None
    )

    specs: list[dict] = []
    for generator_kernel in grid_generator_kernels:
        generator_kernel = str(generator_kernel)
        # `sticky_TCMSR_sym` is grouped with the SR families, not with `sticky_TCMSR`: its omega
        # is pinned at 1 by both kernel builders (omega_cap=1.0 plus the freeze_omega marker that
        # configure_full_em honours), so it learns the same three parameters they do and takes
        # the same three-parameter sweep. Omega is deliberately absent from `params` --
        # _apply_grid_params_to_generator setattrs whatever it is handed, and the omega setter
        # would clip an explicit value, so leaving it out is what makes true_omega read back
        # the pinned value the refit actually uses.
        if generator_kernel in {"sticky_SR", "sticky_SR_rev", "sticky_TCMSR_sym"}:
            for gamma in gamma_values:
                for kappa in kappa_values:
                    for eps in eps_values:
                        specs.append(
                            _generator_spec(
                                len(specs),
                                generator_kernel,
                                {
                                    "gamma": float(gamma),
                                    "kappa": float(kappa),
                                    "eps": float(eps),
                                },
                                momentum_columns=False,
                                grid_gamma=float(gamma),
                                grid_kappa=float(kappa),
                                grid_eps=float(eps),
                            )
                        )
        elif generator_kernel == "sticky_TCMSR":
            for gamma in gamma_values:
                for kappa in kappa_values:
                    for eps in eps_values:
                        for omega in omega_values:
                            specs.append(
                                _generator_spec(
                                    len(specs),
                                    generator_kernel,
                                    {
                                        "gamma": float(gamma),
                                        "kappa": float(kappa),
                                        "eps": float(eps),
                                        "omega": float(omega),
                                    },
                                    momentum_columns=False,
                                    grid_gamma=float(gamma),
                                    grid_kappa=float(kappa),
                                    grid_eps=float(eps),
                                    grid_omega=float(omega),
                                )
                            )
        elif generator_kernel == "continuous_EM":
            for sigma_arm in sigma_values:
                # sigma_wheel is swept over the same bin range as sigma_arm rather than as
                # a ratio of it: the two widths are independent parameters of the kernel,
                # and a ratio both couples them and (for ratios > 1) pushes sigma_wheel
                # onto the cap, where the logistic is saturated.
                for sigma_wheel in sigma_values:
                    for kappa in kappa_values:
                        for eps in eps_values:
                            specs.append(
                                _generator_spec(
                                    len(specs),
                                    generator_kernel,
                                    {
                                        "sigma_arm": float(sigma_arm),
                                        "sigma_wheel": float(sigma_wheel),
                                        "kappa": float(kappa),
                                        "eps": float(eps),
                                    },
                                    momentum_columns=False,
                                    grid_kappa=float(kappa),
                                    grid_eps=float(eps),
                                    grid_sigma_arm=float(sigma_arm),
                                    grid_sigma_wheel=float(sigma_wheel),
                                    # Kept for schema stability; the widths are swept
                                    # independently, so this is the realised ratio, not a
                                    # control.
                                    grid_wheel_ratio=float(sigma_wheel) / float(sigma_arm),
                                )
                            )
        elif generator_kernel == "momentum":
            if momentum_a_values is None or momentum_sigma_v_values is None:
                raise ValueError(
                    "momentum needs momentum_a_span and momentum_sigma_v_span; the sigma_v "
                    "window depends on dt and geometry, so it cannot be defaulted here."
                )
            for a_val in momentum_a_values:
                for sigma_v in momentum_sigma_v_values:
                    specs.append(
                        _generator_spec(
                            len(specs),
                            generator_kernel,
                            # (a, sigma_v), not (lambda_m, sigma_m): lambda_m enters the
                            # kernel only through a, and sigma_m is not separable from it.
                            # _apply_grid_params_to_generator rebuilds the kernel from these.
                            {
                                "a": float(a_val),
                                "sigma_v_cm_s": float(sigma_v),
                            },
                            momentum_columns=True,
                            grid_a=float(a_val),
                            grid_sigma_v=float(sigma_v),
                        )
                    )
        elif generator_kernel == "segment_uniform":
            # Its whole content is eps and kappa over a fixed uniform base, so these two axes
            # ARE its parameter space -- no third axis exists to sweep.
            for kappa in kappa_values:
                for eps in eps_values:
                    specs.append(
                        _generator_spec(
                            len(specs),
                            generator_kernel,
                            {
                                "kappa": float(kappa),
                                "eps": float(eps),
                            },
                            momentum_columns=True,
                            grid_kappa=float(kappa),
                            grid_eps=float(eps),
                        )
                    )
        else:
            raise ValueError(f"Unsupported grid generator kernel: {generator_kernel}")

    return specs


def _build_generator_random_specs(
    grid_generator_kernels: list[str],
    random_sample_size: int,
    random_sample_seed: int,
    grid_eps_max: float,
    grid_sigma_min: float,
    grid_sigma_max: float,
    momentum_a_span: tuple[float, float] | None = None,
    momentum_sigma_v_span: tuple[float, float] | None = None,
    continuous_wheel_axis: bool = True,
) -> list[dict]:
    """Construct one generator spec per random parameter tuple.

    `momentum_a_span` / `momentum_sigma_v_span` bound momentum's two axes. They are passed in
    rather than derived here because the sigma_v window depends on the session's dt and
    geometry: `template._momentum_spans_and_narrow` takes them from the fit's full `a` grid
    and the momentum model's own resolved sigma_v axis, so a generator can never be drawn
    somewhere its refit could not reach.

    `continuous_wheel_axis=False` is for state spaces with no wheel (the two-map datasets):
    continuous_EM then draws sigma_arm, kappa and eps only, and its sigma_wheel and
    wheel-ratio columns are NaN.
    """
    rng = np.random.default_rng(int(random_sample_seed))

    specs: list[dict] = []
    for generator_kernel in grid_generator_kernels:
        generator_kernel = str(generator_kernel)
        # `sticky_TCMSR_sym` draws the SR families' three parameters, without omega: see the
        # same branch of `_build_generator_grid_specs`.
        for _sample_index in range(int(random_sample_size)):
            if generator_kernel in {"sticky_SR", "sticky_SR_rev", "sticky_TCMSR_sym"}:
                gamma = float(rng.uniform(0.0, 0.99))
                kappa = float(rng.uniform(0.0, 0.99))
                eps = float(rng.uniform(0.0, float(grid_eps_max)))
                specs.append(
                    _generator_spec(
                        len(specs),
                        generator_kernel,
                        {
                            "gamma": gamma,
                            "kappa": kappa,
                            "eps": eps,
                        },
                        momentum_columns=False,
                        grid_gamma=gamma,
                        grid_kappa=kappa,
                        grid_eps=eps,
                    )
                )
            elif generator_kernel == "sticky_TCMSR":
                gamma = float(rng.uniform(0.0, 0.99))
                kappa = float(rng.uniform(0.0, 0.99))
                eps = float(rng.uniform(0.0, float(grid_eps_max)))
                omega = float(rng.uniform(0.0, 0.99))
                specs.append(
                    _generator_spec(
                        len(specs),
                        generator_kernel,
                        {
                            "gamma": gamma,
                            "kappa": kappa,
                            "eps": eps,
                            "omega": omega,
                        },
                        momentum_columns=False,
                        grid_gamma=gamma,
                        grid_kappa=kappa,
                        grid_eps=eps,
                        grid_omega=omega,
                    )
                )
            elif generator_kernel == "continuous_EM":
                sigma_arm = float(rng.uniform(float(grid_sigma_min), float(grid_sigma_max)))
                if not continuous_wheel_axis:
                    kappa = float(rng.uniform(0.0, 0.99))
                    eps = float(rng.uniform(0.0, float(grid_eps_max)))
                    specs.append(
                        _generator_spec(
                            len(specs),
                            generator_kernel,
                            {"sigma_arm": sigma_arm, "kappa": kappa, "eps": eps},
                            momentum_columns=False,
                            grid_kappa=kappa,
                            grid_eps=eps,
                            grid_sigma_arm=sigma_arm,
                        )
                    )
                    continue
                # Drawn over the same bin range as sigma_arm, independently -- see the
                # grid path above.
                sigma_wheel = float(rng.uniform(float(grid_sigma_min), float(grid_sigma_max)))
                kappa = float(rng.uniform(0.0, 0.99))
                eps = float(rng.uniform(0.0, float(grid_eps_max)))
                specs.append(
                    _generator_spec(
                        len(specs),
                        generator_kernel,
                        {
                            "sigma_arm": sigma_arm,
                            "sigma_wheel": sigma_wheel,
                            "kappa": kappa,
                            "eps": eps,
                        },
                        momentum_columns=False,
                        grid_kappa=kappa,
                        grid_eps=eps,
                        grid_sigma_arm=sigma_arm,
                        grid_sigma_wheel=sigma_wheel,
                        # Realised ratio, kept only for schema stability (see grid path).
                        grid_wheel_ratio=sigma_wheel / sigma_arm,
                    )
                )
            elif generator_kernel == "momentum":
                if momentum_a_span is None or momentum_sigma_v_span is None:
                    raise ValueError(
                        "momentum needs momentum_a_span and momentum_sigma_v_span; the "
                        "sigma_v window depends on dt and geometry, so it cannot be "
                        "defaulted here."
                    )
                # Uniform inside the refit grid's span, so it is reachable in RANGE but
                # essentially never on a grid point. The residual between true_a and the
                # recovered a is therefore grid quantization, not estimator bias -- read
                # momentum's parameter-recovery row differently from every other kernel's.
                a_val = float(rng.uniform(*momentum_a_span))
                sigma_v = float(rng.uniform(*momentum_sigma_v_span))
                specs.append(
                    _generator_spec(
                        len(specs),
                        generator_kernel,
                        {"a": a_val, "sigma_v_cm_s": sigma_v},
                        momentum_columns=True,
                        grid_a=a_val,
                        grid_sigma_v=sigma_v,
                    )
                )
            elif generator_kernel == "segment_uniform":
                kappa = float(rng.uniform(0.0, 0.99))
                eps = float(rng.uniform(0.0, float(grid_eps_max)))
                specs.append(
                    _generator_spec(
                        len(specs),
                        generator_kernel,
                        {"kappa": kappa, "eps": eps},
                        momentum_columns=True,
                        grid_kappa=kappa,
                        grid_eps=eps,
                    )
                )
            else:
                raise ValueError(f"Unsupported random generator kernel: {generator_kernel}")

    return specs


def _build_fixed_generator_replicate_specs(
    fixed_generator_kernels: list[str],
    fixed_generator_reps: int,
    start_index: int = 0,
) -> list[dict]:
    """Construct generator specs for repeated samples from fixed fitted models."""
    specs: list[dict] = []
    for generator_kernel in fixed_generator_kernels:
        generator_kernel = str(generator_kernel)
        for replicate_index in range(int(fixed_generator_reps)):
            spec_index = int(start_index) + len(specs)
            specs.append(
                _generator_spec(
                    spec_index,
                    generator_kernel,
                    {},
                    momentum_columns=False,
                    replicate=int(replicate_index),
                )
            )
    return specs
