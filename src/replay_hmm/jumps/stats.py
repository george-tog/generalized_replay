"""Jump statistics: directionality, the asymmetry permutation tests and their title text.

`HOVER_BINS` is the hover threshold of `asymmetry_2sided_pvalue`: jumps with
``|jump| <= HOVER_BINS`` are dropped before the test.
"""
import numpy as np

HOVER_BINS = 0

def event_directionality_from_jumps(jumps: np.ndarray) -> tuple[int, int, int, float]:
    vals = np.asarray(jumps, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    nonzero = vals[vals != 0.0]
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    n_nonzero = int(n_pos + n_neg)
    if n_nonzero <= 0:
        return n_pos, n_neg, n_nonzero, float("nan")
    return n_pos, n_neg, n_nonzero, float((n_pos - n_neg) / n_nonzero)


def format_pvalue(
    p_value: float,
    *,
    n_perm: int | None = None,
    mathtext: bool = True,
) -> tuple[str, str]:
    """(relation, text) for a permutation p: one significant digit, exponent always shown.

    A Monte-Carlo p is a mean of ``n_perm`` booleans, so it is exactly 0.0 whenever no
    permutation beats the observation. The test cannot resolve below ``1/n_perm``, so
    that case is reported as the bound it is -- relation ``"<"`` on ``1/n_perm`` -- and
    never as ``p = 0``.

    ``mathtext=True`` returns matplotlib mathtext (``$2\\times10^{-5}$``) for figure text;
    pass ``mathtext=False`` for stdout, where the raw ``$...$`` would show verbatim.
    """
    if not np.isfinite(p_value):
        return "=", "NA"

    relation = "="
    value = float(p_value)
    if value <= 0.0 and n_perm is not None and int(n_perm) > 0:
        relation = "<"
        value = 1.0 / int(n_perm)

    # Round through the format spec so Python carries the mantissa (9.6e-5 -> 1e-04).
    plain = f"{value:.0e}"
    if not mathtext:
        return relation, plain
    mantissa, exponent = plain.split("e")
    return relation, rf"${mantissa}\times10^{{{int(exponent)}}}$"


def asymmetry_title_text(
    jumps: np.ndarray,
    *,
    n_perm: int = 10000,
    seed: int = 42,
    separator: str = "\n ",
    mathtext: bool = True,
) -> str:
    asym, p_value, n_jumps = asymmetry_2sided_pvalue(
        np.asarray(jumps, dtype=float),
        n_perm=n_perm,
        seed=seed,
    )
    rel, p_txt = format_pvalue(p_value, n_perm=n_perm, mathtext=mathtext)
    if np.isfinite(asym) and np.isfinite(p_value):
        return f"N={n_jumps}, asym={asym:.3g}{separator}2-sided p{rel}{p_txt}"
    return f"N={n_jumps}, asymmetry index=NA{separator}2-sided p=NA"


def step_size_title_text(steps: np.ndarray) -> str:
    vals = np.asarray(steps, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "N=0, stay=NA"
    return f"N={vals.size}, stay={100.0 * np.mean(vals == 0):.0f}%"


def asymmetry_2sided_pvalue(jumps: np.ndarray, n_perm: int = 50000, seed: int = 42) -> tuple[float, float, int]:
    jumps = np.asarray(jumps, dtype=float)
    jumps = jumps[np.isfinite(jumps)]
    jumps = jumps[np.abs(jumps) > HOVER_BINS]

    N = int(jumps.size)
    if N == 0:
        return float("nan"), float("nan"), 0


    j = jumps.astype(np.int64) # make ints for np.bincount
    max_k = int(np.abs(j).max(initial=0))
    if max_k < 1:
        return 0.0, 1.0, N

    # d[k-1] = P(+k) - P(-k)
    pos = np.bincount(j[j > 0], minlength=max_k + 1).astype(np.float64) # count number of each of the positive entries
    neg = np.bincount(-j[j < 0], minlength=max_k + 1).astype(np.float64) # count number of each of the negative entries
    n_k = (pos[1:] + neg[1:]).astype(int) # total jumps with |jump|=k
    diff = (pos[1:] - neg[1:]) / N # difference in counts of positive and negative jumps
    w = np.ones_like(n_k, dtype=float)  # every jump size weighted equally

    asym = float(np.sum(w * (diff)))

    rng = np.random.default_rng(seed)
    K = n_k.size
    x = rng.binomial(n=n_k, p=0.5, size=(n_perm, K))
    # permuted n(+k)-n(-k) = 2*x_k - n_k
    diff_perm = (2 * x - n_k) / N
    asym_perm = diff_perm @ w

    p = np.mean(np.abs(asym_perm) >= np.abs(asym))

    return asym, p, N
