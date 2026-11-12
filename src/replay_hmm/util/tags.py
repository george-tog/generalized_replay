"""Run-tag tokens: how floats, rat lists and free-form tags are spelled in directory and file names.

`fmt_float_tag` (`.2g`) and `float_to_tag` (`.6g`, integers bare) are different spellings
of the same float, and each already names directories on disk. They are deliberately not
merged: switching a builder from one to the other renames its runs.
"""


def fmt_float_tag(x: float) -> str:
    """Filesystem-safe `.2g` token for a float in run directory names (`-` -> `m`, `.` -> `p`)."""
    return f"{float(x):.2g}".replace("-", "m").replace(".", "p")


def sec_to_ms_tag(sec: float) -> str:
    ms = 1000.0 * float(sec)
    if abs(ms - round(ms)) < 1e-9:
        return str(int(round(ms)))
    return f"{ms:.3f}".rstrip("0").rstrip(".")


def float_to_tag(
    value: float,
    *,
    precision: int = 6,
    replace_negative: bool = False,
) -> str:
    x = float(value)
    if abs(x - round(x)) < 1e-9:
        out = str(int(round(x)))
    else:
        out = f"{x:.{int(precision)}g}".replace(".", "p")
    if replace_negative:
        out = out.replace("-", "m")
    return out


def float_equal(a: float, b: float, atol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= float(atol)


def format_tag8(value: float) -> str:
    return float_to_tag(float(value), precision=8, replace_negative=True)


def sanitize_tag(run_tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_tag)


def pooled_rat_tag(rats: list[str]) -> str:
    seen: set[str] = set()
    uniq: list[str] = []
    for rat in rats:
        name = str(rat).strip()
        key = name.lower()
        if (not name) or (key in seen):
            continue
        seen.add(key)
        uniq.append(name)
    return "-".join(sorted(uniq, key=str.lower)) if len(uniq) > 0 else "rats"
