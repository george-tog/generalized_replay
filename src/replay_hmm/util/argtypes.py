"""argparse `type=` callables shared by the CLIs.

The types that read a module constant (sigma caps, momentum grids, the arm names) live beside
their constants.
"""


def bool_arg(value) -> bool:
    """argparse type for explicit booleans, so they can be set and swept like any value."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


def opt_float_arg(value) -> float | None:
    """argparse type for a float that may be disabled with 'none'."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"none", "off", "null", ""}:
        return None
    return float(s)
